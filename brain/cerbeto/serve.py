"""`cerbeto serve` (spec 0019): the student behind a small local HTTP server, and the voice panel that talks to it.

The checkpoint is loaded once, before the server listens. Replies go through one lock — one generation at a time; a
second request while the student is answering gets 409 at once (no queue). Histories are kept per `conversation_id`,
in memory only. The server listens on 127.0.0.1 only (there is no `--host`), checks the Host and Origin headers so a
web page elsewhere cannot use it, and never writes the admin's words to disk: stdout gets only counts and timings.

    GET  /                  the voice panel (voice_panel.html)
    GET  /v1/health         {"ready", "student", "version", "params", "device", "k", "temperature", "busy", "voice"}
    POST /v1/reply          {"conversation_id", "utterance_id", "text", "memory"} → {"utterance_id", "answer", "thought", "seconds", "tokens"}
    POST /v1/reset          {"conversation_id"} → {"ok": true}
    POST /voice/transcribe  audio/wav → {"text"}            (local Whisper, English)
    POST /voice/speak       {"text"} → audio/wav             (local Piper, English voice)
"""
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from cerbeto.voice import MAX_AUDIO, Voice, VoiceError

HOST = "127.0.0.1"
PORT = 8766
PAGE = Path(__file__).with_name("voice_panel.html")
MAX_TEXT = 500                  # characters of one utterance
MAX_JSON = 16 * 1024            # bytes of a JSON request
MAX_ID = 100                    # characters of a conversation / utterance id
MAX_CONVERSATIONS = 100         # the oldest history is forgotten beyond this


def answer_line_done(raw: str) -> bool:
    """True once the visible answer has a complete first line: `Talker.parts` keeps only that line, so whatever the
    student writes after it is thrown away (v6 on CPU: "What is 2 plus 3?" ran 60 tokens, 3.4 s, for a 9-token answer)."""
    if "<|answer|>" in raw:
        answer = raw.rsplit("<|answer|>", 1)[1]
    elif "<|think|>" in raw and "<|/think|>" not in raw:
        return False                                    # still thinking: a newline in the thought is not the end
    else:
        answer = re.sub(r"<\|think\|>.*?<\|/think\|>", "", raw, flags=re.S)
    line, newline, _ = re.sub(r"<\|[^|]*\|>", "", answer).lstrip().partition("\n")
    return bool(newline and line.strip())


class Busy(Exception):
    pass


class BadRequest(Exception):
    pass


class Student:
    """The one `Talker` with a history per conversation, swapped in under the lock."""

    def __init__(self, talker, info: dict):
        self.talker, self.info = talker, info
        self.lock = threading.Lock()
        self.histories: dict[str, str] = {}

    @property
    def busy(self) -> bool:
        return self.lock.locked()

    def reply(self, conversation_id: str, utterance_id: str, text: str, memory: bool = False) -> dict:
        if not self.lock.acquire(blocking=False):
            raise Busy
        try:
            t = self.talker
            t.history = self.histories.get(conversation_id, "")
            t0 = time.perf_counter()
            try:
                # on_raw selects Talker's token-by-token path, which stops at <|eos|>; model.generate would always
                # run all max_new_tokens (v6 on CPU: ~3.4 s instead of ~0.4 s for a short answer). It also stops at
                # the end of the answer's first line, the only line the panel shows.
                answer = t.reply(text, remember=memory, on_raw=answer_line_done)
            finally:
                if memory:
                    self.histories.pop(conversation_id, None)
                    self.histories[conversation_id] = t.history          # newest last
                    while len(self.histories) > MAX_CONVERSATIONS:
                        self.histories.pop(next(iter(self.histories)))
                t.history = ""
            seconds = time.perf_counter() - t0
            tokens = int(((t.last or {}).get("timing") or {}).get("tokens", 0))
            return {"utterance_id": utterance_id, "answer": answer, "thought": t.last_thought,
                    "seconds": round(seconds, 3), "tokens": tokens}
        finally:
            self.lock.release()

    def reset(self, conversation_id: str) -> None:
        with self.lock:
            self.histories.pop(conversation_id, None)


def _id(body: dict, key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_ID:
        raise BadRequest(f"{key} must be a non-empty string of at most {MAX_ID} characters")
    return value


def parse_reply(body) -> tuple[str, str, str, bool]:
    if not isinstance(body, dict):
        raise BadRequest("expected a JSON object")
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise BadRequest("text must not be empty")
    if len(text) > MAX_TEXT:
        raise BadRequest(f"text is longer than {MAX_TEXT} characters")
    memory = body.get("memory", False)
    if not isinstance(memory, bool):
        raise BadRequest("memory must be true or false")
    return _id(body, "conversation_id"), _id(body, "utterance_id"), " ".join(text.split()), memory


def log(event: str, **numbers) -> None:
    """Counts and timings only — never the admin's words or the answer."""
    print(f"{time.strftime('%H:%M:%S')} {event} " + " ".join(f"{k}={v}" for k, v in numbers.items()), flush=True)


def make_handler(student: Student, voice: Voice, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    allowed_origins = {f"http://{h}" for h in allowed_hosts}

    class Handler(BaseHTTPRequestHandler):
        server_version = "cerbeto-serve"

        def log_message(self, *args):        # quiet: request lines are not logged
            pass

        def _send(self, body: bytes, kind: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: dict, code: int = 200) -> None:
            self._send(json.dumps(data).encode(), "application/json; charset=utf-8", code)

        def _allowed(self) -> bool:
            """Loopback only, also against DNS rebinding and against other web pages posting here."""
            if self.headers.get("Host") not in allowed_hosts:
                self._json({"error": "forbidden host"}, 403)
                return False
            origin = self.headers.get("Origin")
            if origin is not None and origin not in allowed_origins:
                self._json({"error": "forbidden origin"}, 403)
                return False
            return True

        def _body(self, limit: int) -> bytes:
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                raise BadRequest("bad Content-Length") from None
            if n > limit:
                raise BadRequest("request too large")
            return self.rfile.read(n) if n > 0 else b""

        def _json_body(self):
            try:
                return json.loads(self._body(MAX_JSON).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise BadRequest("bad JSON") from None

        def do_GET(self):
            if not self._allowed():
                return
            if self.path in ("/", "/index.html"):
                return self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            if self.path == "/v1/health":
                return self._json({"ready": True, **student.info, "busy": student.busy, "voice": voice.status()})
            return self._json({"error": "not found"}, 404)

        def do_POST(self):
            if not self._allowed():
                return
            try:
                if self.path == "/v1/reply":
                    cid, uid, text, memory = parse_reply(self._json_body())
                    out = student.reply(cid, uid, text, memory)
                    log("reply", chars=len(text), tokens=out["tokens"], seconds=out["seconds"], memory=int(memory))
                    return self._json(out)
                if self.path == "/v1/reset":
                    body = self._json_body()
                    student.reset(_id(body if isinstance(body, dict) else {}, "conversation_id"))
                    return self._json({"ok": True})
                if self.path == "/voice/transcribe":
                    t0 = time.perf_counter()
                    text = voice.transcribe(self._body(MAX_AUDIO))
                    seconds = round(time.perf_counter() - t0, 3)
                    log("transcribe", chars=len(text), seconds=seconds)
                    return self._json({"text": text, "seconds": seconds})
                if self.path == "/voice/speak":
                    body = self._json_body()
                    text = body.get("text") if isinstance(body, dict) else None
                    if not isinstance(text, str):
                        raise BadRequest("text must be a string")
                    t0 = time.perf_counter()
                    audio = voice.speak(text)
                    log("speak", chars=len(text), seconds=round(time.perf_counter() - t0, 3))
                    return self._send(audio, "audio/wav")
                return self._json({"error": "not found"}, 404)
            except BadRequest as e:
                return self._json({"error": str(e)}, 400)
            except Busy:
                return self._json({"error": "busy"}, 409)
            except VoiceError as e:
                return self._json({"error": str(e)}, 503)
            except Exception as e:      # the server keeps running; the lock is already released
                log("error", path=self.path, kind=type(e).__name__)
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return Handler


def make_server(student: Student, voice: Voice, port: int = PORT) -> ThreadingHTTPServer:
    """Bound to 127.0.0.1 only; port 0 picks a free port (tests)."""
    server = ThreadingHTTPServer((HOST, port), None)
    server.RequestHandlerClass = make_handler(student, voice, server.server_address[1])
    server.daemon_threads = True
    return server


def run(student: str, version: str | None, checkpoint: str | None, device: str | None, k: int, temperature: float,
        port: int = PORT) -> None:
    from cerbeto.talk import Talker, load_student, load_checkpoint
    if checkpoint:
        model, version, dev = load_checkpoint(checkpoint, device)
        student = "checkpoint"
    else:
        model, version, dev = load_student(student, version, device)
    talker = Talker(model, dev, k, temperature)
    info = {"student": student, "version": version, "params": model.num_params(), "device": str(dev), "k": k,
            "temperature": temperature}
    voice = Voice()
    server = make_server(Student(talker, info), voice, port)
    status = voice.status()
    print(f"cerbeto serve: {student}/{version} · {info['params']:,} parameters · {dev} · K={k} · T={temperature}")
    print(f"  voice: speech recognition {'ready' if status['listening'] else 'MISSING'} ({voice.config.whisper_model}), "
          f"voice {'ready' if status['speaking'] else 'MISSING'} ({voice.config.piper_voice})")
    print(f"  panel on http://{HOST}:{server.server_address[1]}  (loopback only; Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
