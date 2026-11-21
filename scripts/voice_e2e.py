"""End-to-end check of the voice panel's server side (spec 0019, G3/G4): a recorded question → local Whisper → the
student → the answer → local Piper, through the real `cerbeto serve` handler, with timings of every stage.

No port is opened: requests go through the same handler class the server uses, over an in-memory socket, so it also
runs where binding is not allowed. The "admin's" questions are synthesized by Piper in a *different* voice than the
one that answers, then down-sampled to 16 kHz mono exactly as the panel's `wav()` does it.

Run from brain/:
    uv run python ../scripts/voice_e2e.py --checkpoint <folder with architecture.json + model.safetensors> \
        --admin-voice <some other Piper voice .onnx>
with CERBETO_WHISPER_MODEL / CERBETO_PIPER_BIN / CERBETO_PIPER_VOICE pointing at the local programs and weights
(defaults: ../agentaro/storage/voice). Prints a table and writes JSON with --out.
"""
import argparse
import http.client
import io
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

from cerbeto import serve
from cerbeto.serve import Student, make_handler
from cerbeto.voice import Voice

QUESTIONS = [
    "What is the opposite of hot?",
    "What is the capital of France?",
    "How many legs does a spider have?",
    "What colour is the sky on a clear day?",
    "What do bees make?",
    "Is ice cold or hot?",
    "What is two plus three?",
    "What do you call a baby cat?",
    "Which animal says moo?",
    "What is the opposite of big?",
]
FOLLOW_UPS = ["What is the opposite of up?", "And the opposite of left?", "Thank you, Cerbeto. Goodbye."]


class FakeSocket:
    def __init__(self, data: bytes):
        self.rfile, self.out = io.BytesIO(data), io.BytesIO()

    def makefile(self, mode, *_a, **_k):
        return self.rfile if "r" in mode else self.out

    def sendall(self, b):
        self.out.write(b)


class Client:
    """HTTP/1.1 requests through the real handler class (Host/Origin as the panel's browser sends them)."""

    def __init__(self, handler, port):
        self.handler, self.port = handler, port

    def call(self, method, path, body=None, raw=None, kind="application/json", origin=True):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else b"")
        head = {"Host": f"127.0.0.1:{self.port}", "Content-Type": kind, "Content-Length": str(len(data))}
        if origin and method == "POST":
            head["Origin"] = f"http://127.0.0.1:{self.port}"
        req = f"{method} {path} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in head.items()) + "\r\n"
        sock = FakeSocket(req.encode() + data)
        self.handler(sock, ("127.0.0.1", 50000), None)
        r = http.client.HTTPResponse(FakeSocket(sock.out.getvalue()))
        r.begin()
        payload = r.read()
        return r.status, (json.loads(payload) if (r.getheader("Content-Type") or "").startswith("application/json")
                          else payload)


def panel_wav(samples: np.ndarray, rate: int) -> bytes:
    """The panel's wav(): box-average down to 16 kHz, 16-bit mono."""
    ratio = rate / 16000
    n = int(len(samples) // ratio)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        a, b = int(i * ratio), min(len(samples), int((i + 1) * ratio))
        out[i] = samples[a:b].mean() if b > a else 0.0
    pcm = np.clip(out, -1, 1)
    pcm = np.where(pcm < 0, pcm * 0x8000, pcm * 0x7FFF).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000), w.writeframes(pcm.tobytes())
    return buf.getvalue()


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data)) as w:
        rate, frames = w.getframerate(), w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768, rate


def recording(piper: str, voice: str, text: str, pad_s: float = 0.3) -> bytes:
    """The admin's question, as the panel would upload it: Piper (other voice) + a little silence on both sides."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "q.wav"
        subprocess.run([piper, "-m", voice, "-f", str(path)], input=text.encode(), check=True, capture_output=True,
                       env={**{k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR") if k in os.environ},
                            "ORT_DISABLE_TELEMETRY": "1"})
        s, rate = read_wav(path.read_bytes())
    pad = np.zeros(int(rate * pad_s), dtype=np.float32)
    return panel_wav(np.concatenate([pad, s, pad]), rate)


def sentences(text: str) -> list[str]:
    """The panel's sentences() (without the >500-character split, never hit by one-line answers)."""
    return [t.strip() for t in re.split(r"(?<=[.!?…:;])\s+|\n+", text) if t.strip()]


def words(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower().replace("colour", "color")).split()


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


def turn(c: Client, audio: bytes, cid: str, uid: str, memory: bool, expected: str) -> dict:
    """One spoken turn, timed the way the panel times it (end of speech = upload starts)."""
    t_end = time.perf_counter()
    code, stt = c.call("POST", "/voice/transcribe", raw=audio, kind="audio/wav")
    assert code == 200, (code, stt)
    t_stt = time.perf_counter()
    code, out = c.call("POST", "/v1/reply", {"conversation_id": cid, "utterance_id": uid, "text": stt["text"],
                                              "memory": memory})
    assert code == 200 and out["utterance_id"] == uid, (code, out)
    t_model = time.perf_counter()
    parts = sentences(out["answer"])
    first_sound, spoken = None, []
    for i, p in enumerate(parts):     # the panel requests all sentences at once; the first one gates the first sound
        code, wav_bytes = c.call("POST", "/voice/speak", {"text": p})
        assert code == 200 and wav_bytes[:4] == b"RIFF", (code, wav_bytes[:80])
        if i == 0:
            first_sound = time.perf_counter()
        s, rate = read_wav(wav_bytes)
        spoken.append(len(s) / rate)
    return {"expected": expected, "heard": stt["text"], "wer": round(wer(expected, stt["text"]), 3),
            "answer": out["answer"], "thought": out["thought"], "tokens": out["tokens"],
            "audio_s": round(len(audio) / 32000, 2), "stt_s": round(t_stt - t_end, 3),
            "model_s": out["seconds"], "model_http_s": round(t_model - t_stt, 3),
            "tts_first_s": round((first_sound or t_model) - t_model, 3),
            "first_sound_s": round((first_sound or t_model) - t_end, 3), "speech_s": round(sum(spoken), 2),
            "sentences": len(parts)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--admin-voice", required=True, help="a Piper voice for the test questions (not the answer voice)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out")
    a = ap.parse_args()

    connects = []                                       # every socket connect from this process (N2)
    sys.addaudithook(lambda ev, args: connects.append(repr(args[1])) if ev == "socket.connect" else None)

    from cerbeto.talk import Talker, load_checkpoint
    t0 = time.perf_counter()
    model, version, dev = load_checkpoint(a.checkpoint, a.device)
    talker = Talker(model, dev, 4, 0.0)
    load_s = time.perf_counter() - t0
    voice = Voice()
    info = {"student": "checkpoint", "version": version, "params": model.num_params(), "device": str(dev), "k": 4,
            "temperature": 0.0}
    student = Student(talker, info)
    c = Client(make_handler(student, voice, serve.PORT), serve.PORT)
    report = {"load_s": round(load_s, 2), "version": version, "params": info["params"]}

    code, health = c.call("GET", "/v1/health")
    assert code == 200 and health["voice"] == {"listening": True, "speaking": True}, health
    report["health"] = health
    code, page = c.call("GET", "/")
    report["page_external_urls"] = sorted(set(re.findall(rb"(?:src|href)=\"(https?://[^\"]+)", page)))

    # warm-up (first Whisper / Piper run loads weights from disk)
    c.call("POST", "/voice/speak", {"text": "Ready."})
    c.call("POST", "/voice/transcribe", raw=recording(voice.config.piper_bin, a.admin_voice, "Hello."), kind="audio/wav")

    print(f"{version}: {info['params'] / 1e6:.0f}M parameters, loaded in {load_s:.2f} s on {dev}")
    turns = []
    for i, q in enumerate(QUESTIONS):
        r = turn(c, recording(voice.config.piper_bin, a.admin_voice, q), "single", f"q{i}", False, q)
        turns.append(r)
        print(f"  [{i + 1:2}] heard {r['heard']!r} (WER {r['wer']:.2f}) → {r['answer']!r}  "
              f"STT {r['stt_s']:.2f} s · model {r['model_s']:.2f} s · TTS {r['tts_first_s']:.2f} s · "
              f"first sound {r['first_sound_s']:.2f} s")
    report["turns"] = turns

    conv = []
    for i, q in enumerate(FOLLOW_UPS):                  # several turns of one conversation with memory on
        conv.append(turn(c, recording(voice.config.piper_bin, a.admin_voice, q), "memory", f"m{i}", True, q))
        print(f"  [mem {i + 1}] {conv[-1]['heard']!r} → {conv[-1]['answer']!r}")
    report["conversation"] = conv
    report["history_kept"] = "memory" in student.histories
    c.call("POST", "/v1/reset", {"conversation_id": "memory"})
    report["history_after_reset"] = "memory" in student.histories

    # a long utterance (~15 s): the shortened audio context must still cover all of it
    long_q = ("I would like to ask you something about the weather. Yesterday it was raining all day in the city, "
              "and today the sun is shining, but the wind is still very strong. Tell me, is it a good idea to go for "
              "a walk in the park this afternoon, or should I stay at home and read a book?")
    audio = recording(voice.config.piper_bin, a.admin_voice, long_q)
    t = time.perf_counter()
    heard = c.call("POST", "/voice/transcribe", raw=audio, kind="audio/wav")[1]["text"]
    report["long"] = {"audio_s": round(len(audio) / 32000, 1), "wer": round(wer(long_q, heard), 3),
                      "stt_s": round(time.perf_counter() - t, 2), "heard": heard}

    # silence and noise: the panel must get "" (→ "didn't catch that"), nothing goes to the model
    rng = np.random.default_rng(0)
    silence = panel_wav(np.zeros(16000 * 2, dtype=np.float32), 16000)
    noise = panel_wav((rng.standard_normal(16000 * 2) * 0.02).astype(np.float32), 16000)
    report["silence"] = c.call("POST", "/voice/transcribe", raw=silence, kind="audio/wav")[1]
    report["noise"] = c.call("POST", "/voice/transcribe", raw=noise, kind="audio/wav")[1]

    # a second question while the student answers: 409 at once, not queued
    gate, entered = threading.Event(), threading.Event()
    real = talker.reply

    def slow(*args, **kw):
        entered.set()
        gate.wait(10)
        return real(*args, **kw)

    talker.reply = slow
    first = {}
    th = threading.Thread(target=lambda: first.update(r=c.call("POST", "/v1/reply", {
        "conversation_id": "x", "utterance_id": "a", "text": "What is the opposite of hot?"})))
    th.start()
    entered.wait(10)
    t = time.perf_counter()
    second = c.call("POST", "/v1/reply", {"conversation_id": "x", "utterance_id": "b", "text": "Hello?"})
    report["busy"] = {"code": second[0], "seconds": round(time.perf_counter() - t, 4)}
    gate.set()
    th.join(30)
    talker.reply = real
    report["busy"]["first_code"] = first["r"][0]

    report["non_loopback_connects"] = [x for x in connects if "127.0.0.1" not in x and "localhost" not in x]

    def med(key, rows=turns):
        return round(statistics.median(r[key] for r in rows), 3)

    report["summary"] = {
        "turns": len(turns), "transcripts_exact": sum(r["wer"] == 0 for r in turns),
        "mean_wer": round(statistics.mean(r["wer"] for r in turns), 3),
        "stt_median_s": med("stt_s"), "model_median_s": med("model_s"), "tts_first_median_s": med("tts_first_s"),
        "first_sound_median_s": med("first_sound_s"), "first_sound_max_s": max(r["first_sound_s"] for r in turns),
    }
    print(json.dumps({k: report[k] for k in ("summary", "long", "silence", "noise", "busy", "history_kept",
                                             "history_after_reset", "non_loopback_connects", "page_external_urls")},
                     indent=1, default=str))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
