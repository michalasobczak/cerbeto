"""`cerbeto serve` (spec 0019 G1, N1): the HTTP contract around `Talker`, on a scripted talker and on a tiny student."""
import http.client
import io
import json
import threading

import pytest
import torch

from cerbeto import paths, serve
from cerbeto.serve import Student, make_handler, make_server
from cerbeto.voice import Voice, VoiceConfig

INFO = {"student": "final", "version": "test", "params": 1, "device": "cpu", "k": 4, "temperature": 0.0}


class ScriptedTalker:
    """Talker's interface: answers with how many turns its history holds; can block or fail on request."""

    def __init__(self):
        self.history, self.last_thought, self.last = "", "", None
        self.gate = None                        # threading.Event: reply waits for it
        self.started = threading.Event()
        self.fail = False

    def reply(self, say, remember=None, on_raw=None):
        self.started.set()
        if self.gate:
            self.gate.wait(5)
        if self.fail:
            raise RuntimeError("boom")
        turns = self.history.count("<|human|>") if remember else 0      # as Talker: no history without memory
        if remember:
            self.history += f"<|human|> {say}\n<|student|> ok<|eos|>"
        self.last_thought = "thinking"
        self.last = {"timing": {"tokens": 3}}
        return f"heard {turns}: {say}"


class FakeSocket:
    """One HTTP exchange without a network socket: the handler reads `data` and writes into `out`."""

    def __init__(self, data: bytes):
        self.rfile, self.out = io.BytesIO(data), io.BytesIO()

    def makefile(self, mode, *_args, **_kw):
        return self.rfile if "r" in mode else self.out

    def sendall(self, b):
        self.out.write(b)


HANDLERS = {}           # port → handler class of the fixture


@pytest.fixture
def running(tmp_path):
    """(talker, port): the server's handler with a scripted talker; voice programs point at nothing.
    Requests go through the real handler but not through a socket (sandboxes may forbid binding)."""
    talker = ScriptedTalker()
    none = str(tmp_path / "missing")
    HANDLERS[serve.PORT] = make_handler(Student(talker, INFO), Voice(VoiceConfig(none, none, none, none)), serve.PORT)
    yield talker, serve.PORT


def call(port, method, path, body=None, headers=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else b"")
    head = {"Host": f"127.0.0.1:{port}", "Content-Type": "application/json", "Content-Length": str(len(data)), **(headers or {})}
    request = f"{method} {path} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in head.items()) + "\r\n"
    sock = FakeSocket(request.encode() + data)
    HANDLERS[port](sock, ("127.0.0.1", 50000), None)
    r = http.client.HTTPResponse(FakeSocket(sock.out.getvalue()))
    r.begin()
    payload = r.read()
    kind = r.getheader("Content-Type") or ""
    return r.status, (json.loads(payload) if kind.startswith("application/json") else payload)


def ask(port, text, cid="c1", uid="u1", memory=False):
    return call(port, "POST", "/v1/reply", {"conversation_id": cid, "utterance_id": uid, "text": text, "memory": memory})


def test_health_and_reply_follow_the_contract(running):
    _, port = running
    code, health = call(port, "GET", "/v1/health")
    assert code == 200 and health["ready"] is True and health["busy"] is False
    assert {"student", "version", "params", "device", "k", "temperature"} <= set(health)
    assert health["voice"] == {"listening": False, "speaking": False}
    code, out = ask(port, "  What is the  opposite of hot? ", uid="u-7")
    assert code == 200 and set(out) == {"utterance_id", "answer", "thought", "seconds", "tokens"}
    assert out["utterance_id"] == "u-7" and out["answer"] == "heard 0: What is the opposite of hot?"
    assert out["thought"] == "thinking" and out["tokens"] == 3 and out["seconds"] >= 0


@pytest.mark.parametrize("body", [
    {"conversation_id": "c", "utterance_id": "u", "text": ""},
    {"conversation_id": "c", "utterance_id": "u", "text": "   "},
    {"conversation_id": "c", "utterance_id": "u", "text": "x" * 501},
    {"conversation_id": "c", "utterance_id": "u", "text": 5},
    {"utterance_id": "u", "text": "hi"},
    {"conversation_id": "c", "text": "hi"},
    {"conversation_id": "c", "utterance_id": "u", "text": "hi", "memory": "yes"},
    ["not", "an", "object"],
])
def test_bad_requests_get_400(running, body):
    _, port = running
    assert call(port, "POST", "/v1/reply", body)[0] == 400


def test_bad_json_and_unknown_paths(running):
    _, port = running
    assert call(port, "POST", "/v1/reply", raw=b"{not json")[0] == 400
    assert call(port, "POST", "/v1/reset", {})[0] == 400
    assert call(port, "GET", "/v1/nothing")[0] == 404
    assert ask(port, "x" * 500)[0] == 200                 # the limit itself is allowed


def test_busy_gets_409_at_once_and_is_not_queued(running):
    talker, port = running
    talker.gate = threading.Event()
    first = {}
    t = threading.Thread(target=lambda: first.update(r=ask(port, "first")))
    t.start()
    assert talker.started.wait(5)
    assert call(port, "GET", "/v1/health")[1]["busy"] is True
    assert ask(port, "second", uid="u2") == (409, {"error": "busy"})
    talker.gate.set()
    t.join(5)
    assert first["r"][0] == 200 and first["r"][1]["answer"] == "heard 0: first"
    assert call(port, "GET", "/v1/health")[1]["busy"] is False


def test_an_exception_is_500_and_the_next_reply_works(running):
    talker, port = running
    talker.fail = True
    code, out = ask(port, "hi")
    assert code == 500 and "boom" in out["error"]
    talker.fail = False
    assert ask(port, "hi")[0] == 200


def test_histories_of_two_conversations_do_not_mix_and_reset_forgets(running):
    _, port = running
    assert ask(port, "a1", cid="A", memory=True)[1]["answer"] == "heard 0: a1"
    assert ask(port, "a2", cid="A", memory=True)[1]["answer"] == "heard 1: a2"
    assert ask(port, "b1", cid="B", memory=True)[1]["answer"] == "heard 0: b1"
    assert ask(port, "a3", cid="A", memory=True)[1]["answer"] == "heard 2: a3"
    assert ask(port, "a4", cid="A")[1]["answer"] == "heard 0: a4"            # memory off: no history (as `cerbeto chat`)
    assert call(port, "POST", "/v1/reset", {"conversation_id": "A"}) == (200, {"ok": True})
    assert ask(port, "a5", cid="A", memory=True)[1]["answer"] == "heard 0: a5"
    assert ask(port, "b2", cid="B", memory=True)[1]["answer"] == "heard 1: b2"


def test_only_loopback_and_no_host_option(running):
    """N1: bound to 127.0.0.1, `serve` has no --host; other Host or Origin headers are refused (DNS rebinding, CSRF)."""
    _, port = running
    assert call(port, "GET", "/v1/health", headers={"Host": "evil.example"})[0] == 403
    assert call(port, "POST", "/v1/reply", {"conversation_id": "c", "utterance_id": "u", "text": "hi"},
                headers={"Origin": "http://evil.example"})[0] == 403
    assert call(port, "GET", "/v1/health", headers={"Origin": f"http://localhost:{port}"})[0] == 200
    import inspect
    assert "host" not in inspect.signature(serve.run).parameters

def test_serve_has_no_host_option(monkeypatch, capsys):
    from cerbeto import cli
    monkeypatch.setattr("sys.argv", ["cerbeto", "serve", "--help"])
    with pytest.raises(SystemExit):
        cli.main()
    help_text = capsys.readouterr().out
    assert "--port" in help_text and "--checkpoint" in help_text and "--host" not in help_text


def test_the_panel_page_is_served_and_never_speaks_the_thought(running):
    _, port = running
    code, page = call(port, "GET", "/")
    assert code == 200 and b"/v1/reply" in page and b"/voice/transcribe" in page and b"/voice/speak" in page
    # N4: what goes to speech is built from the answer only
    assert b"const parts = sentences(answer)" in page and b"sentences(data.thought" not in page
    for state in (b'"offline"', b'"ready"', b'"listening"', b'"processing"', b'"speaking"', b'"error"'):
        assert state in page


def test_voice_endpoints_say_when_programs_are_missing(running):
    _, port = running
    code, out = call(port, "POST", "/voice/speak", {"text": "Hello."})
    assert code == 503 and "not installed" in out["error"]
    code, out = call(port, "POST", "/voice/transcribe", raw=b"RIFF0000WAVE", headers={"Content-Type": "audio/wav"})
    assert code == 503 and "not installed" in out["error"]


def test_a_real_socket_on_loopback_only():
    """N1 with a real socket: bound to 127.0.0.1, health answers over TCP."""
    try:
        server = make_server(Student(ScriptedTalker(), INFO), Voice(), port=0)
    except PermissionError:
        pytest.skip("binding a local port is not permitted here (sandbox)")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        host, port = server.server_address
        assert host == "127.0.0.1"
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("GET", "/v1/health")
        r = conn.getresponse()
        assert r.status == 200 and json.loads(r.read())["ready"] is True
        conn.close()
    finally:
        server.shutdown()
        server.server_close()


needs_tokenizer = pytest.mark.skipif(not paths.TOKENIZER_FILE.exists(), reason="tokenizer not trained")


@needs_tokenizer
def test_a_real_talker_on_a_tiny_student(monkeypatch):
    """The server around the real `Talker`: one-line answer, thought apart, history only with memory."""
    from cerbeto.model.student import Student as Model, StudentConfig
    from cerbeto.talk import Talker
    from cerbeto.tokenizer import load
    tok = load()
    script = tok.encode("<|think|> hot is warm<|/think|><|answer|> Cold.\nmore<|eos|> never").ids
    start = {}

    def scripted_forward(self, idx, k=4):        # always puts the next scripted token on top (as test_talk)
        n = start.setdefault("n", idx.shape[1])
        logits = torch.zeros(1, idx.shape[1], self.cfg.vocab_size)
        logits[0, -1, script[idx.shape[1] - n]] = 1.0
        return logits

    monkeypatch.setattr(Model, "forward", scripted_forward)
    cfg = StudentConfig(vocab_size=tok.get_vocab_size(), d_model=32, n_heads=2, head_dim=16, mlp_hidden=64,
                        n_prelude=1, n_core=1, n_coda=1, max_seq_len=128)
    s = Student(Talker(Model(cfg, seed=0), torch.device("cpu"), temperature=0.0), INFO)
    out = s.reply("c", "u", "What is the opposite of hot?")
    assert out["answer"] == "Cold." and out["thought"] == "hot is warm"
    newline = next(i for i in range(1, len(script) + 1) if "\n" in tok.decode(script[:i], skip_special_tokens=False))
    assert out["tokens"] == newline < script.index(tok.token_to_id("<|eos|>")) + 1   # stopped after "Cold.\n", not 60
    assert s.histories == {} and s.talker.history == ""
    start.clear()
    s.reply("c", "u", "And of big?", memory=True)
    assert s.histories["c"].startswith("<|human|> And of big?") and s.talker.history == ""


@pytest.mark.parametrize("raw, done", [
    ("<|answer|> Cold.", False),
    ("<|answer|> Cold.\n", True),
    ("<|answer|> \n", False),                                  # an empty first line is not an answer yet
    ("<|think|> hot is\nwarm", False),                         # a newline inside the thought is not the end
    ("<|think|> hot\nis warm<|/think|> Cold.\nmore", True),
    ("<|think|> x<|/think|><|answer|> Cold.\nmore", True),
    ("I can see you be it.\nounting", True),                  # no markers at all
])
def test_answer_line_done(raw, done):
    assert serve.answer_line_done(raw) is done


@needs_tokenizer
def test_generation_stops_after_the_first_answer_line(monkeypatch):
    """Step 4: the panel shows only the first line, so the student stops there instead of running 60 tokens."""
    from cerbeto.model.student import Student as Model, StudentConfig
    from cerbeto.talk import Talker
    from cerbeto.tokenizer import load
    tok = load()
    script = tok.encode("<|answer|> Five.\nounting is the number that is a number that is a number<|eos|>").ids
    start = {}

    def scripted_forward(self, idx, k=4):
        n = start.setdefault("n", idx.shape[1])
        logits = torch.zeros(1, idx.shape[1], self.cfg.vocab_size)
        logits[0, -1, script[min(idx.shape[1] - n, len(script) - 1)]] = 1.0
        return logits

    monkeypatch.setattr(Model, "forward", scripted_forward)
    cfg = StudentConfig(vocab_size=tok.get_vocab_size(), d_model=32, n_heads=2, head_dim=16, mlp_hidden=64,
                        n_prelude=1, n_core=1, n_coda=1, max_seq_len=128)
    s = Student(Talker(Model(cfg, seed=0), torch.device("cpu"), temperature=0.0), INFO)
    out = s.reply("c", "u", "What is two plus three?")
    newline = next(i for i in range(1, len(script) + 1) if "\n" in tok.decode(script[:i], skip_special_tokens=False))
    assert out["answer"] == "Five." and out["tokens"] == newline < len(script) - 1
