"""`cerbeto chat`: talk to a student checkpoint in the terminal and probe what it can do.

The conversation is kept as context in the chat format (`<|human|> …` / `<|student|> …`), the same one
the nursery speech shelf and the exams use. Slash commands probe single skills without the history.
"""
import re
import time
from pathlib import Path

import torch

from cerbeto import paths
from cerbeto.model import checkpoint
from cerbeto.model.student import Student, pick_device
from cerbeto.talk_stats import analyse, layer_names, render
from cerbeto.tokenizer import load as load_tokenizer

HELP = """commands:
  /story <beginning>    continue a story (no chat history)
  /ask <n> <question>   ask the same question n times — is the answer stable?
  /think <question>     answer with K = 1, 2, 4, 8 passes of the recurrent core (greedy)
  /k <n>                passes of the recurrent core for chat ("think longer"), now {k}
  /temp <t>             sampling temperature, 0 = always the most likely token, now {temperature}
  /reset                forget the conversation
  /stats [on|off]       inference statistics of the last reply: speed, certainty per token, attention per block
                        (on: after every reply)
  /info                 which student is loaded
  /help  /quit"""


class Talker:
    def __init__(self, model: Student, device, k: int = 4, temperature: float = 0.7, max_new_tokens: int = 60):
        self.model, self.device = model.eval(), device
        self.k, self.temperature, self.max_new_tokens = k, temperature, max_new_tokens
        self.tok = load_tokenizer()
        self.eos = self.tok.token_to_id("<|eos|>")
        self.stops = {self.tok.token_to_id(t) for t in ("<|eos|>", "<|human|>", "<|student|>")}
        self.history = ""
        self.last_thought = ""          # what the student thought (<|think|> … <|/think|>) before its last reply
        self.last = None                # the last exchange, for /stats

    @torch.no_grad()
    def _continue(self, text: str, k: int, temperature: float, max_new_tokens: int, stops: set[int]) -> str:
        ids = [self.eos] + self.tok.encode(text).ids
        idx = torch.tensor([ids], device=self.device)
        t0 = time.perf_counter()
        out = self.model.generate(idx, max_new_tokens, k=k, temperature=temperature, top_k=40)[0, len(ids):].tolist()
        seconds = time.perf_counter() - t0
        cut = next((i for i, t in enumerate(out) if t in stops), len(out))
        self.last = {"ids": ids + out[:cut + 1], "n_ctx": len(ids), "n_hist": 1, "k": k,     # + the stop token
                     "timing": {"seconds": seconds, "tokens": len(out), "temperature": temperature,
                                "device": str(self.device)}}
        return self.tok.decode(out[:cut], skip_special_tokens=False)

    def reply(self, say: str, remember: bool = True, k: int | None = None, temperature: float | None = None) -> str:
        context = (self.history if remember else "") + f"<|human|> {say}\n<|student|>"
        raw = self._continue(context, k or self.k, self.temperature if temperature is None else temperature,
                             self.max_new_tokens, self.stops)
        if remember and self.history:
            self.last["n_hist"] = 1 + len(self.tok.encode(self.history).ids)
        thought = re.search(r"<\|think\|>(.*?)(?:<\|/think\|>|<\|answer\|>|$)", raw, re.S)
        self.last_thought = thought[1].strip() if thought else ""
        answer = raw.rsplit("<|answer|>", 1)[1] if "<|answer|>" in raw else re.sub(r"<\|think\|>.*?(<\|/think\|>|$)", "", raw, flags=re.S)
        answer = re.sub(r"<\|[^|]*\|>", "", answer).split("\n")[0].strip()
        if remember:
            self.history = f"{context} {answer}\n"
        return answer

    def stats(self) -> str:
        if not self.last or len(self.last["ids"]) <= self.last["n_ctx"]:
            return "(no reply yet)"
        last = self.last
        st = analyse(self.model, last["ids"], last["n_hist"], last["n_ctx"], last["k"], self.device)
        timing = {**last["timing"], "seconds": last["timing"]["seconds"] * len(st["reply"]) / max(1, last["timing"]["tokens"])}
        return render(st, self.tok, layer_names(self.model.cfg, last["k"]), timing, self.model, last["k"])

    def story(self, beginning: str, max_new_tokens: int = 150) -> str:
        return re.sub(r"<\|[^|]*\|>", "", self._continue(beginning, self.k, self.temperature, max_new_tokens, {self.eos})).strip()

    def ask(self, question: str, n: int) -> list[str]:
        return [self.reply(question, remember=False) for _ in range(n)]

    def think(self, question: str, ks=(1, 2, 4, 8)) -> dict[int, str]:
        return {k: self.reply(question, remember=False, k=k, temperature=0.0) for k in ks}


def load_student(student: str, version: str | None, device: str | None):
    ckpt_dir = paths.REPO / "checkpoints" / student
    version = version or (ckpt_dir / "LATEST").read_text().strip()
    dev = torch.device(device) if device else pick_device()
    return checkpoint.load(Path(ckpt_dir / version), dev), version, dev


HISTORY_FILE = paths.REPO / "runs" / "chat_history"


def _enable_line_editing() -> None:
    """Cursor keys for editing (←/→) and for earlier lines (↑/↓), kept across sessions in runs/ (outside git)."""
    try:
        import atexit
        import readline
    except ImportError:  # e.g. Windows without pyreadline — plain input still works
        return
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        readline.read_history_file(HISTORY_FILE)
    except (FileNotFoundError, OSError):
        pass
    readline.set_history_length(1000)
    atexit.register(readline.write_history_file, HISTORY_FILE)


def _thought(talker: Talker) -> str:
    return f"\033[2m  (thinks: {talker.last_thought})\033[0m\n" if talker.last_thought else ""


def run(student: str, version: str | None, device: str | None, k: int, temperature: float,
        says: list[str] | None = None, stats: bool = False) -> None:
    model, version, dev = load_student(student, version, device)
    talker = Talker(model, dev, k, temperature)
    info = f"{student}/{version} · {model.num_params():,} parameters · {dev}"
    if says:  # one-shot: a scripted conversation, e.g. for scripts or quick checks
        for say in says:
            answer = talker.reply(say)
            print(f"you> {say}\n{_thought(talker)}student> {answer}")
            if stats:
                print(talker.stats())
        return
    _enable_line_editing()
    print(f"Talking to {info}. /help for commands, /quit to leave; ↑/↓ browse earlier lines.")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if not line.startswith("/"):
            answer = talker.reply(line)
            print(f"{_thought(talker)}student> {answer}")
            if stats:
                print(talker.stats())
            continue
        cmd, _, rest = line.partition(" ")
        rest = rest.strip()
        try:
            if cmd in ("/quit", "/exit"):
                return
            elif cmd == "/help":
                print(HELP.format(k=talker.k, temperature=talker.temperature))
            elif cmd == "/stats":
                if rest in ("on", "off"):
                    stats = rest == "on"
                    print(f"(statistics after every reply: {rest})")
                else:
                    print(talker.stats())
            elif cmd == "/info":
                print(info)
            elif cmd == "/reset":
                talker.history = ""
                print("(conversation forgotten)")
            elif cmd == "/k":
                talker.k = max(1, int(rest))
                print(f"(K = {talker.k})")
            elif cmd == "/temp":
                talker.temperature = max(0.0, float(rest))
                print(f"(temperature = {talker.temperature})")
            elif cmd == "/story":
                print(f"{rest} {talker.story(rest)}")
            elif cmd == "/ask":
                n, _, question = rest.partition(" ")
                for i, answer in enumerate(talker.ask(question, int(n)), 1):
                    print(f"  {i}. {answer}")
            elif cmd == "/think":
                for k, answer in talker.think(rest).items():
                    print(f"  K={k}: {answer}")
            else:
                print("unknown command — /help")
        except ValueError:
            print("bad argument — /help")
