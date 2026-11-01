"""`cerbeto chat`: talk to a student checkpoint in the terminal and probe what it can do.

The conversation is kept as context in the chat format (`<|human|> …` / `<|student|> …`), the same one
the nursery speech shelf and the exams use. Slash commands probe single skills without the history.
"""
from pathlib import Path

import torch

from cerbeto import paths
from cerbeto.model import checkpoint
from cerbeto.model.student import Student, pick_device
from cerbeto.tokenizer import load as load_tokenizer

HELP = """commands:
  /story <beginning>    continue a story (no chat history)
  /ask <n> <question>   ask the same question n times — is the answer stable?
  /think <question>     answer with K = 1, 2, 4, 8 passes of the recurrent core (greedy)
  /k <n>                passes of the recurrent core for chat ("think longer"), now {k}
  /temp <t>             sampling temperature, 0 = always the most likely token, now {temperature}
  /reset                forget the conversation
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

    @torch.no_grad()
    def _continue(self, text: str, k: int, temperature: float, max_new_tokens: int, stops: set[int]) -> str:
        ids = [self.eos] + self.tok.encode(text).ids
        idx = torch.tensor([ids], device=self.device)
        out = self.model.generate(idx, max_new_tokens, k=k, temperature=temperature, top_k=40)[0, len(ids):].tolist()
        cut = next((i for i, t in enumerate(out) if t in stops), len(out))
        return self.tok.decode(out[:cut])

    def reply(self, say: str, remember: bool = True, k: int | None = None, temperature: float | None = None) -> str:
        context = (self.history if remember else "") + f"<|human|> {say}\n<|student|>"
        answer = self._continue(context, k or self.k, self.temperature if temperature is None else temperature,
                                self.max_new_tokens, self.stops).split("\n")[0].strip()
        if remember:
            self.history = f"{context} {answer}\n"
        return answer

    def story(self, beginning: str, max_new_tokens: int = 150) -> str:
        return self._continue(beginning, self.k, self.temperature, max_new_tokens, {self.eos}).strip()

    def ask(self, question: str, n: int) -> list[str]:
        return [self.reply(question, remember=False) for _ in range(n)]

    def think(self, question: str, ks=(1, 2, 4, 8)) -> dict[int, str]:
        return {k: self.reply(question, remember=False, k=k, temperature=0.0) for k in ks}


def load_student(student: str, version: str | None, device: str | None):
    ckpt_dir = paths.REPO / "checkpoints" / student
    version = version or (ckpt_dir / "LATEST").read_text().strip()
    dev = torch.device(device) if device else pick_device()
    return checkpoint.load(Path(ckpt_dir / version), dev), version, dev


def run(student: str, version: str | None, device: str | None, k: int, temperature: float,
        says: list[str] | None = None) -> None:
    model, version, dev = load_student(student, version, device)
    talker = Talker(model, dev, k, temperature)
    info = f"{student}/{version} · {model.num_params():,} parameters · {dev}"
    if says:  # one-shot: a scripted conversation, e.g. for scripts or quick checks
        for say in says:
            print(f"you> {say}\nstudent> {talker.reply(say)}")
        return
    print(f"Talking to {info}. /help for commands, /quit to leave.")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if not line.startswith("/"):
            print(f"student> {talker.reply(line)}")
            continue
        cmd, _, rest = line.partition(" ")
        rest = rest.strip()
        try:
            if cmd in ("/quit", "/exit"):
                return
            elif cmd == "/help":
                print(HELP.format(k=talker.k, temperature=talker.temperature))
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
