"""Nursery conversation probe (spec 0003, gate G3): can the student hold a simple conversational turn?"""
import json
from pathlib import Path

import torch
import yaml

from cerbeto.model.student import Student
from cerbeto.teachers import chat
from cerbeto.tokenizer import load as load_tokenizer

PROBE = Path(__file__).with_name("conversation_probe.yaml")


@torch.no_grad()
def reply(model: Student, say: str, k: int, device, max_new_tokens: int = 40, temperature: float = 0.7) -> str:
    tok = load_tokenizer()
    ids = torch.tensor([tok.encode(f"<|human|> {say}\n<|student|>").ids], device=device)
    out = model.generate(ids, max_new_tokens, k=k, temperature=temperature, top_k=40)[0, ids.shape[1]:].tolist()
    stops = {tok.token_to_id(t) for t in ("<|eos|>", "<|human|>", "<|student|>")}
    cut = next((i for i, t in enumerate(out) if t in stops), len(out))
    return tok.decode(out[:cut]).split("\n")[0].strip()


def grade(say: str, answer: str, rubric: str, teacher: str | None = None) -> dict:
    content = chat([{"role": "system", "content": rubric},
                    {"role": "user", "content": f'Adult: "{say}"\nChild: "{answer}"'}],
                   teacher=teacher, temperature=0.0, json_output=True)
    try:
        result = json.loads(content)
        return {"score": int(result["score"]), "reason": str(result.get("reason", ""))}
    except (ValueError, KeyError, TypeError):
        return {"score": 1, "reason": f"unparseable grade: {content[:80]}"}


def evaluate(model: Student, k: int, device, teacher: str | None = None, threshold: int = 3) -> dict:
    probe = yaml.safe_load(PROBE.read_text())
    torch.manual_seed(0)
    items = []
    for category, lines in probe["situations"].items():
        for say in lines:
            answer = reply(model, say, k, device)
            items.append({"category": category, "say": say, "reply": answer,
                          **grade(say, answer, probe["rubric"], teacher)})
    fitting = sum(i["score"] >= threshold for i in items) / len(items)
    by_cat = {c: sum(i["score"] >= threshold for i in items if i["category"] == c) /
              sum(i["category"] == c for i in items) for c in probe["situations"]}
    return {"fitting": fitting, "by_category": by_cat, "items": items}
