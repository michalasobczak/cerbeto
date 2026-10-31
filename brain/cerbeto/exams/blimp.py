"""BLiMP (BabyLM filtered): does the student prefer the grammatical sentence of each minimal pair?"""
import json
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from cerbeto import paths
from cerbeto.model.student import Student
from cerbeto.tokenizer import load as load_tokenizer

BLIMP = paths.LIBRARY / "blimp"


def pairs(limit_per_phenomenon: int | None = None) -> dict[str, list[tuple[str, str]]]:
    out = {}
    for f in sorted(BLIMP.glob("*/train-*.parquet")):
        rows = pq.read_table(f, columns=["sentence_good", "sentence_bad"]).to_pylist()
        out[f.parent.name] = [(r["sentence_good"], r["sentence_bad"]) for r in rows][:limit_per_phenomenon]
    return out


@torch.no_grad()
def sentence_logprobs(model: Student, sentences: list[str], k: int, device, batch_size: int = 128) -> list[float]:
    """Sum of log p(token) over each sentence, conditioned on a leading <|eos|> (document boundary)."""
    tok = load_tokenizer()
    eos = tok.token_to_id("<|eos|>")
    encoded = [[eos, *e.ids] for e in tok.encode_batch(sentences)]
    scores = []
    model.eval()
    for i in range(0, len(encoded), batch_size):
        chunk = encoded[i:i + batch_size]
        width = max(map(len, chunk))
        ids = torch.full((len(chunk), width), eos, dtype=torch.long)
        mask = torch.zeros(len(chunk), width - 1)
        for j, seq in enumerate(chunk):
            ids[j, :len(seq)] = torch.tensor(seq)
            mask[j, :len(seq) - 1] = 1
        ids, mask = ids.to(device), mask.to(device)
        logp = F.log_softmax(model(ids[:, :-1], k).float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
        scores += (logp * mask).sum(-1).tolist()
    return scores


def evaluate(model: Student, k: int, device, limit_per_phenomenon: int | None = None) -> dict:
    per = {}
    for phenomenon, ps in pairs(limit_per_phenomenon).items():
        good = sentence_logprobs(model, [g for g, _ in ps], k, device)
        bad = sentence_logprobs(model, [b for _, b in ps], k, device)
        per[phenomenon] = sum(g > b for g, b in zip(good, bad)) / len(ps)
    return {"accuracy": sum(per.values()) / len(per), "phenomena": per}


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z']+", s.lower()))


def check_leak(shelves: list[str]) -> dict:
    """Gate N1 (spec 0003): count BLiMP sentences that occur verbatim (normalized) in the training shelves."""
    targets = {_norm(s) for ps in pairs().values() for pair in ps for s in pair}
    hits = defaultdict(int)
    splitter = re.compile(r"(?<=[.!?])\s+|\n")
    for shelf in shelves:
        with (paths.SHELVES / shelf / "train.jsonl").open() as f:
            for line in f:
                text = json.loads(line)["text"].replace("<|human|>", "").replace("<|student|>", "")
                for sentence in splitter.split(text):
                    if _norm(sentence) in targets:
                        hits[shelf] += 1
    return {"blimp_sentences": len(targets), "hits": dict(hits)}
