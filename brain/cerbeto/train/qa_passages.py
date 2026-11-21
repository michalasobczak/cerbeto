"""Spec 0019, step 1: passages for the fact questions, taken from the shelves the student read (train split only, so the
validation stays unseen). Text shelves (train.jsonl) are read directly; the older shelves keep only their token streams,
which are decoded document by document (every document ends with <|eos|>). A passage is a run of whole paragraphs,
120-350 words, mostly letters (no tables, code or long formulas). 2% of passages, chosen by hash, are held out.

    uv run python -m cerbeto.train.qa_passages --n 120000 --out ../data/qa/passages.jsonl
"""
import argparse, hashlib, html, json, random
from pathlib import Path

import numpy as np

from cerbeto import paths
from cerbeto.tokenizer import load
from cerbeto.train.data import bin_name

# shares of passages per shelf: encyclopedic and textbook text first, where the facts are
SHARES = {"s2_wiki_plus": 0.25, "s4_textbooks": 0.25, "s4_edu_12": 0.20, "s3_edu_plus": 0.15, "s3_edu_8": 0.10, "s2_textbooks": 0.05}
VOCAB = 16384
MIN_W, MAX_W = 120, 350


def clean(t: str) -> bool:
    letters = sum(c.isalpha() or c.isspace() for c in t)
    return letters / max(1, len(t)) > 0.9 and sum(ord(c) < 128 for c in t) / max(1, len(t)) > 0.98 and "http" not in t


def re_space(p: str) -> str:
    return " ".join(p.split())


def window(doc: str, rng: random.Random) -> str | None:
    paras = [re_space(html.unescape(p)) for p in doc.split("\n") if len(p.split()) >= 8]
    if not paras:
        return None
    start = rng.randrange(len(paras))
    out, n = [], 0
    for p in paras[start:]:
        w = len(p.split())
        if n + w > MAX_W and n >= MIN_W:
            break
        out.append(p); n += w
        if n >= MAX_W:
            break
    text = "\n".join(out)
    if not MIN_W <= len(text.split()) <= MAX_W + 150 or not clean(text):
        return None
    return text


def docs_from_jsonl(f: Path, rng: random.Random):
    size = f.stat().st_size
    with f.open("rb") as fh:
        while True:
            fh.seek(rng.randrange(size)); fh.readline()
            line = fh.readline()
            if line:
                yield json.loads(line)["text"]


def docs_from_bin(f: Path, rng: random.Random, tok, eos: int):
    t = np.memmap(f, dtype=np.uint16, mode="r")
    while True:
        i = rng.randrange(len(t) - 10000)
        chunk = np.asarray(t[i:i + 10000])
        ends = np.flatnonzero(chunk == eos)
        if len(ends) >= 2:
            yield tok.decode(chunk[ends[0] + 1:ends[1]].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120000)
    ap.add_argument("--out", type=Path, default=paths.DATA / "qa" / "passages.jsonl")
    ap.add_argument("--seed", type=int, default=19)
    a = ap.parse_args()
    rng, tok = random.Random(a.seed), load(VOCAB)
    eos = tok.token_to_id("<|eos|>")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    seen, counts = set(), {}
    with a.out.open("w") as out:
        for shelf, share in SHARES.items():
            root = paths.SHELVES / shelf
            src = root / "train.jsonl"
            docs = docs_from_jsonl(src, rng) if src.exists() else docs_from_bin(root / bin_name("train", VOCAB), rng, tok, eos)
            want, got, tries = int(a.n * share), 0, 0
            while got < want and tries < want * 20:
                tries += 1
                text = window(next(docs), rng)
                if not text:
                    continue
                pid = hashlib.sha1(text.encode()).hexdigest()[:16]
                if pid in seen:
                    continue
                seen.add(pid)
                split = "valid" if int(pid, 16) % 50 == 0 else "train"
                out.write(json.dumps({"pid": pid, "shelf": shelf, "split": split, "text": text}) + "\n")
                got += 1
            counts[shelf] = got
            print(shelf, got, "of", want, f"({tries} tries)", flush=True)
    print("passages", sum(counts.values()), "->", a.out)


if __name__ == "__main__":
    main()
