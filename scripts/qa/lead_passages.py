"""Spec 0021, step 1: passages of basic knowledge for generate.py. The shelf windows of spec 0020 are random, so the
plainest facts (the capital of a country, what a baby animal is called) are rarely in them: 84 of the 120 knowledge
questions v20 never answers have no pair with their answer. Here a passage is the opening of a Simple English Wikipedia
article, and the articles are the ones other articles mention most (countries, animals, the body, planets, ...).
An article counts as mentioned by another when its title, one to three words, appears in that article's text.
The pairs then go through generate.py and merge.py like the others (merge.py drops anything near a bench1000 question,
so the benchmark asks facts the student was taught in other words, never its own questions).

    uv run --with pyarrow python scripts/qa/lead_passages.py data/library/simplewiki/simplewiki-20231101.parquet \
        --n 40000 --exclude ~/qa_data/passages.jsonl ~/qa_data/passages2.jsonl --out ~/qa_data/passages3.jsonl
"""
import argparse, hashlib, json, re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

MIN_W, MAX_W = 60, 350
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def clean(t: str) -> bool:
    letters = sum(c.isalpha() or c.isspace() for c in t)
    return letters / max(1, len(t)) > 0.85 and sum(ord(c) < 128 for c in t) / max(1, len(t)) > 0.98 and "http" not in t


def key(title: str) -> str | None:
    """A title that can be counted in running text: 1-3 plain words, no digits, brackets or list pages."""
    if re.search(r"[\d(),:/]", title) or title.lower().startswith(("list of", "index of")):
        return None
    ws = WORD.findall(title.lower())
    return " ".join(ws) if 1 <= len(ws) <= 3 and " ".join(ws) == re.sub(r"[^a-z' ]", "", title.lower()).strip() else None


def lead(text: str) -> str | None:
    out, n = [], 0
    for p in text.split("\n"):
        p = " ".join(p.split())
        if not p:
            continue
        if len(p.split()) < 8:
            if out:              # the first heading or short line after the lead ends it
                break
            continue
        out.append(p); n += len(p.split())
        if n >= 120:
            break
    t = "\n".join(out)
    if not MIN_W <= len(t.split()) <= MAX_W + 150 or not clean(t):
        return None
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", type=Path)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[], help="earlier passage files: none of their passages again")
    a = ap.parse_args()
    rows = pq.read_table(a.parquet, columns=["title", "text"]).to_pylist()
    titles = {}
    for i, r in enumerate(rows):
        k = key(r["title"])
        if k and k not in titles:
            titles[k] = i
    print(len(rows), "articles,", len(titles), "countable titles", flush=True)
    mentions = Counter()
    for j, r in enumerate(rows):
        ws = WORD.findall(r["text"].lower())
        grams = set(ws) | {" ".join(ws[i:i + 2]) for i in range(len(ws) - 1)} | {" ".join(ws[i:i + 3]) for i in range(len(ws) - 2)}
        for g in grams:
            if g in titles and titles[g] != j:
                mentions[g] += 1
        if j % 50000 == 0:
            print(j, "articles read", flush=True)
    seen = {json.loads(x)["pid"] for f in a.exclude for x in f.open()}
    got = 0
    with a.out.open("w") as out:
        for k, m in mentions.most_common():
            if got >= a.n:
                break
            r = rows[titles[k]]
            text = lead(r["text"])
            if not text or "may refer to" in text[:300]:
                continue
            pid = hashlib.sha1(text.encode()).hexdigest()[:16]
            if pid in seen:
                continue
            seen.add(pid)
            split = "valid" if int(pid, 16) % 50 == 0 else "train"
            out.write(json.dumps({"pid": pid, "shelf": "s2_wiki_plus", "split": split, "title": r["title"],
                                  "mentions": m, "text": text}) + "\n")
            got += 1
    print("passages", got, "(least mentioned:", m, ") ->", a.out)


if __name__ == "__main__":
    main()
