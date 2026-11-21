"""Spec 0020, step 3: the shards of generate.py into the tuning files. Duplicate questions go (the first one stays), and so do
questions too close to any of bench1000's (same question, or most of the same content words), so the benchmark stays a
test. Passages held out by qa_passages.py give the validation pairs.

    python3 scripts/qa/merge.py data/qa/qa_*.jsonl     ->  data/talk/qa_facts.jsonl, data/talk/qa_facts_valid.jsonl
"""
import json, re, sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STOP = set("a an the is are was were do does did of in on at to for and or what which who where when why how many much "
           "can could you your we our it its be by with from that this there their they as not".split())
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
words = lambda s: {w for w in norm(s).split() if w not in STOP and len(w) > 1}

bench = json.loads((REPO / "scripts/hf/bench1000.json").read_text())
bq = [x["q"] if isinstance(x["q"], str) else x["q"][-1] for x in bench]
bnorm = {norm(q) for q in bq}
bwords = [words(q) for q in bq]
index = {}
for k, ws in enumerate(bwords):
    for w in ws:
        index.setdefault(w, []).append(k)


def near_bench(q: str) -> bool:
    if norm(q) in bnorm:
        return True
    ws = words(q)
    if not ws:
        return False
    hits = Counter(k for w in ws for k in index.get(w, []))
    return any(c / len(ws | bwords[k]) >= 0.6 for k, c in hits.items())


seen, out, stats = set(), {"train": [], "valid": []}, Counter()
for f in sys.argv[1:]:
    for line in open(f):
        r = json.loads(line)
        k = norm(r["q"])
        if k in seen:
            stats["duplicate"] += 1
            continue
        seen.add(k)
        if near_bench(r["q"]):
            stats["near bench1000"] += 1
            continue
        out[r["split"]].append({"q": r["q"], "a": r["a"], "type": r["type"], "shelf": r["shelf"]})
for split, name in (("train", "qa_facts.jsonl"), ("valid", "qa_facts_valid.jsonl")):
    p = REPO / "data/talk" / name
    p.write_text("".join(json.dumps(r) + "\n" for r in out[split]))
    print(f"{split}: {len(out[split])} -> {p}")
print(dict(stats))
print("types", Counter(r["type"] for r in out["train"]).most_common())
print("shelves", Counter(r["shelf"] for r in out["train"]).most_common())
