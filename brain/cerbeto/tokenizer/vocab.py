"""Spec 0012: a larger vocabulary. Train byte-level BPE tokenizers on the 1B shelf mix (sampled by its weights, plus
the teacher questions) and measure how many bytes one token carries on each shelf's validation text.

    uv run python -m cerbeto.tokenizer.vocab measure --sizes 16384 32768 --mb 200
    uv run python -m cerbeto.tokenizer.vocab train --size 16384 --mb 400
"""
import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from cerbeto import paths
from cerbeto.tokenizer import SPECIAL_TOKENS, _sample, load

# the 1B run's shelf mix (final_test.ARM_SHELF_MIX["bil"]), copied so this module needs no torch
MIX = {"s3_edu_plus": 0.32, "s2_textbooks": 0.22, "s0_dialogue": 0.15, "s0_stories": 0.15,
       "s0_speech": 0.08, "s1_books": 0.05, "s2_wiki_plus": 0.03}
QA_FILE = paths.DATA / "talk" / "qa_teacher.jsonl"
OUT = paths.TOKENIZER_FILE.parent


def corpus(total_mb: int) -> Iterator[str]:
    """Shelves sampled by the mix weights (train split only), then the teacher questions and answers."""
    for shelf, w in MIX.items():
        print(f"  sampling {shelf} ({w * total_mb:.0f} MB)", flush=True)
        yield from _sample(paths.SHELVES / shelf, int(w * total_mb * 1e6))
    if QA_FILE.exists():
        for line in QA_FILE.open():
            d = json.loads(line)
            yield f"{d['q']} {d['a']}"


def train(size: int, total_mb: int, out: Path) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=size, min_frequency=2, special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(corpus(total_mb), trainer=trainer)
    # same special-token ids as the 8k tokenizer, so chat formats and the pad/eos ids do not move
    assert [tok.token_to_id(s) for s in SPECIAL_TOKENS] == list(range(len(SPECIAL_TOKENS)))
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    return tok


def bytes_per_token(tok: Tokenizer, shelf: str, mb: float = 5) -> tuple[float, int]:
    texts = []
    taken = 0
    with (paths.SHELVES / shelf / "valid.jsonl").open() as f:
        for line in f:
            texts.append(json.loads(line)["text"])
            taken += len(texts[-1].encode())
            if taken >= mb * 1e6:
                break
    n = sum(len(e.ids) for e in tok.encode_batch(texts))
    return taken / n, taken


def measure(sizes: list[int], total_mb: int) -> dict:
    toks = {8192: load()}
    for s in sizes:
        t0 = time.time()
        toks[s] = train(s, total_mb, OUT / f"trial_{s}.json")
        print(f"trained {s} in {time.time() - t0:.0f} s", flush=True)
    rows = {}
    for shelf in MIX:
        rows[shelf] = {s: round(bytes_per_token(t, shelf)[0], 3) for s, t in toks.items()}
    weighted = {s: round(sum(MIX[sh] * rows[sh][s] for sh in MIX), 3) for s in toks}
    print(f"{'shelf':16}" + "".join(f"{s:>10}" for s in toks) + "   (bytes per token on valid; more = shorter text)")
    for sh, r in rows.items():
        print(f"{sh:16}" + "".join(f"{r[s]:10.3f}" for s in toks)
              + "   " + " ".join(f"{s // 1024}k −{100 * (1 - r[8192] / r[s]):.1f}%" for s in sizes))
    print(f"{'mix':16}" + "".join(f"{weighted[s]:10.3f}" for s in toks)
          + "   " + " ".join(f"{s // 1024}k −{100 * (1 - weighted[8192] / weighted[s]):.1f}%" for s in sizes))
    words = ["photosynthesis", "electricity", "dinosaur", "kitten", "Jupiter", "Paris", "vegetable", "temperature"]
    for w in words:
        print(f"  {w:15}" + "  ".join(f"{s // 1024}k:{len(t.encode(' ' + w).ids)}" for s, t in toks.items()))
    res = {"rows": rows, "mix": weighted}
    (OUT / "trial_measure.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["measure", "train"])
    p.add_argument("--sizes", type=int, nargs="+", default=[16384, 32768])
    p.add_argument("--size", type=int, default=16384)
    p.add_argument("--mb", type=int, default=200)
    a = p.parse_args()
    if a.what == "measure":
        measure(a.sizes, a.mb)
    else:
        train(a.size, a.mb, OUT / f"tokenizer_{a.size}.json")


if __name__ == "__main__":
    main()
