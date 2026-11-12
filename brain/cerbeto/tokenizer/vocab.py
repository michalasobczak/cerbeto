"""Spec 0012: a larger vocabulary. Train byte-level BPE tokenizers on the 1B shelf mix (sampled by its weights, plus
the teacher questions) and measure how many bytes one token carries on each shelf's validation text.

    uv run python -m cerbeto.tokenizer.vocab measure --sizes 16384 32768 --mb 200
    uv run python -m cerbeto.tokenizer.vocab train --size 16384 --mb 400
    uv run python -m cerbeto.tokenizer.vocab tokenize --size 16384          # the 1B shelves -> train_16384.bin
    uv run python -m cerbeto.tokenizer.vocab bpb <checkpoint> [<checkpoint> ...]   # bits per byte, any vocabulary
    uv run python -m cerbeto.tokenizer.vocab where <checkpoint> <checkpoint>     # bits per byte by word kind
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


def transfer(old, new_vocab: int):
    """The same student with a new vocabulary: every weight but the embedding copied; each new token's embedding is the
    mean of the old embeddings of the pieces the old tokenizer cuts it into (both are byte-level BPE on the same
    alphabet, so any token is a string the old BPE model can cut). Special tokens keep their ids. The head is tied."""
    import torch
    from cerbeto.model.student import Student, StudentConfig
    old_tok, new_tok = load(old.cfg.vocab_size), load(new_vocab)
    new = Student(StudentConfig.from_dict({**old.cfg.to_dict(), "vocab_size": new_vocab}))
    state = {k: v for k, v in old.state_dict().items()}
    e_old = state.pop("embed.weight")
    e_new = torch.empty(new_vocab, e_old.shape[1], dtype=e_old.dtype)
    pieces = 0
    for i in range(new_vocab):
        t = new_tok.id_to_token(i)
        ids = [old_tok.token_to_id(t)] if t in SPECIAL_TOKENS else [p.id for p in old_tok.model.tokenize(t)]
        e_new[i] = e_old[ids].mean(0)
        pieces += len(ids)
    state["embed.weight"] = e_new
    missing, unexpected = new.load_state_dict(state, strict=False)
    assert not unexpected and all(k == "embed.weight" for k in missing), (missing, unexpected)
    print(f"transfer {old.cfg.vocab_size} -> {new_vocab}: {pieces / new_vocab:.2f} old pieces per new token", flush=True)
    return new


def bpb(paths_: list[str], kb: int = 300, shelves=("s0_speech", "s0_stories", "s3_edu_plus", "s2_textbooks", "s0_dialogue")) -> dict:
    """Bits per byte on the first `kb` KB of each shelf's validation text: comparable across vocabularies."""
    import math
    import torch
    from cerbeto.model import checkpoint
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = {}
    for path in paths_:
        if hasattr(path, "cfg"):            # a model already in memory (tests, or a fresh transfer)
            model, path = path.to(dev).eval(), f"model {path.cfg.vocab_size}"
        else:
            p = Path(path)
            model = checkpoint.load(p if p.is_absolute() else paths.REPO / p, dev).eval()
        tok = load(model.cfg.vocab_size)
        eos = tok.token_to_id("<|eos|>")
        row = {}
        for shelf in shelves:
            texts, n_bytes = [], 0
            with (paths.SHELVES / shelf / "valid.jsonl").open() as f:
                for line in f:
                    texts.append(json.loads(line)["text"])
                    n_bytes += len(texts[-1].encode())
                    if n_bytes >= kb * 1000:
                        break
            ids = [i for e in tok.encode_batch(texts) for i in (eos, *e.ids)]
            n = (len(ids) - 1) // 512 * 512
            x = torch.tensor(ids[:n], device=dev).view(-1, 512)
            y = torch.tensor(ids[1:n + 1], device=dev).view(-1, 512)
            nats = 0.0
            with torch.no_grad():
                for j in range(0, len(x), 8):
                    nats += model.loss(x[j:j + 8], y[j:j + 8], 1).item() * y[j:j + 8].numel()
            row[shelf] = round(nats / math.log(2) / (n_bytes * n / len(ids)), 4)    # bytes of the scored part
        out[str(path)] = row
        print(f"{str(path)[-60:]:60}", "  ".join(f"{k.split('_', 1)[1]} {v:.4f}" for k, v in row.items()), flush=True)
    return out


def where(paths_: list[str], docs: int = 400, chars: int = 1500,
          shelves=("s0_speech", "s0_stories", "s3_edu_plus", "s2_textbooks", "s0_dialogue")) -> dict:
    """Where does one model lose against another across vocabularies: bits per byte split by word kind. Both tokenizers
    pre-tokenize the same way (byte-level regex), so token boundaries meet at word boundaries and each word's loss is
    exact in both. A word is 'same' when both cut it into the same pieces, 'merged' when the larger vocabulary uses
    fewer pieces. Each document is cut to `chars` characters so it fits in one 512-token window in both."""
    import bisect
    import math
    import torch
    import torch.nn.functional as F
    from cerbeto.model import checkpoint
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    models = []
    for path in paths_:
        p = Path(path)
        m = checkpoint.load(p if p.is_absolute() else paths.REPO / p, dev).eval()
        models.append((path, m, load(m.cfg.vocab_size)))
    toks = [t for _, _, t in models]
    out = {}
    for shelf in shelves:
        texts = []
        with (paths.SHELVES / shelf / "valid.jsonl").open() as f:
            for line in f:
                texts.append(json.loads(line)["text"][:chars])
                if len(texts) >= docs:
                    break
        acc = {}           # (model, kind) -> [bits, bytes, words]
        for text in texts:
            words = toks[0].pre_tokenizer.pre_tokenize_str(text)
            starts = [s for _, (s, _) in words]
            encs = [t.encode(text) for t in toks]
            pieces = [[[] for _ in words] for _ in toks]
            for k, e in enumerate(encs):
                for tid, (s, _) in zip(e.ids, e.offsets):
                    pieces[k][bisect.bisect_right(starts, s) - 1].append(tid)
            bits = []
            for k, ((name, m, t), e) in enumerate(zip(models, encs)):
                eos = t.token_to_id("<|eos|>")
                ids = torch.tensor([eos, *e.ids][:513], device=dev)
                with torch.no_grad():
                    logits = m(ids[None, :-1], 1)[0].float()
                nll = F.cross_entropy(logits, ids[1:], reduction="none").tolist()
                wb, pos = [], 0
                for w in range(len(words)):
                    n = len(pieces[k][w])
                    if pos + n > len(nll):
                        break
                    wb.append(sum(nll[pos:pos + n]) / math.log(2))
                    pos += n
                bits.append(wb)
            for w, (_, (s, e_)) in enumerate(words[:min(len(b) for b in bits)]):
                strs = [[toks[j].id_to_token(i) for i in pieces[j][w]] for j in range(len(toks))]
                kind = "same" if all(x == strs[0] for x in strs) else "merged"
                nb = len(text[s:e_].encode())
                for k, (name, _, _) in enumerate(models):
                    a = acc.setdefault((name, kind), [0.0, 0, 0])
                    a[0] += bits[k][w]
                    a[1] += nb
                    a[2] += 1
        row = {f"{n[-40:]}|{kind}": {"bpb": round(b / max(nb, 1), 4), "bytes": nb, "words": nw}
               for (n, kind), (b, nb, nw) in acc.items()}
        out[shelf] = row
        print(shelf, flush=True)
        for k_, v in sorted(row.items(), key=lambda kv: kv[0].split("|")[1]):
            print(f"  {k_:52} bpb {v['bpb']:.4f}  bytes {v['bytes']:8}  words {v['words']}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["measure", "train", "tokenize", "bpb", "where"])
    p.add_argument("checkpoints", nargs="*")
    p.add_argument("--sizes", type=int, nargs="+", default=[16384, 32768])
    p.add_argument("--size", type=int, default=16384)
    p.add_argument("--mb", type=int, default=200)
    a = p.parse_args()
    if a.what == "measure":
        measure(a.sizes, a.mb)
    elif a.what == "train":
        train(a.size, a.mb, OUT / f"tokenizer_{a.size}.json")
    elif a.what == "tokenize":
        from cerbeto.library.tokenize import tokenize_all
        tokenize_all(vocab=a.size, shelves=list(MIX))
    elif a.what == "where":
        where(a.checkpoints)
    else:
        bpb(a.checkpoints)


if __name__ == "__main__":
    main()
