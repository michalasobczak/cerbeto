"""Library report — shelves, licences, token counts and the 0001 acceptance numbers."""
import json

from cerbeto import paths
from cerbeto.tokenizer import load

STAGE0_MIN_TOKENS = 250_000_000
MAX_TOKENS_PER_WORD = 1.4


def tokens_per_word(sample_docs: int = 2000) -> float:
    tok = load()
    words = tokens = 0
    with (paths.SHELVES / "s0_stories" / "valid.jsonl").open() as f:
        for i, line in enumerate(f):
            if i == sample_docs:
                break
            text = json.loads(line)["text"]
            words += len(text.split())
            tokens += len(tok.encode(text).ids)
    return tokens / max(words, 1)


def collect() -> list[dict]:
    return [json.loads((d / "manifest.json").read_text())
            for d in sorted(paths.SHELVES.iterdir()) if (d / "manifest.json").exists()]


def report() -> bool:
    shelves = collect()
    print(f"{'shelf':<11}{'stage':>6}{'docs':>11}{'words':>14}{'tokens train':>15}{'tokens valid':>14}"
          f"{'rejected¶':>11}  license")
    stage0 = 0
    for m in shelves:
        tok = m.get("tokens", {})
        if m["stage"] == 0:
            stage0 += tok.get("train", 0)
        print(f"{m['shelf']:<11}{m['stage']:>6}{m['docs']['train'] + m['docs']['valid']:>11,}"
              f"{m['words']['train']:>14,}{tok.get('train', 0):>15,}{tok.get('valid', 0):>14,}"
              f"{m.get('paragraphs_rejected', 0):>11,}  {m['license']}")
    tpw = tokens_per_word()
    ok_stage0 = stage0 >= STAGE0_MIN_TOKENS
    ok_tpw = tpw <= MAX_TOKENS_PER_WORD
    ok_licence = all(m.get("license") for m in shelves)
    print(f"\nStage 0 tokens: {stage0:,} (gate ≥ {STAGE0_MIN_TOKENS:,}) {'PASS' if ok_stage0 else 'FAIL'}")
    print(f"Tokens per word (TinyStories valid): {tpw:.3f} (gate ≤ {MAX_TOKENS_PER_WORD}) {'PASS' if ok_tpw else 'FAIL'}")
    print(f"Every shelf has a licence: {'PASS' if ok_licence else 'FAIL'}")
    return ok_stage0 and ok_tpw and ok_licence
