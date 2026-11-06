"""Gates G1 and N2 (spec 0001): lossless round trip and a closed 8192-token vocabulary."""
import json
import random

import pytest

from cerbeto import paths
from cerbeto.tokenizer import SPECIAL_TOKENS, load

pytestmark = [pytest.mark.skipif(not paths.TOKENIZER_FILE.exists(), reason="tokenizer not trained yet"),
              pytest.mark.skipif(not paths.SHELVES.exists(), reason="no shelves on this machine (dev copy)")]


def _paragraphs(n: int) -> list[str]:
    pool = []
    for shelf in sorted(paths.SHELVES.iterdir()):
        valid = shelf / "valid.jsonl"
        if valid.exists():
            with valid.open() as f:
                for _, line in zip(range(4000), f):
                    pool += [p for p in json.loads(line)["text"].split("\n") if p.strip()]
    rng = random.Random(1234)
    return rng.sample(pool, min(n, len(pool)))


def test_round_trip_10000_paragraphs():
    tok = load()
    texts = _paragraphs(10_000)
    assert len(texts) == 10_000
    for text, enc in zip(texts, tok.encode_batch(texts)):
        assert tok.decode(enc.ids, skip_special_tokens=False) == text


def test_round_trip_unusual_text():
    tok = load()
    for text in ["Zażółć gęślą jaźń 🙂", "  double  spaces\tand tabs\n\nnew lines ", "ÖFB – 1964/65 «quotes»"]:
        assert tok.decode(tok.encode(text).ids, skip_special_tokens=False) == text


def test_vocabulary_is_closed():
    tok = load()
    assert tok.get_vocab_size() == 8192
    for special in SPECIAL_TOKENS:
        assert tok.token_to_id(special) is not None
    ids = tok.encode(" ".join(_paragraphs(500))).ids
    assert max(ids) < 8192
