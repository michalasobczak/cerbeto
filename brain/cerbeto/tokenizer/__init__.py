"""Cerbeto's own byte-level BPE tokenizers, trained from scratch: 8192 tokens on the school library (`tokenizer.json`),
16384 on the 1B shelf mix (`tokenizer_16384.json`, spec 0012). A model picks its tokenizer by its vocabulary size."""
import json
import random
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from cerbeto import paths

SPECIAL_TOKENS = [
    "<|pad|>", "<|bos|>", "<|eos|>",
    "<|teacher|>", "<|student|>", "<|human|>",
    "<|think|>", "<|/think|>", "<|answer|>",
]


def _sample(shelf_dir: Path, max_bytes: int, seed: int = 0) -> Iterator[str]:
    """Reservoir-free balanced sample: stride through the shelf until `max_bytes` are taken."""
    path = shelf_dir / "train.jsonl"
    total = path.stat().st_size
    keep_ratio = min(1.0, max_bytes / max(total, 1))
    rng = random.Random(seed)
    taken = 0
    with path.open() as f:
        for line in f:
            if rng.random() <= keep_ratio:
                text = json.loads(line)["text"]
                taken += len(text)
                yield text
                if taken >= max_bytes:
                    return


def train(vocab_size: int, sample_mb_per_shelf: int, out: Path = paths.TOKENIZER_FILE) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    shelves = sorted(p for p in paths.SHELVES.iterdir() if (p / "train.jsonl").exists())

    def corpus() -> Iterator[str]:
        for shelf in shelves:
            print(f"  sampling {shelf.name}")
            yield from _sample(shelf, sample_mb_per_shelf * 1_000_000)

    tok.train_from_iterator(corpus(), trainer=trainer)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    return tok


def path_for(vocab_size: int = 8192) -> Path:
    return paths.TOKENIZER_FILE if vocab_size == 8192 else paths.TOKENIZER_FILE.parent / f"tokenizer_{vocab_size}.json"


def load(vocab_size: int = 8192) -> Tokenizer:
    return Tokenizer.from_file(str(path_for(vocab_size)))
