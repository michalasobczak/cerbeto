"""Turns text shelves into flat uint16 token streams (train.bin / valid.bin) for training."""
import json
from pathlib import Path

import numpy as np

from cerbeto import paths
from cerbeto.tokenizer import load, path_for
from cerbeto.train.data import bin_name


def _encode_file(src: Path, dst: Path, batch_docs: int = 4096, vocab: int = 8192) -> int:
    tok = load(vocab)
    eos = tok.token_to_id("<|eos|>")
    total = 0
    with src.open() as f, dst.open("wb") as out:
        batch: list[str] = []

        def flush() -> int:
            ids = [i for enc in tok.encode_batch(batch) for i in (*enc.ids, eos)]
            np.asarray(ids, dtype=np.uint16).tofile(out)
            return len(ids)

        for line in f:
            batch.append(json.loads(line)["text"])
            if len(batch) == batch_docs:
                total += flush()
                batch = []
        if batch:
            total += flush()
    return total


def tokenize_shelf(shelf_dir: Path, force: bool = False, vocab: int = 8192) -> dict:
    manifest_path = shelf_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tok_hash = path_for(vocab).stat().st_mtime_ns
    key = "" if vocab == 8192 else f"_{vocab}"
    if not force and manifest.get(f"tokenizer_mtime{key}") == tok_hash and (shelf_dir / bin_name("train", vocab)).exists():
        print(f"[{shelf_dir.name}] tokens up to date — skipping")
        return manifest
    tokens = {}
    for split in ("train", "valid"):
        print(f"[{shelf_dir.name}] tokenizing {split} ({vocab})", flush=True)
        tokens[split] = _encode_file(shelf_dir / f"{split}.jsonl", shelf_dir / bin_name(split, vocab), vocab=vocab)
    manifest[f"tokens{key}"] = tokens
    manifest[f"tokenizer_mtime{key}"] = tok_hash
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def tokenize_all(force: bool = False, vocab: int = 8192, shelves: list[str] | None = None) -> None:
    for shelf_dir in sorted(paths.SHELVES.iterdir()):
        if (shelf_dir / "manifest.json").exists() and (shelves is None or shelf_dir.name in shelves):
            tokenize_shelf(shelf_dir, force, vocab)
