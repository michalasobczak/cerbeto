"""Turns text shelves into flat uint16 token streams (train.bin / valid.bin) for training."""
import json
from pathlib import Path

import numpy as np

from cerbeto import paths
from cerbeto.tokenizer import load


def _encode_file(src: Path, dst: Path, batch_docs: int = 4096) -> int:
    tok = load()
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


def tokenize_shelf(shelf_dir: Path, force: bool = False) -> dict:
    manifest_path = shelf_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tok_hash = paths.TOKENIZER_FILE.stat().st_mtime_ns
    if not force and manifest.get("tokenizer_mtime") == tok_hash and (shelf_dir / "train.bin").exists():
        print(f"[{shelf_dir.name}] tokens up to date — skipping")
        return manifest
    tokens = {}
    for split in ("train", "valid"):
        print(f"[{shelf_dir.name}] tokenizing {split}")
        tokens[split] = _encode_file(shelf_dir / f"{split}.jsonl", shelf_dir / f"{split}.bin")
    manifest["tokens"] = tokens
    manifest["tokenizer_mtime"] = tok_hash
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def tokenize_all(force: bool = False) -> None:
    for shelf_dir in sorted(paths.SHELVES.iterdir()):
        if (shelf_dir / "manifest.json").exists():
            tokenize_shelf(shelf_dir, force)
