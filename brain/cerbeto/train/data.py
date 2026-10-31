"""Token-window sampling from shelf streams (uint16 .bin files written by `cerbeto library tokenize`)."""
from pathlib import Path

import numpy as np
import torch

from cerbeto import paths


class ShelfStream:
    def __init__(self, shelf: str, split: str, seq_len: int, root: Path = paths.SHELVES):
        self.tokens = np.memmap(root / shelf / f"{split}.bin", dtype=np.uint16, mode="r")
        self.seq_len = seq_len

    def batch(self, batch_size: int, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
        """Random windows of seq_len + 1 tokens (inputs and shifted targets)."""
        starts = rng.integers(0, len(self.tokens) - self.seq_len - 1, size=batch_size)
        windows = np.stack([self.tokens[s:s + self.seq_len + 1] for s in starts]).astype(np.int64)
        return torch.from_numpy(windows).to(device)

    def fixed_batches(self, n: int, batch_size: int, device: torch.device, seed: int = 0) -> list[torch.Tensor]:
        rng = np.random.default_rng(seed)
        return [self.batch(batch_size, rng, device) for _ in range(n)]
