"""Gate G5 (spec 0002): training + inference of a 30M student peaks at ≤ 3 GB (process RSS + GPU)."""
import os

import pytest
import torch

from cerbeto.model.student import Student, StudentConfig, count_params, pick_device
from cerbeto.train.smoke import memory_mb

LIMIT_MB = 3 * 1024


@pytest.mark.slow
def test_30m_student_fits_in_3gb():
    device = pick_device()
    cfg = StudentConfig(d_model=512, n_heads=16, mlp_hidden=1408, n_core=4)
    assert 29_000_000 < count_params(cfg) <= 30_000_000
    model = Student(cfg, seed=0).to(device)
    model.checkpointing = True
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    peak = 0.0
    for _ in range(5):  # micro-batch 8 × 512 (batch 16 = 2 accumulated micro-batches), deepest recurrence
        batch = torch.randint(0, cfg.vocab_size, (8, cfg.max_seq_len + 1), device=device)
        opt.zero_grad()
        model.loss(batch[:, :-1], batch[:, 1:], k=6).backward()
        opt.step()
        peak = max(peak, memory_mb(device))
    model.eval()
    with torch.no_grad():
        model.generate(torch.zeros(1, 8, dtype=torch.long, device=device), 32, k=8)
    peak = max(peak, memory_mb(device))
    print(f"peak memory: {peak:.0f} MB (allocated now: {torch.mps.current_allocated_memory() / 2**20:.0f} MB)" if device.type == "mps" else f"peak memory: {peak:.0f} MB")
    assert peak <= LIMIT_MB
