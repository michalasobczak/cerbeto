"""Negative gates N1 and N2 (spec 0002): no external weights/tokenizers are fetched; bf16 never appears."""
import socket

import pytest
import torch

from cerbeto.model import checkpoint
from cerbeto.model.student import Student
from cerbeto.tokenizer import load as load_tokenizer
from cerbeto import paths


def test_no_network_access(monkeypatch, tiny_cfg, tmp_path):
    def refuse(*_args, **_kwargs):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    model = Student(tiny_cfg, seed=1)
    checkpoint.load(checkpoint.save(model, tmp_path / "v"))
    if paths.TOKENIZER_FILE.exists():
        load_tokenizer()


def test_no_bf16_anywhere(trained_tiny, tmp_path):
    model, opt, _ = trained_tiny
    tensors = list(model.state_dict().values())
    tensors += [v for st in opt.state.values() for v in st.values() if torch.is_tensor(v)]
    loaded, opt2 = checkpoint.load(checkpoint.save(model, tmp_path / "v", opt), with_optimizer=True)
    tensors += list(loaded.state_dict().values())
    assert all(t.dtype != torch.bfloat16 for t in tensors)


def test_bf16_precision_is_refused():
    from cerbeto.train.smoke import run
    with pytest.raises(ValueError):
        run(minutes=0, precision="bf16")
