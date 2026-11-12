"""Spec 0012: moving a student to another vocabulary keeps every weight but the embedding, and a token the old
tokenizer has as one piece keeps its embedding exactly (here: the same vocabulary, so the model is unchanged)."""
import torch

from cerbeto.model.student import Student, StudentConfig
from cerbeto.tokenizer import load, path_for
from cerbeto.tokenizer.vocab import transfer
from cerbeto.train.data import bin_name


def tiny():
    torch.manual_seed(0)
    return Student(StudentConfig(vocab_size=8192, d_model=64, n_heads=2, head_dim=32, mlp_hidden=128,
                                 n_prelude=1, n_core=1, n_coda=1, max_seq_len=64)).eval()


def test_transfer_same_vocab_is_identity():
    old = tiny()
    new = transfer(old, 8192).eval()
    x = torch.tensor([load().encode("<|human|> Where do fish live?\n<|student|>").ids])
    with torch.no_grad():
        assert torch.allclose(old(x, 1), new(x, 1), atol=1e-6)


def test_names_for_the_old_vocabulary_do_not_change():
    assert bin_name("train") == "train.bin" and bin_name("valid", 16384) == "valid_16384.bin"
    assert path_for(8192).name == "tokenizer.json" and path_for(16384).name == "tokenizer_16384.json"
