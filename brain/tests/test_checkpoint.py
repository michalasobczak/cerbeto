"""Gate G1 (spec 0002): save → load reproduces logits exactly; optimizer and growth log survive."""
import torch

from cerbeto.model import checkpoint, growth


def test_save_load_identical_logits(trained_tiny, tmp_path):
    model, opt, data = trained_tiny
    model.eval()
    before = model(data[:, :-1], k=3)
    checkpoint.save(model, tmp_path / "v1", opt)
    loaded, opt2 = checkpoint.load(tmp_path / "v1", with_optimizer=True)
    loaded.eval()
    assert (before - loaded(data[:, :-1], k=3)).abs().max().item() < 1e-6
    for p_old, p_new in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(opt.state[p_old]["exp_avg"], opt2.state[p_new]["exp_avg"])


def test_grown_checkpoint_keeps_architecture_and_log(trained_tiny, tmp_path):
    model, opt, data = trained_tiny
    v1 = checkpoint.save(model, tmp_path / "v1", opt)
    log = v1 / "growth_log.jsonl"
    grown = growth.apply(model, "grow_depth", data, k=2, max_params=10**8, max_delta_loss=0.01, log_path=log)
    v2 = checkpoint.save(grown, tmp_path / "v2", parent=v1)
    loaded = checkpoint.load(v2)
    assert loaded.cfg.n_core == model.cfg.n_core + 1
    assert "grow_depth" in (v2 / "growth_log.jsonl").read_text()
