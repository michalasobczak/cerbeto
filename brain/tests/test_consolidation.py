"""Synaptic consolidation (EWC): zero penalty at the anchor, positive after drift, survives growth."""
import torch

from cerbeto.model import growth
from cerbeto.model.consolidation import EWC


def test_penalty_zero_at_anchor_and_grows_with_drift(trained_tiny, tmp_path):
    model, _, data = trained_tiny
    ewc = EWC.estimate(model, [data, data.flip(0)], k=2)
    assert ewc.penalty(model).item() == 0
    with torch.no_grad():
        model.core[0].mlp.up.weight.add_(0.1)
    assert ewc.penalty(model).item() > 0
    ewc.save(tmp_path / "ewc.safetensors")
    assert EWC.load(tmp_path / "ewc.safetensors").penalty(model).item() == ewc.penalty(model).item()


def test_penalty_ignores_weights_added_by_growth(trained_tiny):
    model, _, data = trained_tiny
    ewc = EWC.estimate(model, [data], k=2)
    grown, _ = growth.grow_width(model, extra_mlp=16)
    assert ewc.penalty(grown).item() == 0
