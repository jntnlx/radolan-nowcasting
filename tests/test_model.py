"""Network contracts: shapes, persistence at initialization, stochastic members."""

import torch

from nowcast.model import Nowcaster


def small_model(optical_flow=False):
    return Nowcaster(history=4, leads=5, widths=(8, 16, 16, 32), depths=(1, 1, 1, 1), noise_levels=(2, 3), c_cond=16,
                     velocity_level=2, optical_flow=optical_flow)


def inputs(b=2, h=32, w=64):
    torch.manual_seed(0)
    return torch.rand(b, 4, h, w) * 3, torch.ones(b, 1, h, w, dtype=torch.bool), torch.randn(b, 4)


def test_untrained_network_without_flow_is_eulerian_persistence():
    out = small_model()(*inputs(), members=3)
    assert out["forecast"].shape == (2, 3, 5, 32, 64)
    latest = inputs()[0][:, -1]
    assert torch.allclose(out["forecast"], latest[:, None, None].expand_as(out["forecast"]), atol=1e-5)


def test_members_differ_once_heads_are_trained():
    model = small_model()
    torch.nn.init.normal_(model.source_head.weight, std=0.1)
    torch.nn.init.normal_(model.velocity_head.weight, std=0.1)
    f = model(*inputs(), members=2)["forecast"]
    assert (f[:, 0] - f[:, 1]).abs().mean() > 1e-3
    assert (f >= 0).all()  # log(1 + R) of a non-negative rain rate


def test_untrained_network_with_flow_is_optical_flow_extrapolation():
    from nowcast.advection import departure_points, sample
    out = small_model(optical_flow=True)(*inputs(), members=2)
    latest = inputs()[0][:, -1:]
    expected = torch.cat([sample(latest, p) for p in departure_points(out["prior"], 5, upsample=4)], dim=1).clamp(min=0)
    assert torch.allclose(out["forecast"][:, 0], expected, atol=1e-4)
