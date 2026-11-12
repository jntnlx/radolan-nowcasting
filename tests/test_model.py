"""Network contracts: shapes, persistence at initialization, stochastic members."""

import torch

from nowcast.model import Nowcaster


def small_model(optical_flow=False):
    return Nowcaster(history=4, leads=5, widths=(8, 16, 16, 32), depths=(1, 1, 1, 1), noise_levels=(1, 3), film_levels=(3, 1), c_cond=16,
                     velocity_level=2, optical_flow=optical_flow)


def inputs(b=2, h=32, w=64):
    torch.manual_seed(0)
    return torch.rand(b, 4, h, w) * 3, torch.ones(b, 1, h, w, dtype=torch.bool), torch.randn(b, 4)


def test_untrained_network_without_flow_is_eulerian_persistence():
    out = small_model()(*inputs(), members=3)
    assert out["forecast"].shape == (2, 3, 5, 32, 64)
    latest = inputs()[0][:, -1]
    assert torch.allclose(out["forecast"], latest[:, None, None].expand_as(out["forecast"]), atol=1e-5)


def test_members_differ_once_trained():
    model = small_model()
    torch.manual_seed(1)
    with torch.no_grad():  # stand-in for training: zero-initialized layers become non-zero
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    f = model(*inputs(), members=2)["forecast"]
    assert (f[:, 0] - f[:, 1]).abs().mean() > 1e-3
    assert (f >= 0).all()  # log(1 + R) of a non-negative rain rate


def test_untrained_network_with_flow_is_optical_flow_extrapolation():
    from nowcast.advection import departure_points, sample
    out = small_model(optical_flow=True)(*inputs(), members=2)
    latest = inputs()[0][:, -1:]
    expected = torch.cat([sample(latest, p) for p in departure_points(out["prior"], 5, upsample=4)], dim=1).clamp(min=0)
    assert torch.allclose(out["forecast"][:, 0], expected, atol=1e-4)


def test_untrained_network_with_motion_noise_is_a_displacement_ensemble():
    from nowcast.advection import deformation, sample
    torch.manual_seed(0)
    model = Nowcaster(history=4, leads=5, widths=(8, 16, 16, 32), depths=(1, 1, 1, 1), film_levels=(3,), c_cond=16,
                      velocity_level=2, optical_flow=False, motion_noise=0.3, motion_cells=8, source_level=1)  # 32 px correlation
    x, valid, context = inputs(h=64, w=64)
    out = model(x, valid, context, members=3)
    f = out["forecast"]
    assert f.shape == (2, 3, 5, 64, 64) and (f[:, 0] - f[:, 1]).abs().mean() > 1e-3   # members differ before any training
    # every member is the analysis seen from its own departure points (no source yet) ...
    latest = x[:, -1:].repeat_interleave(3, dim=0)
    assert torch.allclose(f.reshape(6, 5, 64, 64)[:, -1], sample(latest, out["points"][4])[:, 0].clamp(min=0), atol=1e-4)
    # ... under a smooth perturbation that shifts features without deforming them much,
    assert float(deformation(out["points"][4].detach())[0].median()) < 1.2
    # while the physics terms of the objective see the common motion only (here: none).
    assert float(out["points_rigid"].sub(out["points_rigid"][:1]).abs().max()) < 1e-6
    assert torch.allclose(out["points_past"][0], out["points_past"][0][:1].expand_as(out["points_past"][0]))
