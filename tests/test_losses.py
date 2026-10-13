"""CRPS estimators and the motion-consistency objective."""

import math

import torch

from nowcast.advection import departure_points, grid
from nowcast.losses import crps_ensemble, motion_consistency, multiscale_crps


def gaussian_crps(mu, sigma, y):
    z = (y - mu) / sigma
    pdf = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def test_single_member_crps_is_absolute_error():
    x, y = torch.randn(1, 50), torch.randn(50)
    assert torch.allclose(crps_ensemble(x, y), (x[0] - y).abs())


def test_sorted_spread_matches_pairwise_sum():
    x, y = torch.randn(5, 30), torch.randn(30)
    pairwise = (x[:, None] - x[None]).abs().sum((0, 1))
    expected = (x - y).abs().mean(0) - pairwise / (2 * 5 * 4)
    assert torch.allclose(crps_ensemble(x, y, alpha=1.0), expected, atol=1e-5)


def test_fair_crps_is_unbiased_for_small_ensembles():
    # Two-member ensembles drawn from N(0, 1): the fair score's mean must match
    # the analytic CRPS of the full distribution; the plain ensemble CRPS must not.
    torch.manual_seed(0)
    y = torch.randn(400_000)
    members = torch.randn(2, 400_000)
    exact = gaussian_crps(0.0, 1.0, y).mean()
    assert abs(float(crps_ensemble(members, y, alpha=1.0).mean() - exact)) < 0.005
    assert float(crps_ensemble(members, y, alpha=0.0).mean() - exact) > 0.1


def test_multiscale_crps_vanishes_for_perfect_forecast():
    target = torch.rand(2, 3, 64, 64) * 3
    forecast = target[:, None].expand(2, 4, 3, 64, 64)
    valid = torch.rand(2, 3, 64, 64) > 0.2
    scores = multiscale_crps(forecast, target, valid, scales=(1, 4, 16))
    assert all(float(v) < 1e-6 for v in scores.values())


def test_motion_consistency_prefers_true_velocity():
    pos = grid(1, 64, 64)
    pattern = lambda shift: torch.sin(0.3 * (pos[:, 0] - shift)) * torch.cos(0.2 * pos[:, 1]) + 1
    frames = torch.stack([pattern(-2.0 * (5 - j)) for j in range(6)], dim=1)  # moves +2 px/step in x
    valid = torch.ones(1, 64, 64, dtype=torch.bool)
    true = torch.zeros(1, 2, 64, 64)
    true[:, 0] = 2.0
    loss_true = motion_consistency(frames, valid, departure_points(true, 5))
    loss_zero = motion_consistency(frames, valid, departure_points(torch.zeros_like(true), 5))
    assert float(loss_true) < 0.05 < 10 * 0.05 < float(loss_zero)  # residual: unknown inflow at the edge


def test_motion_consistency_cannot_be_gamed_by_leaving_the_domain():
    pos = grid(1, 64, 64)
    frames = torch.stack([torch.sin(0.3 * (pos[:, 0] + 2.0 * (5 - j))) + 1 for j in range(6)], dim=1)
    valid = torch.ones(1, 64, 64, dtype=torch.bool)
    away = torch.zeros(1, 2, 64, 64)
    away[:, 0] = 100.0  # every departure point far outside
    true = torch.zeros(1, 2, 64, 64)
    true[:, 0] = 2.0
    assert float(motion_consistency(frames, valid, departure_points(away, 5))) > 10 * float(
        motion_consistency(frames, valid, departure_points(true, 5)))


def test_power_transform_crps_vanishes_for_perfect_forecast_and_weights_heavy_rain():
    target = torch.zeros(1, 1, 16, 16)
    target[..., :8, :] = torch.log1p(torch.tensor(20.0))  # heavy rain on half the field
    valid = torch.ones_like(target, dtype=torch.bool)
    perfect = target[:, None].expand(1, 2, 1, 16, 16)
    assert float(multiscale_crps(perfect, target, valid, scales=(1, 4), power=0.5)["crps"]) < 1e-6
    weak = perfect * 0.5  # heavy rain underestimated
    log_score = float(multiscale_crps(weak, target, valid, scales=(1,))["crps"])
    sqrt_score = float(multiscale_crps(weak, target, valid, scales=(1,), power=0.5)["crps"])
    assert sqrt_score > log_score  # the square-root transform penalizes the heavy-rain error more


def test_power_transform_has_finite_gradients_at_zero_rain():
    target = torch.zeros(1, 1, 8, 8)
    forecast = torch.zeros(1, 2, 1, 8, 8, requires_grad=True)  # dry forecast, dry observation
    valid = torch.ones_like(target, dtype=torch.bool)
    multiscale_crps(forecast, target, valid, scales=(1, 4), power=0.5)["crps"].backward()
    assert torch.isfinite(forecast.grad).all()
