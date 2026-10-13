"""
Training objectives.

The forecast is an ensemble, trained with the continuous ranked probability
score (CRPS), a strictly proper scoring rule: its expectation is minimized
only by forecasting the true conditional distribution of the future. Unlike
MSE, which is minimized by the (blurry) conditional mean, a CRPS-trained
ensemble is rewarded for sharp members whose spread matches the actual
uncertainty.

Pixelwise CRPS constrains only marginal distributions. An ensemble of
spatially white noise could score well per pixel yet be useless for an area
or a catchment. The score is therefore also applied to area-mean rain rates
at several scales (multiscale CRPS): each term is proper for the distribution
of the pooled quantity, so the sum rewards correct spatial structure of the
uncertainty up to the largest pooling scale.
"""

import torch

from .advection import sample
from .transforms import pool_rate, rate_to_y, y_to_rate


def crps_ensemble(members: torch.Tensor, obs: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """
    Almost-fair ensemble CRPS (Lang et al. 2024, AIFS-CRPS), elementwise.

    members: (M, ...), obs: (...).
        CRPS = mean_i |x_i - y| - (1 - eps) / (2 M (M - 1)) * sum_ij |x_i - x_j|,  eps = (1 - alpha) / M

    alpha = 1 is the fair CRPS (Ferro 2014), an unbiased estimate of the CRPS
    of the infinite ensemble the members are drawn from. alpha < 1 blends in a
    little of the standard (biased) ensemble CRPS, which removes a degenerate
    direction of the fair score in which members can drift apart at no cost.
    """
    m = members.shape[0]
    skill = (members - obs).abs().mean(0)
    if m == 1:
        return skill
    # sum_ij |x_i - x_j| = 2 sum_i (2i - m + 1) x_(i) for ascending order statistics x_(i)
    ranks = torch.arange(m, device=members.device, dtype=members.dtype)
    coefficient = (2 * ranks - m + 1).reshape(m, *[1] * (members.ndim - 1))
    pair_sum = 2 * (coefficient * members.sort(dim=0).values).sum(0)
    eps = (1 - alpha) / m
    return skill - (1 - eps) * pair_sum / (2 * m * (m - 1))


def multiscale_crps(
    forecast: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    scales: tuple[int, ...] = (1, 4, 16, 64),
    alpha: float = 0.95,
    power: float | None = None,
) -> dict[str, torch.Tensor]:
    """
    Mean CRPS of area-mean rain rates at several scales (km).

    The score is computed for y = log(1 + R), or for (R + eps)**power - eps**power
    if `power` is given (eps = 0.01 mm/h keeps the gradient finite at R = 0).
    The choice of transform g sets the threshold weighting: the CRPS of g(R)
    is the Brier score integrated over thresholds r with weight g'(r), i.e.
    1 / (1 + r) for log(1 + R) and ~ r**(power - 1) for the power transform,
    which gives heavy rain relatively more weight for power closer to 1.

    forecast: (B, M, K, H, W) ensemble in y;  target: (B, K, H, W) in y;
    valid: (B, K, H, W) bool, where the target has radar coverage.
    """
    members = forecast.movedim(1, 0)
    eps = 0.01
    transform = rate_to_y if power is None else (lambda rate: (rate.clamp(min=0) + eps) ** power - eps**power)
    scores = {}
    for scale in scales:
        if scale == 1 and power is None:
            pred, obs, ok = members, target, valid
        else:
            pred, _ = pool_rate(y_to_rate(members), valid, scale)
            obs, ok = pool_rate(y_to_rate(target), valid, scale)
            pred, obs = transform(pred), transform(obs)
        score = crps_ensemble(pred, obs, alpha)
        scores[f"crps_{scale}km"] = (score * ok).sum() / ok.sum().clamp(min=1)
    scores["crps"] = torch.stack([scores[f"crps_{s}km"] for s in scales]).mean()
    return scores


def motion_consistency(inputs: torch.Tensor, valid: torch.Tensor, points: list[torch.Tensor]) -> torch.Tensor:
    """
    How well the velocity field explains the *observed past*.

    The parcel at x at the analysis time was at X_j(x) j steps earlier, so the
    observed frame t0 - j sampled at X_j should reproduce the frame at t0.
    This is the brightness-constancy objective of variational optical flow,
    evaluated with the same advection operator as the forecast. It uses
    inputs only, anchoring the learned velocity to physical motion without
    access to the future. L1 keeps it robust to growth and decay.

    Every pixel with radar coverage counts, including those whose departure
    point leaves the field (they see the replicated edge value). Averaging
    only over departure points inside the field would let the network zero
    this loss by pointing all trajectories off the map.

    inputs: (B, T, H, W) in y, oldest first;  valid: (B, H, W);  points: [X_1, ...].
    """
    t = inputs.shape[1]
    latest = inputs[:, -1]
    n = valid.sum().clamp(min=1)
    losses = []
    for j in range(1, t):
        past = sample(inputs[:, t - 1 - j:t - j], points[j - 1], padding="border")[:, 0]
        losses.append(((past - latest).abs() * valid).sum() / n)
    return torch.stack(losses).mean()


def advection_crps(latest: torch.Tensor, points: list[torch.Tensor], target: torch.Tensor, valid: torch.Tensor,
                   scales: tuple[int, ...] = (1, 16), alpha: float = 0.95) -> torch.Tensor:
    """
    CRPS of the motion-only forecast: the analysis field advected by each member's motion.

    In the full forecast, a growth/decay term can partly compensate a motion
    error, which dilutes the training signal for the velocity. Scoring pure
    advection against the future isolates motion: the score is lowest for
    motion fields that carry the present rain to where the future rain is,
    and, being a proper ensemble score, for a member spread of the motion
    that matches the actual displacement uncertainty.

    latest: (B, 1, H, W) analysis field in y; points: K departure points, each
    (B * M, 2, H, W), member-major; target, valid: (B, K, H, W).
    """
    b, k = target.shape[:2]
    m = points[0].shape[0] // b
    field = latest.float().repeat_interleave(m, dim=0)
    advected = torch.cat([sample(field, points[j]) for j in range(k)], dim=1)  # (B * M, K, H, W)
    return multiscale_crps(advected.reshape(b, m, k, *advected.shape[-2:]), target, valid, scales, alpha)["crps"]


def smoothness(velocity: torch.Tensor) -> torch.Tensor:
    """
    Horn-Schunck regularizer: mean squared spatial gradient of the velocity field (..., 2, h, w).

    Where there is no precipitation, neither the forecast nor the past
    constrains the motion (dry air advected is still dry). This prior fills
    such regions by smooth continuation of the constrained flow, as the
    steering winds themselves are smooth.
    """
    dx = velocity[..., :, 1:] - velocity[..., :, :-1]
    dy = velocity[..., 1:, :] - velocity[..., :-1, :]
    return dx.pow(2).mean() + dy.pow(2).mean()
