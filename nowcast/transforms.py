"""
Rain rate <-> model space.

The model works in y = log(1 + R), with R the rain rate in mm/h.

Why this transform: a CRPS computed in y equals the integral of Brier scores
over all rain-rate thresholds r, weighted by dy/dr = 1 / (1 + r)
(Gneiting & Ranjan 2011, threshold-weighted CRPS). Above ~1 mm/h every decade
of intensity (1-10, 10-100 mm/h) therefore counts equally, while drizzle below
the ~0.1 mm/h radar detection floor counts little. Dry pixels map to exactly
y = 0, so no normalization statistics are needed and the transform is fixed
independently of the data.
"""

import torch
import torch.nn.functional as F


def rate_to_y(rate: torch.Tensor) -> torch.Tensor:
    return torch.log1p(rate)


def y_to_rate(y: torch.Tensor) -> torch.Tensor:
    return torch.expm1(y.clamp(min=0))


def pool_rate(rate: torch.Tensor, valid: torch.Tensor, scale: int, min_valid: float = 0.5):
    """
    Average rain rate over non-overlapping scale x scale windows, ignoring invalid pixels.

    Averaging happens on rates, not on y: an area-mean rate is the physically
    meaningful quantity (it conserves water). Windows with less than
    `min_valid` valid pixels are marked invalid.

    rate:  (..., H, W);  valid: bool, broadcastable to rate.
    Returns (pooled rate, pooled validity), each (..., H // scale, W // scale).
    """
    if scale == 1:
        return rate, valid.expand_as(rate)
    shape = rate.shape
    weight = valid.expand_as(rate).to(rate.dtype)
    num = F.avg_pool2d((rate * weight).reshape(-1, 1, *shape[-2:]), scale)
    den = F.avg_pool2d(weight.reshape(-1, 1, *shape[-2:]), scale)
    pooled = (num / den.clamp(min=1e-6)).reshape(*shape[:-2], *num.shape[-2:])
    pooled_valid = (den >= min_valid).reshape(pooled.shape)
    return pooled, pooled_valid
