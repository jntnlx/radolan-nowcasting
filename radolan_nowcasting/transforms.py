"""
Unit conversions and log-rain transforms for RADOLAN-YW data.

Transform chain:
    raw RADOLAN (mm/5min) → rate (mm/h) → log-rain → normalized log-rain

Note: All functions accept both numpy arrays and PyTorch tensors.
"""

import json
import math
from pathlib import Path

import numpy as np

from .config import MAX_RAIN_RATE_MMH, MIN_RAIN_RATE_MMH, PROCESSED_CACHE_DIR

try:
    import torch
except ImportError:
    torch = None


# Normalization statistics

def load_normalization_stats(stats_path: Path | None = None) -> tuple[float, float]:
    """
    Load train-event mean and std from processed cache.

    - Data-dependent. Computed from train_event only.
    - Must match cache version used for training.
    """
    path = stats_path or (PROCESSED_CACHE_DIR / "stats.json")
    if not path.exists():
        raise FileNotFoundError(
            f"Normalization stats not found: {path}. "
            "Build the cache first (scripts/build_cache.py)."
        )
    with open(path) as f:
        stats = json.load(f)

    if stats.get("transform") != "log_rain_mmh":
        raise ValueError(f"Unexpected transform in {path}: {stats.get('transform')!r}")
    if stats.get("unit") != "mm/h":
        raise ValueError(f"Unexpected unit in {path}: {stats.get('unit')!r}")

    mean, std = float(stats["mean"]), float(stats["std"])
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError(f"Invalid normalization stats: mean={mean}, std={std}")
    return mean, std


def _resolve_stats(mean: float | None, std: float | None) -> tuple[float, float]:
    """Return explicit (mean, std) or load from cache."""
    if mean is None and std is None:
        return load_normalization_stats()
    if mean is None or std is None:
        raise ValueError("Pass both mean and std, or neither to auto-load.")
    return float(mean), float(std)


# Forward transforms (mm/h → log-rain → normalized)

def rate_mmh_to_log(rate_mmh):
    """
    Convert rain rate (mm/h) to log-rain: `10 * log10(max(rate, 0.01))`.

    `0.01 mm/h` floor avoids log(0) and maps approx. "dry" pixels
    to finite lower bound of -20 dB. Note: NaN inputs remain NaN!
    """
    if torch is not None and torch.is_tensor(rate_mmh):
        rate = rate_mmh.float()
        clamped = torch.clamp(rate, min=MIN_RAIN_RATE_MMH)
        result = 10.0 * torch.log10(clamped)
        return torch.where(torch.isnan(rate), float("nan"), result)

    arr = np.asarray(rate_mmh, dtype=np.float64)
    clamped = np.clip(arr, MIN_RAIN_RATE_MMH, None)
    result = 10.0 * np.log10(clamped)
    result = np.where(np.isnan(arr), np.nan, result)
    return float(result) if np.ndim(rate_mmh) == 0 else result


def log_to_rate_mmh(log_rain):
    """Invert log-rain back to mm/h: 10^(log_rain / 10)."""
    if torch is not None and torch.is_tensor(log_rain):
        return torch.pow(10.0, log_rain.float() / 10.0)
    arr = np.asarray(log_rain, dtype=np.float64)
    result = np.power(10.0, arr / 10.0)
    return float(result) if np.ndim(log_rain) == 0 else result


def normalize_log(log_rain, mean: float | None = None, std: float | None = None):
    """Normalize log-rain using train-event statistics."""
    m, s = _resolve_stats(mean, std)
    return (log_rain - m) / s


def denormalize_log(x, mean: float | None = None, std: float | None = None):
    """Reverse normalized log-rain to raw log-rain space."""
    m, s = _resolve_stats(mean, std)
    return x * s + m


# Composite transforms

def normalized_to_rate_mmh(x, mean: float | None = None, std: float | None = None):
    """Convert normalized log-rain to clipped physical rain rate (mm/h)."""
    log_rain = denormalize_log(x, mean=mean, std=std)
    log_min = 10.0 * math.log10(MIN_RAIN_RATE_MMH)
    log_max = 10.0 * math.log10(MAX_RAIN_RATE_MMH)
    if torch is not None and torch.is_tensor(log_rain):
        clipped = torch.clamp(log_rain, min=log_min, max=log_max)
    else:
        clipped = np.clip(log_rain, log_min, log_max)
    return log_to_rate_mmh(clipped)


def rate_mmh_threshold_to_normalized(
    threshold_mmh: float,
    mean: float | None = None,
    std: float | None = None,
) -> float:
    """Map a physical threshold (mm/h) to normalized log-rain space."""
    m, s = _resolve_stats(mean, std)
    threshold = max(float(threshold_mmh), MIN_RAIN_RATE_MMH)
    log_value = 10.0 * math.log10(threshold)
    return (log_value - m) / s
