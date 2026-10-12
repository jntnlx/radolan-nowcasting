"""
Training samples: radar windows drawn from the frame store.

A sample is a spatial crop of `history` past frames and `leads` future frames
around an analysis time t0 (the last observed frame).

Sampling and why it is unbiased
-------------------------------
Precipitation is rare: most crops at most times are dry. Samples are drawn
with probability increasing with the rain present *at the analysis time*,

    q(t0, crop)  proportional to  floor + mean rain rate in the crop at t0 (mm/h),

first choosing t0 by the domain-wide amount, then the crop within it. Because
q depends only on information available at t0 (the inputs), the conditional
distribution of the future given the inputs is unchanged by this selection.
A proper scoring rule is minimized by that conditional distribution under any
input-only reweighting, so the oversampling of rainy situations spends model
capacity where it matters without biasing the optimal forecast. The floor
keeps dry scenes, where initiation happens, in the training distribution.

Workers return raw depth integers; conversion to model space happens on the
GPU (`to_model_space`), which halves the host-to-device traffic.
"""

import math

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .config import DOMAIN_SHAPE
from .radklim import DEPTH_QUANTUM_MM, NODATA
from .store import TILE, TILES, Season


def time_context(t) -> np.ndarray:
    """sin/cos of UTC time of day and of day of year."""
    day = 2 * math.pi * (t.hour * 60 + t.minute) / 1440
    year = 2 * math.pi * (t.timetuple().tm_yday - 1) / 365.25
    return np.array([math.sin(day), math.cos(day), math.sin(year), math.cos(year)], dtype=np.float32)


class Windows:
    """Sampling distribution over (season, t0, crop) for a set of years."""

    def __init__(self, years, history: int, leads: int, crop: tuple[int, int] = DOMAIN_SHAPE,
                 floor: float = 0.05, min_valid: float = 0.3):
        if crop[0] % TILE or crop[1] % TILE:
            raise ValueError(f"crop {crop} must be a multiple of the {TILE} km tile")
        self.seasons = [Season(y) for y in years]
        self.history, self.leads, self.crop, self.floor = history, leads, crop, floor
        self.crop_tiles = (crop[0] // TILE, crop[1] // TILE)
        self.offsets = [(r, c) for r in range(TILES[0] - self.crop_tiles[0] + 1)
                        for c in range(TILES[1] - self.crop_tiles[1] + 1)]

        times, weights = [], []
        for s, season in enumerate(self.seasons):
            t0 = season.complete_windows(history, leads)
            rain, valid = season.tile_rain[t0].astype(np.float32), season.tile_valid[t0].astype(np.float32)
            domain_rain = (rain * valid).sum((1, 2)) / valid.sum((1, 2)).clip(min=1e-6)
            keep = valid.mean((1, 2)) >= min_valid
            times.append(np.stack([np.full(keep.sum(), s), t0[keep]], axis=1))
            weights.append(floor + domain_rain[keep])
        self.times = np.concatenate(times)
        self.time_weights = np.concatenate(weights)
        self.time_weights /= self.time_weights.sum()

    def draw(self, rng: np.random.Generator) -> tuple[int, int, int, int]:
        """Draw (season index, t0, tile row, tile col) from q."""
        s, t0 = self.times[rng.choice(len(self.times), p=self.time_weights)]
        season = self.seasons[s]
        h, w = self.crop_tiles
        rain, valid = season.tile_rain[t0].astype(np.float32), season.tile_valid[t0].astype(np.float32)
        crop_rain = np.array([(rain[r:r + h, c:c + w] * valid[r:r + h, c:c + w]).sum()
                              / max(valid[r:r + h, c:c + w].sum(), 1e-6) for r, c in self.offsets])
        p = self.floor + crop_rain
        r, c = self.offsets[rng.choice(len(self.offsets), p=p / p.sum())]
        return int(s), int(t0), r, c

    def load(self, s: int, t0: int, r: int, c: int) -> dict:
        season = self.seasons[s]
        rows = slice(r * TILE, r * TILE + self.crop[0])
        cols = slice(c * TILE, c * TILE + self.crop[1])
        depth = season.frames(t0 - self.history + 1, t0 + self.leads + 1, rows, cols)
        return {
            "depth": torch.from_numpy(np.ascontiguousarray(depth).view(np.int16)),  # bit-identical; NODATA reads as -1
            "context": torch.from_numpy(time_context(season.time(t0))),
            "key": torch.tensor([season.year, t0, r * TILE, c * TILE]),
        }


class RandomWindows(IterableDataset):
    """Endless stream of samples from q; each DataLoader worker has its own seeded generator."""

    def __init__(self, windows: Windows, seed: int):
        self.windows, self.seed = windows, seed

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        while True:
            yield self.windows.load(*self.windows.draw(rng))


class FixedWindows(Dataset):
    """A reproducible finite sample from q, for validation during training."""

    def __init__(self, windows: Windows, n: int, seed: int):
        rng = np.random.default_rng(seed)
        self.windows = windows
        self.keys = [windows.draw(rng) for _ in range(n)]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        return self.windows.load(*self.keys[i])


def to_model_space(batch: dict, history: int, device) -> dict:
    """Depth integers (B, T, H, W) -> inputs/targets in y = log(1 + R) and validity masks."""
    depth = batch["depth"].to(device, non_blocking=True).to(torch.int32) & 0xFFFF  # undo int16 view
    valid = depth != int(NODATA)
    rate = torch.where(valid, depth.float() * (DEPTH_QUANTUM_MM * 12), 0.0)
    y = torch.log1p(rate)
    return {
        "inputs": y[:, :history],
        "input_valid": valid[:, :history].all(dim=1, keepdim=True),
        "targets": y[:, history:],
        "target_valid": valid[:, history:],
        "context": batch["context"].to(device, non_blocking=True),
    }
