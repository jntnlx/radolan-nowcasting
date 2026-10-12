"""
Classical reference nowcasts (pysteps).

    persistence     the latest observation, unchanged (Eulerian persistence)
    extrapolation   Lagrangian persistence: Lucas-Kanade optical flow and
                    semi-Lagrangian advection of the latest field
    steps           STEPS (Bowler et al. 2006; Seed 2003), the operational
                    probabilistic radar nowcast at several services, including
                    DWD (STEPS-DWD). It advects a cascade decomposition of the
                    field, lets each scale decay with its own autoregressive
                    model (small scales lose predictability first), and adds
                    stochastic noise with the observed spatial correlation.

All take rain rates (T, H, W) in mm/h, oldest first, NaN where the radar has
no data, and return an ensemble (M, K, H, W) of rain rates. Missing data are
treated as dry, the same information the network receives.
"""

import numpy as np
from pysteps import motion, nowcasts
from pysteps.utils import transformation

RAIN_THRESHOLD = 0.1  # mm/h, below which pysteps treats pixels as dry
DB_ZERO = -15.0       # dB value assigned to dry pixels


def _to_db(rate: np.ndarray) -> np.ndarray:
    rate = np.nan_to_num(rate, nan=0.0)
    db, _ = transformation.dB_transform(rate, {"transform": None, "zerovalue": 0.0, "threshold": RAIN_THRESHOLD},
                                        threshold=RAIN_THRESHOLD, zerovalue=DB_ZERO)
    return db


def _from_db(db: np.ndarray) -> np.ndarray:
    rate = 10 ** (db / 10)
    rate[~np.isfinite(rate) | (db <= DB_ZERO) | (rate < RAIN_THRESHOLD)] = 0.0
    return rate.astype(np.float32)


def velocity(history: np.ndarray) -> np.ndarray:
    """Lucas-Kanade motion field (2, H, W) in pixels per step from the last three frames."""
    return motion.get_method("lucaskanade")(_to_db(history[-3:]))


def persistence(history: np.ndarray, leads: int) -> np.ndarray:
    latest = np.nan_to_num(history[-1], nan=0.0).astype(np.float32)
    return np.broadcast_to(latest, (1, leads, *latest.shape)).copy()


def extrapolation(history: np.ndarray, leads: int) -> np.ndarray:
    db = nowcasts.get_method("extrapolation")(_to_db(history[-1:])[0], velocity(history), leads)
    return _from_db(np.nan_to_num(db, nan=DB_ZERO))[None]


def steps(history: np.ndarray, leads: int, members: int = 20, seed: int = 0, workers: int = 1, batch: int = 10) -> np.ndarray:
    """
    STEPS ensemble with 8 cascade levels.

    Members are independent given their seeds (own velocity perturbation and
    noise; probability matching acts per member), so the ensemble is computed
    in independently seeded batches to bound memory on the full domain
    (~8 GB per 20 members otherwise). Each lead is stored as float16 as soon
    as it is computed.
    """
    out = np.empty((members, leads, *history.shape[-2:]), dtype=np.float16)
    db, motion_field = _to_db(history[-3:]), velocity(history)
    for start in range(0, members, batch):
        n = min(batch, members - start)
        lead = iter(range(leads))

        def keep(fields, start=start, n=n, lead=lead):  # (n, H, W) in dB
            out[start:start + n, next(lead)] = _from_db(np.nan_to_num(fields, nan=DB_ZERO))

        nowcasts.get_method("steps")(
            db, motion_field, leads,
            n_ens_members=n, n_cascade_levels=8,
            precip_thr=10 * np.log10(RAIN_THRESHOLD), kmperpixel=1.0, timestep=5,
            noise_method="nonparametric", vel_pert_method="bps", mask_method="incremental",
            probmatching_method="cdf", seed=seed * 1000 + start, num_workers=workers, fft_method="pyfftw",
            callback=keep, return_output=False,
        )
    return out
