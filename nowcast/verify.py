"""
Forecast verification on the radar domain.

`Scores` accumulates sufficient statistics (sums and counts) over analysis
times, per lead time, so that every score is pooled over all verified pixels
rather than averaged over cases. Forecasts are ensembles of rain rate
(M, K, H, W) in mm/h; a deterministic forecast is the case M = 1.
Observations are rain rates (K, H, W) with NaN where the radar has no data;
those pixels are excluded from every score.

Scores
    CRPS        fair ensemble CRPS, in y = log(1 + R) (the training metric,
                i.e. Brier scores integrated over thresholds with weight
                1 / (1 + r)) at 1, 4, 16, 64 km, and in mm/h at 1 km.
                For M = 1 this is the mean absolute error.
    Brier       Brier score of exceedance probabilities and reliability table.
    FSS         fractions skill score of each member (Roberts & Lean 2008),
                pooled over members, per threshold and neighbourhood size.
    Bias        frequency bias: forecast / observed exceedance area.
    Spread      ensemble variance vs. squared error of the ensemble mean in y.
                Calibrated: mse = (M + 1) / M * variance.
    Rank        rank histogram of observations within the ensemble (in y, at
                pixels where observation or any member is wet).
    Spectrum    radially averaged power spectrum of y on a fully covered
                512 x 512 km window, for observations and member 0.
"""

import math

import torch
import torch.nn.functional as F

from .losses import crps_ensemble
from .transforms import pool_rate, rate_to_y

CRPS_SCALES = (1, 4, 16, 64)
THRESHOLDS = (0.5, 1.0, 5.0, 10.0, 20.0)  # mm/h
FSS_THRESHOLDS = (1.0, 5.0, 10.0)
FSS_WINDOWS = (1, 5, 11, 21, 41)          # km
SPECTRUM_WINDOW = (slice(256, 768), slice(160, 672))  # inside permanent radar coverage


def _fractions(binary: torch.Tensor, valid: torch.Tensor, n: int) -> torch.Tensor:
    """Fraction of valid pixels exceeding a threshold in an n x n neighbourhood."""
    if n == 1:
        return binary
    pad = n // 2
    num = F.avg_pool2d((binary * valid).unsqueeze(1), n, stride=1, padding=pad, count_include_pad=False)
    den = F.avg_pool2d(valid.unsqueeze(1), n, stride=1, padding=pad, count_include_pad=False)
    return (num / den.clamp(min=1e-6)).squeeze(1)


def radial_spectrum(field: torch.Tensor) -> torch.Tensor:
    """Radially averaged power spectrum of (..., N, N) fields with a Hann taper; index = wavenumber."""
    n = field.shape[-1]
    taper = torch.hann_window(n, device=field.device, dtype=field.dtype)
    window = taper[:, None] * taper[None]
    power = torch.fft.fft2((field - field.mean((-2, -1), keepdim=True)) * window).abs() ** 2
    k = torch.fft.fftfreq(n, device=field.device) * n
    radius = torch.sqrt(k[:, None] ** 2 + k[None] ** 2).round().long().clamp(max=n // 2)
    flat = power.reshape(-1, n * n)
    out = torch.zeros(flat.shape[0], n // 2 + 1, device=field.device, dtype=field.dtype)
    out.index_add_(1, radius.reshape(-1), flat)
    counts = torch.bincount(radius.reshape(-1), minlength=n // 2 + 1).to(field.dtype)
    return (out / counts).reshape(*field.shape[:-2], n // 2 + 1)


class Scores:
    def __init__(self, leads: int, members: int, device="cuda"):
        self.leads, self.members, self.device = leads, members, device
        z = lambda *shape: torch.zeros(*shape, leads, dtype=torch.float64, device=device)
        self.n = z()
        self.crps = {s: z() for s in CRPS_SCALES}
        self.crps_n = {s: z() for s in CRPS_SCALES}
        self.crps_mmh = z()
        self.brier = z(len(THRESHOLDS))
        self.obs_events = z(len(THRESHOLDS))
        self.fcst_events = z(len(THRESHOLDS))
        self.reliability_n = z(len(THRESHOLDS), members + 1)       # pixels per forecast probability
        self.reliability_obs = z(len(THRESHOLDS), members + 1)     # of which observed events
        self.fss_num = z(len(FSS_THRESHOLDS), len(FSS_WINDOWS))
        self.fss_den = z(len(FSS_THRESHOLDS), len(FSS_WINDOWS))
        self.mse_mean = z()
        self.variance = z()
        self.rank = z(members + 1)
        self.spectrum_obs = z(257)
        self.spectrum_fcst = z(257)
        self.cases = 0

    @torch.no_grad()
    def add(self, forecast: torch.Tensor, obs: torch.Tensor, chunk: int = 1) -> None:
        """forecast: (M, K, H, W) mm/h;  obs: (K, H, W) mm/h, NaN = no data. Leads are processed in chunks."""
        if forecast.shape[0] != self.members:
            raise ValueError(f"expected {self.members} members, got {forecast.shape[0]}")
        for k in range(0, self.leads, chunk):
            sl = slice(k, min(k + chunk, self.leads))
            self._add(forecast[:, sl].to(self.device, torch.float32), obs[sl].to(self.device, torch.float32), sl)
        self.cases += 1

    def _add(self, forecast, obs, sl) -> None:
        m = self.members
        valid = torch.isfinite(obs)
        obs = torch.where(valid, obs, 0.0)
        forecast = forecast.clamp(min=0)
        lead = lambda t: t.flatten(start_dim=-2).sum(-1).double()  # sum over space -> (..., leads in chunk)

        self.n[sl] += lead(valid)
        y_f, y_o = rate_to_y(forecast), rate_to_y(obs)
        for s in CRPS_SCALES:
            if s > min(obs.shape[-2:]):
                continue  # scale larger than the field (only in small tests)
            if s == 1:
                pf, po, ok = y_f, y_o, valid
            else:
                pf, _ = pool_rate(forecast, valid, s)
                po, ok = pool_rate(obs, valid, s)
                pf, po = rate_to_y(pf), rate_to_y(po)
            self.crps[s][sl] += lead(crps_ensemble(pf, po, alpha=1.0) * ok)
            self.crps_n[s][sl] += lead(ok)
        self.crps_mmh[sl] += lead(crps_ensemble(forecast, obs, alpha=1.0) * valid)

        for i, thr in enumerate(THRESHOLDS):
            event = (obs > thr) & valid
            count = (forecast > thr).sum(0)  # members exceeding: forecast probability = count / M
            self.brier[i, sl] += lead((count / m - event.float()) ** 2 * valid)
            self.obs_events[i, sl] += lead(event)
            self.fcst_events[i, sl] += lead(count / m * valid)
            for c in range(m + 1):
                at = (count == c) & valid
                self.reliability_n[i, c, sl] += lead(at)
                self.reliability_obs[i, c, sl] += lead(at & event)

        validf = valid.float()
        for i, thr in enumerate(FSS_THRESHOLDS):
            o = (obs > thr).float()
            for j, n in enumerate(FSS_WINDOWS):
                fo = _fractions(o, validf, n)
                for f in (forecast > thr).float():
                    ff = _fractions(f, validf, n)
                    self.fss_num[i, j, sl] += lead((ff - fo) ** 2 * validf)
                    self.fss_den[i, j, sl] += lead((ff**2 + fo**2) * validf)

        self.mse_mean[sl] += lead((y_f.mean(0) - y_o) ** 2 * valid)
        if m > 1:
            self.variance[sl] += lead(y_f.var(0) * valid)
            wet = valid & ((y_o > 0) | (y_f > 0).any(0))
            # Random tie-breaking: dry members tie with a dry observation and must be ranked uniformly.
            jitter = lambda t: t + 1e-6 * torch.rand(t.shape, device=self.device)
            rank = (jitter(y_f) < jitter(y_o)).sum(0)
            for r in range(m + 1):
                self.rank[r, sl] += lead((rank == r) & wet)

        rows, cols = SPECTRUM_WINDOW
        if obs.shape[-2] >= rows.stop and obs.shape[-1] >= cols.stop and bool(valid[:, rows, cols].all()):
            self.spectrum_obs[:, sl] += radial_spectrum(y_o[:, rows, cols]).double().T
            self.spectrum_fcst[:, sl] += radial_spectrum(y_f[0, :, rows, cols]).double().T

    def summary(self) -> dict:
        """Pooled scores, each a list over lead times (or nested by threshold / window)."""
        div = lambda a, b: (a / b).cpu().tolist()
        out = {"cases": self.cases, "members": self.members, "pixels": self.n.cpu().tolist()}
        for s in CRPS_SCALES:
            out[f"crps_y_{s}km"] = div(self.crps[s], self.crps_n[s])
        out["crps_mmh"] = div(self.crps_mmh, self.n)
        for i, thr in enumerate(THRESHOLDS):
            base = self.obs_events[i] / self.n
            out[f"brier_{thr}"] = div(self.brier[i], self.n)
            out[f"bss_{thr}"] = (1 - (self.brier[i] / self.n) / (base * (1 - base))).cpu().tolist()
            out[f"freq_bias_{thr}"] = div(self.fcst_events[i], self.obs_events[i])
            out[f"reliability_{thr}"] = {  # per lead: observed frequency for forecast probability c / M
                "observed_frequency": div(self.reliability_obs[i].T, self.reliability_n[i].T),
                "pixels": self.reliability_n[i].T.cpu().tolist(),
            }
        for i, thr in enumerate(FSS_THRESHOLDS):
            for j, n in enumerate(FSS_WINDOWS):
                out[f"fss_{thr}_{n}km"] = (1 - self.fss_num[i, j] / self.fss_den[i, j]).cpu().tolist()
        out["rmse_mean_y"] = (self.mse_mean / self.n).sqrt().cpu().tolist()
        if self.members > 1:
            out["spread_y"] = (self.variance / self.n * (self.members + 1) / self.members).sqrt().cpu().tolist()
            out["rank_histogram"] = (self.rank / self.rank.sum(0)).T.cpu().tolist()
        out["spectrum_obs"] = self.spectrum_obs.T.cpu().tolist()
        out["spectrum_fcst"] = self.spectrum_fcst.T.cpu().tolist()
        return out
