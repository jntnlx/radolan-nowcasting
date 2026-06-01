"""
Deterministic verification metrics.

Implements standard nowcasting verification metrics:

- **RMSE**: Both normalized-log and physical (mm/h) space.

- **CSI/POD/FAR/FBIAS**: Multiple rain-rate thresholds. Categorical
  scores that correspond to correct prediction of rain exceeding threshold.

- **FSS**: Measures forecast accuracy at spatial neighbourhood scale instead 
  of pixel-by-pixel. High FSS corresponds to model locatiing precipitation 
  approximately right. Essential for evaluating convective precipitation
  where exact pixel alignment is not feasible.

- **Conditional bias**: Systematic over/under-prediction as function of
  observed rain rate. Shows if model dampens intense precipitation which
  corresponds to common failure mode of deterministic DL forecasts.

- **Below-threshold false alarms**: How often does model predict rain where
  none was observed. Important in operational contexts.

- **30 minute accumulation RMSE: Eval in accumulated precipitation space.

All metrics computed per batch and accumulated, and subseqeuntly finalized 
into a dictionary. Breakdowns per lead time included to diagnose forecast 
skill degradation.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .config import (
    FSS_NEIGHBORHOODS,
    FSS_THRESHOLDS_MMH,
    SEQ_LEN_OUT,
    THRESHOLDS_MMH,
)
from .transforms import normalized_to_rate_mmh, rate_mmh_threshold_to_normalized


BIAS_BINS_MMH = (
    ("dry", 0.0, 0.1),
    ("0_1_to_1", 0.1, 1.0),
    ("1_to_5", 1.0, 5.0),
    ("5_to_10", 5.0, 10.0),
    ("10_to_20", 10.0, 20.0),
    ("gt_20", 20.0, math.inf),
)


def _counts_to_scores(hits: int, misses: int, false_alarms: int) -> dict[str, float]:
    """Compute categorical scores from contingency table counts."""
    csi_denom = hits + misses + false_alarms
    pod_denom = hits + misses
    far_denom = hits + false_alarms
    return {
        "csi": hits / csi_denom if csi_denom > 0 else float("nan"),
        "pod": hits / pod_denom if pod_denom > 0 else float("nan"),
        "far": false_alarms / far_denom if far_denom > 0 else float("nan"),
        "fbias": (hits + false_alarms) / pod_denom if pod_denom > 0 else float("nan"),
    }


def _prefixed(prefix: str, key: str) -> str:
    return f"{prefix}_{key}" if prefix else key


@dataclass
class ForecastMetricAccumulator:
    """Accumulate deterministic metrics over batches for one forecast source."""

    mean: float
    std: float

    # Continuous scores
    total_se_norm: float = 0.0
    total_se_norm_rain: float = 0.0
    total_pixels: int = 0
    rain_pixels: int = 0
    total_se_mmh: float = 0.0
    total_abs_mmh: float = 0.0
    total_pixels_mmh: int = 0
    total_se_accum30: float = 0.0
    total_pixels_accum30: int = 0

    # Continuous scores (per lead time)
    per_lead_se_norm: list[float] = field(default_factory=lambda: [0.0] * SEQ_LEN_OUT)
    per_lead_se_mmh: list[float] = field(default_factory=lambda: [0.0] * SEQ_LEN_OUT)
    per_lead_pixels: list[int] = field(default_factory=lambda: [0] * SEQ_LEN_OUT)

    # Categorical scores at each threshold
    hits: dict[float, int] = field(default_factory=lambda: {t: 0 for t in THRESHOLDS_MMH})
    misses: dict[float, int] = field(default_factory=lambda: {t: 0 for t in THRESHOLDS_MMH})
    false_alarms: dict[float, int] = field(default_factory=lambda: {t: 0 for t in THRESHOLDS_MMH})

    # Categorical (per lead)
    per_lead_hits: dict[float, list[int]] = field(
        default_factory=lambda: {t: [0] * SEQ_LEN_OUT for t in THRESHOLDS_MMH}
    )
    per_lead_misses: dict[float, list[int]] = field(
        default_factory=lambda: {t: [0] * SEQ_LEN_OUT for t in THRESHOLDS_MMH}
    )
    per_lead_false_alarms: dict[float, list[int]] = field(
        default_factory=lambda: {t: [0] * SEQ_LEN_OUT for t in THRESHOLDS_MMH}
    )

    # FSS
    fss_mse: dict[float, dict[int, float]] = field(
        default_factory=lambda: {t: {n: 0.0 for n in FSS_NEIGHBORHOODS} for t in FSS_THRESHOLDS_MMH}
    )
    fss_ref: dict[float, dict[int, float]] = field(
        default_factory=lambda: {t: {n: 0.0 for n in FSS_NEIGHBORHOODS} for t in FSS_THRESHOLDS_MMH}
    )

    # False alarms below threshold
    below_threshold_fa: dict[float, int] = field(
        default_factory=lambda: {t: 0 for t in THRESHOLDS_MMH}
    )
    below_threshold_pixels: dict[float, int] = field(
        default_factory=lambda: {t: 0 for t in THRESHOLDS_MMH}
    )

    # Conditional bias
    bias_pred_sum: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name, _, _ in BIAS_BINS_MMH}
    )
    bias_target_sum: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name, _, _ in BIAS_BINS_MMH}
    )
    bias_count: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name, _, _ in BIAS_BINS_MMH}
    )

    def update(self, prediction_norm: torch.Tensor, target_norm: torch.Tensor) -> None:
        """Accumulate metrics for one batch of normalized-log predictions."""
        prediction_norm = prediction_norm.detach()
        target_norm = target_norm.detach()
        batch, leads, h, w = target_norm.shape

        # Continuous error in normalized-log space
        se_norm = (prediction_norm - target_norm).pow(2)
        self.total_se_norm += float(se_norm.sum())
        self.total_pixels += int(target_norm.numel())

        # Rain-only RMSE (pixels > 0.1 mm/h in target)
        rain_threshold = rate_mmh_threshold_to_normalized(0.1, self.mean, self.std)
        rain_mask = target_norm > rain_threshold
        if rain_mask.any():
            self.total_se_norm_rain += float(se_norm[rain_mask].sum())
            self.rain_pixels += int(rain_mask.sum())

        # Errors in physical space
        pred_mmh = normalized_to_rate_mmh(prediction_norm, self.mean, self.std)
        target_mmh = normalized_to_rate_mmh(target_norm, self.mean, self.std)
        err_mmh = pred_mmh - target_mmh
        self.total_se_mmh += float(err_mmh.pow(2).sum())
        self.total_abs_mmh += float(err_mmh.abs().sum())
        self.total_pixels_mmh += int(target_norm.numel())

        # 30 minute accumulation error
        pred_accum = pred_mmh.sum(dim=1) / 12.0
        target_accum = target_mmh.sum(dim=1) / 12.0
        self.total_se_accum30 += float((pred_accum - target_accum).pow(2).sum())
        self.total_pixels_accum30 += int(pred_accum.numel())

        # Continuous (per lead)
        for lead in range(SEQ_LEN_OUT):
            self.per_lead_se_norm[lead] += float(se_norm[:, lead].sum())
            self.per_lead_se_mmh[lead] += float(err_mmh[:, lead].pow(2).sum())
            self.per_lead_pixels[lead] += int(target_norm[:, lead].numel())

        self._update_categorical(prediction_norm, target_norm)
        self._update_fss(prediction_norm, target_norm, batch, leads, h, w)
        self._update_below_threshold_fa(pred_mmh, target_mmh)
        self._update_conditional_bias(pred_mmh, target_mmh)

    def _update_categorical(self, pred_norm, target_norm) -> None:
        for tau in THRESHOLDS_MMH:
            thresh = rate_mmh_threshold_to_normalized(tau, self.mean, self.std)
            p = pred_norm > thresh
            o = target_norm > thresh
            self.hits[tau] += int((p & o).sum())
            self.misses[tau] += int((~p & o).sum())
            self.false_alarms[tau] += int((p & ~o).sum())
            for lead in range(SEQ_LEN_OUT):
                pl, ol = p[:, lead], o[:, lead]
                self.per_lead_hits[tau][lead] += int((pl & ol).sum())
                self.per_lead_misses[tau][lead] += int((~pl & ol).sum())
                self.per_lead_false_alarms[tau][lead] += int((pl & ~ol).sum())

    def _update_fss(self, pred_norm, target_norm, batch, leads, h, w) -> None:
        for tau in FSS_THRESHOLDS_MMH:
            thresh = rate_mmh_threshold_to_normalized(tau, self.mean, self.std)
            p_bin = (pred_norm > thresh).float().reshape(batch * leads, 1, h, w)
            o_bin = (target_norm > thresh).float().reshape(batch * leads, 1, h, w)
            for n in FSS_NEIGHBORHOODS:
                pad = n // 2
                p_frac = F.avg_pool2d(p_bin, n, stride=1, padding=pad, count_include_pad=False)
                o_frac = F.avg_pool2d(o_bin, n, stride=1, padding=pad, count_include_pad=False)
                self.fss_mse[tau][n] += float((p_frac - o_frac).pow(2).sum())
                self.fss_ref[tau][n] += float((p_frac.pow(2) + o_frac.pow(2)).sum())

    def _update_below_threshold_fa(self, pred_mmh, target_mmh) -> None:
        for tau in THRESHOLDS_MMH:
            below = target_mmh <= tau
            self.below_threshold_pixels[tau] += int(below.sum())
            self.below_threshold_fa[tau] += int(((pred_mmh > tau) & below).sum())

    def _update_conditional_bias(self, pred_mmh, target_mmh) -> None:
        for name, lo, hi in BIAS_BINS_MMH:
            if math.isinf(hi):
                mask = target_mmh > lo
            elif lo == 0.0:
                mask = target_mmh <= hi
            else:
                mask = (target_mmh > lo) & (target_mmh <= hi)
            count = int(mask.sum())
            if count == 0:
                continue
            self.bias_count[name] += count
            self.bias_pred_sum[name] += float(pred_mmh[mask].sum())
            self.bias_target_sum[name] += float(target_mmh[mask].sum())

    def finalize(
        self,
        *,
        prefix: str = "",
        reference: "ForecastMetricAccumulator | None" = None,
    ) -> dict[str, float]:
        """Return a flat metric dictionary."""
        out: dict[str, float] = {}

        # Continuous scores
        out[_prefixed(prefix, "rmse_log_norm")] = math.sqrt(
            self.total_se_norm / max(self.total_pixels, 1)
        )
        out[_prefixed(prefix, "rmse_log_norm_rain")] = (
            math.sqrt(self.total_se_norm_rain / self.rain_pixels)
            if self.rain_pixels > 0 else float("nan")
        )
        out[_prefixed(prefix, "rmse_mmh")] = math.sqrt(
            self.total_se_mmh / max(self.total_pixels_mmh, 1)
        )
        out[_prefixed(prefix, "mae_mmh")] = (
            self.total_abs_mmh / max(self.total_pixels_mmh, 1)
        )
        out[_prefixed(prefix, "accum30_rmse_mm")] = math.sqrt(
            self.total_se_accum30 / max(self.total_pixels_accum30, 1)
        )

        # Skill relative to persistence
        if reference is not None:
            out[_prefixed(prefix, "skill_score")] = (
                1.0 - self.total_se_norm / reference.total_se_norm
                if reference.total_se_norm > 0 else float("nan")
            )

        # Continuous (per lead)
        for lead in range(SEQ_LEN_OUT):
            denom = max(self.per_lead_pixels[lead], 1)
            out[_prefixed(prefix, f"rmse_log_norm_t{lead + 1}")] = math.sqrt(
                self.per_lead_se_norm[lead] / denom
            )
            out[_prefixed(prefix, f"rmse_mmh_t{lead + 1}")] = math.sqrt(
                self.per_lead_se_mmh[lead] / denom
            )
            if reference is not None:
                ref = reference.per_lead_se_norm[lead]
                out[_prefixed(prefix, f"skill_score_t{lead + 1}")] = (
                    1.0 - self.per_lead_se_norm[lead] / ref if ref > 0 else float("nan")
                )

        # Categorical scores
        for tau in THRESHOLDS_MMH:
            scores = _counts_to_scores(self.hits[tau], self.misses[tau], self.false_alarms[tau])
            for name, value in scores.items():
                out[_prefixed(prefix, f"{name}_{tau}_mmh")] = value

            # False alarm rate below threshold
            bd = self.below_threshold_pixels[tau]
            fa_rate = self.below_threshold_fa[tau] / bd if bd > 0 else float("nan")
            out[_prefixed(prefix, f"below_threshold_far_{tau}_mmh")] = fa_rate

            # Categorical (per lead)
            for lead in range(SEQ_LEN_OUT):
                scores = _counts_to_scores(
                    self.per_lead_hits[tau][lead],
                    self.per_lead_misses[tau][lead],
                    self.per_lead_false_alarms[tau][lead],
                )
                for name, value in scores.items():
                    out[_prefixed(prefix, f"{name}_{tau}_mmh_t{lead + 1}")] = value

        # FSS
        for tau in FSS_THRESHOLDS_MMH:
            for n in FSS_NEIGHBORHOODS:
                ref = self.fss_ref[tau][n]
                mse = self.fss_mse[tau][n]
                # FSS = 1 when fss_ref ≈ 0: perfect agreement on "no rain at
                # this threshold". Standard convention.
                out[_prefixed(prefix, f"fss_{tau}_mmh_n{n}")] = (
                    1.0 - mse / ref if ref > 1e-10 else 1.0
                )

        # Conditional bias
        for name, _, _ in BIAS_BINS_MMH:
            count = self.bias_count[name]
            if count == 0:
                out[_prefixed(prefix, f"conditional_bias_mmh_{name}")] = float("nan")
                continue
            pred_mean = self.bias_pred_sum[name] / count
            target_mean = self.bias_target_sum[name] / count
            out[_prefixed(prefix, f"conditional_bias_mmh_{name}")] = pred_mean - target_mean

        return out
