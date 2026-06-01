"""
Mixed loss.

Currently: Three components employed to train model:

1. **Huber loss in normalized-log space**: 
   Primary regression target.
   Smooth L1 (Huber) is robust to outliers induced by extreme precipitation events.
   Remains differentiable everywhere. Operating in log space to equalize
   across all rain rates.

2. **Physical MAE in mm/h**: 
   Enables model is penalized for prediction errors in real precipitation units. 
   Prevents minimization of log-space error only while large absolute errors 
   are not considered (e.g. heavy rain)

3. **Threshold-weighted MAE**: up-weights errors where the target exceeds
   standard heavy-rain thresholds (1, 5, 10, 20 mm/h).  This focuses
   learning capacity on the most operationally relevant situations: errors
   at 15 mm/h matter far more than errors at 0.1 mm/h.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HEAVY_RAIN_THRESHOLDS_MMH
from .transforms import normalized_to_rate_mmh, rate_mmh_threshold_to_normalized


class ForecastLoss(nn.Module):
    """Combined loss for the residual U-Net nowcasting baseline.

    Args:
        mean, std: Normalization statistics from the training cache.
        physical_mae_weight: Weight for the mm/h MAE term.
        threshold_weight: Weight for the threshold-weighted MAE term.
        gradient_weight: Weight for spatial gradient penalty (0 = disabled).
    """

    def __init__(
        self,
        mean: float,
        std: float,
        *,
        physical_mae_weight: float = 0.1,
        threshold_weight: float = 0.05,
        gradient_weight: float = 0.0,
    ):
        super().__init__()
        self.mean = float(mean)
        self.std = float(std)
        self.physical_mae_weight = physical_mae_weight
        self.threshold_weight = threshold_weight
        self.gradient_weight = gradient_weight

        # Pre-compute threshold values in normalized-log space
        self.thresholds_norm = [
            rate_mmh_threshold_to_normalized(t, self.mean, self.std)
            for t in HEAVY_RAIN_THRESHOLDS_MMH
        ]

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}"
            )

        # 1. Huber loss in normalized-log space
        huber = F.smooth_l1_loss(prediction, target)

        # 2. Physical MAE in mm/h
        pred_mmh = normalized_to_rate_mmh(prediction, self.mean, self.std)
        target_mmh = normalized_to_rate_mmh(target, self.mean, self.std)
        physical_mae = (pred_mmh - target_mmh).abs().mean()

        # 3. Threshold-weighted MAE: extra penalty where target is heavy rain
        threshold_mae = self._threshold_weighted_mae(prediction, target)

        # Combine
        total = huber + self.physical_mae_weight * physical_mae
        total = total + self.threshold_weight * threshold_mae

        parts = {
            "huber_norm": huber,
            "physical_mae_mmh": physical_mae,
            "threshold_weighted_mae_mmh": threshold_mae,
        }

        # Spatial gradient penalty — disabled by default (gradient_weight=0.0).
        # Planned extension for spectral/gradient loss to reduce blurring;
        # enable via ForecastLoss(gradient_weight=...).
        if self.gradient_weight > 0:
            grad_loss = self._gradient_loss(prediction, target)
            total = total + self.gradient_weight * grad_loss
            parts["gradient_loss"] = grad_loss

        return total, parts

    def _threshold_weighted_mae(
        self, prediction: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        """MAE with cumulative weights above each rain threshold.

        Each threshold adds a mask, so pixels exceeding all N thresholds
        receive (1 + N)× the base error — intentionally focusing gradient
        on the heaviest precipitation.
        """
        pred_mmh = normalized_to_rate_mmh(prediction, self.mean, self.std)
        target_mmh = normalized_to_rate_mmh(target, self.mean, self.std)
        error = (pred_mmh - target_mmh).abs()

        weighted = error.clone()
        for threshold_norm in self.thresholds_norm:
            mask = target > threshold_norm
            weighted = weighted + error * mask.float()

        return weighted.mean()

    @staticmethod
    def _gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Difference in spatial gradients between prediction and target."""
        dy_pred = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
        dx_pred = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
        dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
        dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
        return (dy_pred - dy_target).abs().mean() + (dx_pred - dx_target).abs().mean()
