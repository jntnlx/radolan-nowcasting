"""
Baselines: persistence and Farneback optical-flow advection.

Typically used reference approaches.

- Persistence: Future predictions equal most recent observation. 
  Corresponds to simplest possible forecast reference.

- Farneback optical-flow advection: Motion field estimation from 
  last two observations extrapolated forward. Standard nowcasting 
 approach (semi-Lagrangian advection).
"""

import numpy as np
import torch

from .config import SEQ_LEN_IN, SEQ_LEN_OUT
from .transforms import log_to_rate_mmh, normalize_log, rate_mmh_to_log

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def persistence_nowcast(inputs: torch.Tensor) -> torch.Tensor:
    """Repeat last observed frame across all lead times.

    Args:
        inputs: (batch, SEQ_LEN_IN, H, W) normalized-log input tensor.

    Returns:
        (batch, SEQ_LEN_OUT, H, W) persistence forecast.
    """
    last_frame = inputs[:, -1:, :, :]
    return last_frame.expand(-1, SEQ_LEN_OUT, -1, -1)


def optical_flow_nowcast(
    inputs: torch.Tensor,
    *,
    mean: float,
    std: float,
) -> torch.Tensor | None:
    """Farneback optical-flow advection forecast.

    Estimates motion from last two observed frames, then generates
    the most recent frame forward for each lead time using cv2.remap.

    Note: Returns None if OpenCV is not available.
    """
    if not CV2_AVAILABLE:
        return None

    batch = inputs.shape[0]
    h, w = inputs.shape[2], inputs.shape[3]
    results = []

    for b in range(batch):
        # Extract last two frames in physical mm/h space
        frame_prev = inputs[b, SEQ_LEN_IN - 2].cpu().numpy()
        frame_last = inputs[b, SEQ_LEN_IN - 1].cpu().numpy()

        prev_mmh = log_to_rate_mmh(frame_prev * std + mean).astype(np.float32)
        last_mmh = log_to_rate_mmh(frame_last * std + mean).astype(np.float32)

        # Compute optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            prev_mmh, last_mmh,
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )

        # Advect forward for each lead time
        leads = []
        base_y, base_x = np.mgrid[0:h, 0:w].astype(np.float32)
        for lead in range(1, SEQ_LEN_OUT + 1):
            map_x = base_x + flow[:, :, 0] * lead
            map_y = base_y + flow[:, :, 1] * lead
            warped = cv2.remap(
                last_mmh, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            warped = np.clip(warped, 0.0, None)
            # Convert back to normalized-log space
            log_warped = rate_mmh_to_log(warped)
            norm_warped = normalize_log(log_warped, mean=mean, std=std)
            leads.append(norm_warped)

        results.append(np.stack(leads))

    return torch.from_numpy(np.stack(results)).to(inputs.device).float()
