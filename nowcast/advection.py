"""
Differentiable semi-Lagrangian advection.

Precipitation is, to first order, carried by the mid-tropospheric flow:

    dy/dt + u . grad(y) = s          (y: log rain rate, s: growth/decay)

Along a trajectory x(t) with dx/dt = u, this reads Dy/Dt = s. A semi-Lagrangian
forecast for lead k therefore needs the departure point X_k(x): where the air
parcel arriving at x after k steps was at the analysis time. The forecast is
the analysis-time field, evaluated at X_k(x) by interpolation.

Departure points are integrated backwards with the explicit midpoint rule

    X_{k+1} = X_k - u(X_k - u(X_k) / 2)

and each lead interpolates the analysis field exactly once, so interpolation
diffusion does not accumulate over lead time (as in the pysteps extrapolation
scheme). Velocities are in pixels per step: 1 px / 5 min = 12 km/h = 3.3 m/s.
"""

import torch
import torch.nn.functional as F


def grid(batch: int, height: int, width: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Pixel coordinates (batch, 2, H, W); channel 0 = x (column), 1 = y (row)."""
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([x, y]).expand(batch, 2, height, width)


def sample(field: torch.Tensor, position: torch.Tensor, padding: str = "border") -> torch.Tensor:
    """Bilinearly interpolate field (B, C, H, W) at pixel positions (B, 2, H', W')."""
    height, width = field.shape[-2:]
    normalized = torch.stack([
        2 * position[:, 0] / (width - 1) - 1,
        2 * position[:, 1] / (height - 1) - 1,
    ], dim=-1)
    return F.grid_sample(field, normalized.to(field.dtype), mode="bilinear", padding_mode=padding, align_corners=True)


def departure_points(velocity: torch.Tensor, steps: int, upsample: int = 1) -> list[torch.Tensor]:
    """
    Backward trajectories under a stationary velocity field (B, 2, h, w), in pixels per step.

    Trajectories are integrated on the grid of the velocity field. With
    `upsample` > 1 that grid is coarser than the forecast grid by this factor:
    the (smooth) displacement field is then interpolated to the fine grid,
    which costs upsample^2 fewer interpolations than integrating there.
    Velocities are always in fine-grid pixels per step.

    Returns [X_1, ..., X_steps], each (B, 2, h * upsample, w * upsample) in fine-grid pixels.
    """
    b, _, h, w = velocity.shape
    origin = grid(b, h, w, velocity.device, velocity.dtype)
    velocity = velocity / upsample
    position = origin
    points = []
    for _ in range(steps):
        half = position - 0.5 * sample(velocity, position)
        position = position - sample(velocity, half)
        displacement = upsample * (position - origin)
        if upsample > 1:
            displacement = F.interpolate(displacement, scale_factor=upsample, mode="bilinear", align_corners=False)
        points.append(grid(b, h * upsample, w * upsample, velocity.device, velocity.dtype) + displacement)
    return points


def inside(position: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """True where a departure point lies within the field (B, H, W)."""
    x, y = position[:, 0], position[:, 1]
    return (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)


@torch.compiler.disable  # an inner optimization loop: runs eagerly inside compiled models
def estimate_velocity(
    history: torch.Tensor,
    valid: torch.Tensor,
    grid: int = 16,
    pool: int = 4,
    iterations: int = 50,
    lr: float = 0.3,
    smoothness: float = 0.01,
) -> torch.Tensor:
    """
    Variational optical flow: the stationary velocity that best explains the observed past.

    Minimizes, for each sample independently,

        mean_j | y_{-j}(X_j) - y_0 |  +  smoothness * |grad u|^2

    (brightness constancy along the departure points X_j of the same
    semi-Lagrangian scheme, plus a Horn-Schunck smoothness term) over a
    velocity field on a `grid`-km grid, from rest, by `iterations` Adam steps
    on frames averaged to `pool` km. Comparing area-averaged frames is the
    coarse-scale stage of classical pyramid optical flow: it gives the
    long-range gradients needed for displacements of tens of kilometres.

    history: (B, T, H, W) in y, oldest first;  valid: (B, 1, H, W).
    Returns velocity (B, 2, H / grid, W / grid) in px per step, without gradient.
    """
    with torch.enable_grad(), torch.autocast(history.device.type, enabled=False):
        x = F.avg_pool2d(history.detach().float(), pool)
        mask = (F.avg_pool2d(valid.float(), pool) > 0.5).float()[:, 0]
        b, t = x.shape[:2]
        n = mask.sum((1, 2)).clamp(min=1)
        velocity = torch.zeros(b, 2, history.shape[-2] // grid, history.shape[-1] // grid, device=x.device, requires_grad=True)
        optimizer = torch.optim.Adam([velocity], lr=lr)
        for _ in range(iterations):
            points = departure_points(velocity / pool, t - 1, upsample=grid // pool)
            mismatch = sum(((sample(x[:, t - 1 - j:t - j], points[j - 1])[:, 0] - x[:, -1]).abs() * mask).sum((1, 2)) / n
                           for j in range(1, t)) / (t - 1)
            rough = (velocity[..., 1:] - velocity[..., :-1]).pow(2).mean((1, 2, 3)) + \
                    (velocity[..., 1:, :] - velocity[..., :-1, :]).pow(2).mean((1, 2, 3))
            loss = (mismatch + smoothness * rough).sum()  # samples are independent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return velocity.detach()
