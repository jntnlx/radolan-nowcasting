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

import math

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


def deformation(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Local deformation of a trajectory map X(x), points (B, 2, H, W).

    Returns (anisotropy, determinant), each (B, H - 1, W - 1): the ratio s1 / s2 of the
    singular values of the Jacobian dX/dx and its determinant. A rigid translation
    or rotation has anisotropy 1; a map with anisotropy a draws an isotropic
    feature of the analysis field as a streak with aspect ratio a. A determinant
    <= 0 means that trajectories cross (the map has folded).

    Why this matters: a stationary velocity field with strain rate e stretches
    material lines like exp(e * t). A strain that is harmless for the fit of
    30 minutes of past data shears every advected feature into filaments
    within two hours, as in the stirring of a passive scalar. Precipitation is
    not one (cells are regenerated, patterns stay cellular), so the motion field
    used for extrapolation has to be nearly rigid locally.
    """
    a = points[:, 0, :-1, 1:] - points[:, 0, :-1, :-1]
    b = points[:, 0, 1:, :-1] - points[:, 0, :-1, :-1]
    c = points[:, 1, :-1, 1:] - points[:, 1, :-1, :-1]
    d = points[:, 1, 1:, :-1] - points[:, 1, :-1, :-1]
    q, det = a * a + b * b + c * c + d * d, a * d - b * c
    disc = (q * q - 4 * det * det).clamp(min=0).sqrt()
    return ((q + disc) / 2).sqrt() / ((q - disc) / 2).clamp(min=1e-12).sqrt(), det


def shear(points: torch.Tensor) -> torch.Tensor:
    """
    Squared difference of the singular values of the Jacobian of a trajectory map, (s1 - s2)^2, (B, H - 1, W - 1).

    For J = [[a, b], [c, d]] this is |J|^2 - 2 det J = (a - d)^2 + (b + c)^2: it vanishes exactly
    for rotations with uniform scaling (the Cauchy-Riemann conditions), i.e. for maps that
    keep isotropic texture isotropic. Unlike the anisotropy ratio of `deformation` it is a
    polynomial in the map, with finite gradients everywhere, so it can serve as a penalty.
    """
    a = points[:, 0, :-1, 1:] - points[:, 0, :-1, :-1]
    b = points[:, 0, 1:, :-1] - points[:, 0, :-1, :-1]
    c = points[:, 1, :-1, 1:] - points[:, 1, :-1, :-1]
    d = points[:, 1, 1:, :-1] - points[:, 1, :-1, :-1]
    return (a - d) ** 2 + (b + c) ** 2


@torch.compiler.disable  # an inner optimization loop: runs eagerly inside compiled models
def estimate_velocity(
    history: torch.Tensor,
    valid: torch.Tensor,
    grid: int = 16,
    pool: int = 4,
    iterations: int = 50,
    lr: float = 0.3,
    smoothness: float = 0.1,
    uniform_iterations: int = 30,
    lr_end: float = 0.01,
) -> torch.Tensor:
    """
    Variational optical flow: the stationary velocity that best explains the observed past.

    Minimizes, for each sample independently,

        mean_j | y_{-j}(X_j) - y_0 |  +  smoothness * |grad u|^2

    (brightness constancy along the departure points X_j of the same
    semi-Lagrangian scheme, plus a Horn-Schunck smoothness term) over a
    velocity field on a `grid`-km grid, on frames averaged to `pool` km.
    Comparing area-averaged frames is the coarse-scale stage of classical
    pyramid optical flow: it gives the long-range gradients needed for
    displacements of tens of kilometres.

    Coarse to fine: one uniform velocity per sample is fitted first
    (`uniform_iterations` Adam steps), then the field (`iterations` steps);
    the step size decays as a cosine from `lr` to `lr_end`, so the solver
    converges instead of jittering around the optimum.

    The smoothness weight decides what the field is good for. Growth and
    decay are not motion, but a weakly regularized fit explains them as
    convergence and divergence. With 0.01 the strain of the fitted field
    deformed the 2 h trajectory map severely (in rain: median stretch
    anisotropy 2.4-2.8, 90th percentile 8-10, 1-2% of pixels folded), which
    drew every advected feature as a streak. With 0.1 the map stays nearly
    rigid (1.45, 2.3, 0.02%) at the same skill of pure advection (mean
    absolute error at +15 to +120 min within 0.5%, on 30 full-domain cases of
    the training seasons 2019-20; 0.03 to 0.15 are equivalent in skill and
    differ only in deformation). See `deformation`.

    Legacy behaviour (checkpoints trained before this change): smoothness 0.01,
    uniform_iterations 0, lr_end = lr.

    history: (B, T, H, W) in y, oldest first;  valid: (B, 1, H, W).
    Returns velocity (B, 2, H / grid, W / grid) in px per step, without gradient.
    """
    with torch.enable_grad(), torch.autocast(history.device.type, enabled=False):
        x = F.avg_pool2d(history.detach().float(), pool)
        mask = (F.avg_pool2d(valid.float(), pool) > 0.5).float()[:, 0]
        b, t = x.shape[:2]
        n = mask.sum((1, 2)).clamp(min=1)
        shape = (b, 2, history.shape[-2] // grid, history.shape[-1] // grid)

        def mismatch(velocity):
            points = departure_points(velocity / pool, t - 1, upsample=grid // pool)
            return sum(((sample(x[:, t - 1 - j:t - j], points[j - 1])[:, 0] - x[:, -1]).abs() * mask).sum((1, 2)) / n
                       for j in range(1, t)) / (t - 1)

        def minimize(parameter, loss_fn, steps, beta1=0.9, beta2=0.999, eps=1e-8):
            # Adam, written out: torch.optim touches CUDA state on every step, which fails in the
            # forked data-loader workers that compute the first guess during training (data.py).
            m, v = torch.zeros_like(parameter), torch.zeros_like(parameter)
            for i in range(steps):
                step = lr_end + (lr - lr_end) * 0.5 * (1 + math.cos(math.pi * i / steps))
                gradient, = torch.autograd.grad(loss_fn(parameter).sum(), parameter)  # samples are independent
                m, v = beta1 * m + (1 - beta1) * gradient, beta2 * v + (1 - beta2) * gradient**2
                update = (m / (1 - beta1 ** (i + 1))) / ((v / (1 - beta2 ** (i + 1))).sqrt() + eps)
                parameter = (parameter - step * update).detach().requires_grad_(True)
            return parameter.detach()

        mean = minimize(torch.zeros(b, 2, 1, 1, device=x.device, requires_grad=True), lambda m: mismatch(m.expand(shape)),
                        uniform_iterations)
        velocity = mean.expand(shape).clone().requires_grad_(True)

        def objective(v):
            rough = (v[..., 1:] - v[..., :-1]).pow(2).mean((1, 2, 3)) + (v[..., 1:, :] - v[..., :-1, :]).pow(2).mean((1, 2, 3))
            return mismatch(v) + smoothness * rough

        return minimize(velocity, objective, iterations)
