"""
Stochastic Lagrangian nowcasting network.

The forecast follows the structure of the continuity equation for log rain
rate y along trajectories, Dy/Dt = s:

    y_k(x) = psi_k(X_k(x)),    psi_k = y_0 + S_k

  * a velocity field u, from which departure points X_k are integrated
    (advection.py): where the precipitation arriving at x at lead k was at
    the analysis time t0;
  * a Lagrangian source S_k: the change in log rain rate that the parcel
    starting at a point will undergo by lead k (growth, decay, initiation),
    predicted *where the precipitation is at t0*, i.e. where the network can
    actually see it.

Advection by tens of kilometres is thereby a resampling operation and not
something convolutions must learn. The network only predicts smooth motion
and local evolution.

Motion: a U-Net cannot learn to infer displacements of tens of kilometres
from stacked frames (a network trained so learned no usable motion at all).
The velocity is therefore u = u_0 + du: u_0 is a variational optical flow
fitted to the observed past inside the forward pass (advection.py,
`estimate_velocity`; on the full domain as skilful as pysteps' Lucas-Kanade
extrapolation, at ~0.1 s on the GPU instead of ~13 s on the CPU), and the
network predicts a bounded correction du, trained on the future, together
with the members' motion perturbations.

Both du and S_k are outputs of one U-Net. The encoder sees the radar history,
the coverage mask, and the time of day and year (convection follows a
strong diurnal cycle). The decoder additionally receives Gaussian noise
fields with 32 and 4 km correlation scales, which modulate the
normalization of every decoder block (as in AIFS-CRPS and FGN; noise given
as extra input channels was largely ignored: members 2x under-dispersive).
Each draw of the noise yields one ensemble member: a forecast of motion and
evolution that is coherent over the noise correlation scales. The velocity is predicted on a
16 km grid, so only system-scale noise perturbs the motion while the finer
noise acts on growth and decay. The encoder runs once per sample, the
decoder once per member.

With zero-initialized heads the untrained network is exactly optical-flow
extrapolation (Lagrangian persistence). Training moves it from there.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .advection import departure_points, estimate_velocity, sample


def _groups(channels: int) -> int:
    return math.gcd(channels, 16)


class ResBlock(nn.Module):
    """
    Pre-norm residual block. Its second normalization is modulated (FiLM) by
    a conditioning vector and, in the decoder, by spatial noise fields: every
    block's scale and shift depend on the noise, so the network cannot route
    around it as it can around noise supplied as extra input channels.
    """

    def __init__(self, c_in: int, c_out: int, c_cond: int, c_noise: int = 0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(c_in), c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(c_out), c_out)
        self.film = nn.Linear(c_cond, 2 * c_out)
        self.noise_film = nn.Conv2d(c_noise, 2 * c_out, 1) if c_noise else None
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        if self.noise_film is not None:  # small, not zero: members differ from the first step on
            nn.init.normal_(self.noise_film.weight, std=0.02)
            nn.init.zeros_(self.noise_film.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(cond)[:, :, None, None].chunk(2, dim=1)
        if self.noise_film is not None:
            noise_scale, noise_shift = self.noise_film(noise).chunk(2, dim=1)
            scale, shift = scale + noise_scale, shift + noise_shift
        h = self.norm2(h) * (1 + scale) + shift
        return self.skip(x) + self.conv2(F.silu(h))


class Level(nn.Module):
    def __init__(self, c_in: int, c_out: int, depth: int, c_cond: int, c_noise: int = 0):
        super().__init__()
        self.blocks = nn.ModuleList([ResBlock(c_in if i == 0 else c_out, c_out, c_cond, c_noise) for i in range(depth)])

    def forward(self, x, cond, noise=None):
        for block in self.blocks:
            x = block(x, cond, noise)
        return x


Y_MAX = math.log1p(500.0)  # 500 mm/h: above any rain rate observed in RADKLIM
V_MAX = 10.0                # px per step = 120 km/h: above any precipitation system motion


def clamp_straight_through(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Physical bounds in the forward pass (exact zeros where dry), identity gradient in the backward pass."""
    return x + (x.clamp(low, high) - x).detach()


class Nowcaster(nn.Module):
    """
    inputs:  (B, T, H, W)  log rain y = log(1 + R), oldest first; 0 where no data
    valid:   (B, 1, H, W)  radar coverage over the whole input window
    context: (B, 4)        sin/cos of time of day and of day of year
    returns: dict with forecast (B, M, K, H, W) in y; velocity (B, M, 2, H/16, W/16)
             in px per step on the 16 km grid on which trajectories are
             integrated; the optical-flow first guess (B, 2, H/16, W/16);
             departure points; source fields.
    """

    def __init__(
        self,
        history: int = 6,
        leads: int = 24,
        widths: tuple[int, ...] = (16, 32, 96, 192, 256, 384),
        depths: tuple[int, ...] = (1, 1, 2, 2, 3, 3),
        noise: str = "film",          # "film": noise modulates every decoder block; "concat": extra input channels (lag_v2)
        noise_channels: int = 8,
        noise_levels: tuple[int, ...] = (2, 3, 4, 5),  # film: noise fields at the coarsest and finest of these levels
        c_cond: int = 128,
        velocity_scale: float = 4.0,  # initial sensitivity of the velocity head, px per step
        velocity_level: int = 4,      # velocity on the 2^4 = 16 km grid
        eulerian: bool = False,       # ablation: no advection, all change comes from the source term
        optical_flow: bool = True,    # physical first guess of the motion (False: network-only motion)
    ):
        super().__init__()
        self.history, self.leads = history, leads
        self.noise, self.noise_channels, self.noise_levels = noise, noise_channels, noise_levels
        self.velocity_scale = velocity_scale
        self.stride = 2 ** (len(widths) - 1)

        self.cond = nn.Sequential(nn.Linear(4, c_cond), nn.SiLU(), nn.Linear(c_cond, c_cond))

        # Encoder (deterministic, run once per sample)
        self.stem = nn.Conv2d(history + 3, widths[0], 3, padding=1)  # frames, coverage, first-guess velocity
        self.down = nn.ModuleList([nn.Conv2d(widths[i - 1], widths[i], 3, stride=2, padding=1) for i in range(1, len(widths))])
        self.encoder = nn.ModuleList([Level(w, w, d, c_cond) for w, d in zip(widths, depths)])

        # Decoder (stochastic, run once per member)
        def extra(level):
            return noise_channels if noise == "concat" and level in noise_levels else 0

        c_noise = 2 * noise_channels if noise == "film" else 0
        top = len(widths) - 1
        self.bottleneck = Level(widths[top] + extra(top), widths[top], depths[top], c_cond, c_noise)
        self.up = nn.ModuleList([nn.Conv2d(widths[i + 1], widths[i], 3, padding=1) for i in range(top)])
        self.decoder = nn.ModuleList([Level(2 * widths[i] + extra(i), widths[i], depths[i], c_cond, c_noise)
                                      for i in range(top)])

        # Heads: velocity on a coarse grid, source at full resolution. The steering
        # flow is smooth on the mesoscale, and on this grid only the coarsest noise
        # (16, 32 km) perturbs it: motion uncertainty is a system-scale quantity.
        self.velocity_level = velocity_level
        self.eulerian = eulerian
        self.optical_flow = optical_flow and not eulerian
        self.velocity_head = nn.Conv2d(widths[self.velocity_level], 2, 3, padding=1)
        self.source_head = nn.Conv2d(widths[0], leads, 3, padding=1)
        for head in (self.velocity_head, self.source_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def encode(self, inputs, valid, context, prior):
        cond = self.cond(context)
        prior = F.interpolate(prior, size=inputs.shape[-2:], mode="bilinear", align_corners=False) / V_MAX
        x = self.stem(torch.cat([inputs, valid.to(inputs.dtype), prior.to(inputs.dtype)], dim=1))
        skips = []
        for i, level in enumerate(self.encoder):
            if i > 0:
                x = self.down[i - 1](x)
            x = level(x, cond)
            skips.append(x)
        return skips, cond

    def _noise(self, like: torch.Tensor, level: int, generator) -> list[torch.Tensor]:
        """Noise as extra input channels (noise == "concat")."""
        if self.noise != "concat" or level not in self.noise_levels:
            return []
        shape = (like.shape[0], self.noise_channels, *like.shape[-2:])
        return [torch.randn(shape, device=like.device, dtype=like.dtype, generator=generator)]

    def _noise_fields(self, skips, generator):
        """
        Noise for FiLM modulation (noise == "film"): two Gaussian fields per member,
        drawn on the coarsest and the finest noise level (32 and 4 km by default),
        resampled to every decoder level. Returns one tensor per level, or Nones.
        """
        if self.noise != "film":
            return [None] * len(skips)
        coarse, fine = max(self.noise_levels), min(self.noise_levels)
        fields = [torch.randn((skips[0].shape[0], self.noise_channels, *skips[lvl].shape[-2:]),
                              device=skips[0].device, dtype=skips[0].dtype, generator=generator) for lvl in (coarse, fine)]
        resample = lambda f, size: f if f.shape[-2:] == size else F.interpolate(
            f, size=size, mode="bilinear", align_corners=False, antialias=f.shape[-1] > size[-1])
        return [torch.cat([resample(f, s.shape[-2:]) for f in fields], dim=1) for s in skips]

    def decode(self, skips, cond, prior, generator=None):
        top = len(skips) - 1
        noise = self._noise_fields(skips, generator)
        x = self.bottleneck(torch.cat([skips[top], *self._noise(skips[top], top, generator)], dim=1), cond, noise[top])
        velocity = None
        for i in reversed(range(top)):
            x = self.up[i](F.interpolate(x, scale_factor=2, mode="nearest"))
            x = self.decoder[i](torch.cat([x, skips[i], *self._noise(skips[i], i, generator)], dim=1), cond, noise[i])
            if i == self.velocity_level:
                velocity = self.velocity_head(x)
        correction = V_MAX * torch.tanh(self.velocity_scale * velocity.float() / V_MAX)
        velocity = torch.zeros_like(correction) if self.eulerian else prior + correction
        return velocity, self.source_head(x)

    def first_guess(self, inputs, valid) -> torch.Tensor:
        """Optical-flow first guess of the velocity (zero without optical flow)."""
        b, _, h, w = inputs.shape
        grid = 2 ** self.velocity_level
        if self.optical_flow:
            return estimate_velocity(inputs, valid, grid=grid)
        return inputs.new_zeros(b, 2, h // grid, w // grid, dtype=torch.float32)

    def forward(self, inputs, valid, context, members: int = 2, generator=None, prior=None) -> dict[str, torch.Tensor]:
        """`prior`: a precomputed `first_guess`, to reuse it across calls for more members."""
        b, _, h, w = inputs.shape
        if h % self.stride or w % self.stride:
            raise ValueError(f"spatial size {(h, w)} must be divisible by {self.stride}")
        if prior is None:
            prior = self.first_guess(inputs, valid)
        skips, cond = self.encode(inputs, valid, context, prior)

        # Members share the encoding and the first guess; stack them along the batch axis for one decoder pass.
        skips = [s.repeat_interleave(members, dim=0) for s in skips]
        cond = cond.repeat_interleave(members, dim=0)
        velocity, source = self.decode(skips, cond, prior.repeat_interleave(members, dim=0), generator)
        velocity, source = velocity.float(), source.float()

        latest = inputs[:, -1:].float().repeat_interleave(members, dim=0)
        psi = latest + source  # Lagrangian forecast in analysis-time coordinates
        factor = 2**self.velocity_level
        points = departure_points(velocity, max(self.leads, self.history - 1), upsample=factor)
        # One interpolation per lead, batched over leads: (B*M*K, 1, H, W) sampled at X_k.
        bm = b * members
        stacked = torch.stack(points[:self.leads], dim=1).reshape(bm * self.leads, 2, h, w)
        forecast = sample(psi.reshape(bm * self.leads, 1, h, w), stacked).reshape(bm, self.leads, h, w)
        forecast = clamp_straight_through(forecast, 0.0, Y_MAX)

        unstack = lambda t: t.reshape(b, members, *t.shape[1:])
        return {
            "forecast": unstack(forecast),
            "velocity": unstack(velocity),
            "prior": prior,
            "source": unstack(source),
            "points": points,  # (B * M, 2, H, W) each, member-major within sample
        }
