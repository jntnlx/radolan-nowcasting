# Method

Details behind the summary in the [README](../README.md).

## Data

RADKLIM-YW 2017.002 (DWD): the climate-reprocessed national radar composite,
5-minute precipitation depth on a 1 km polar-stereographic grid. All 19 warm
seasons (April–October, 2006–2024, 1.17 million frames) are re-encoded
losslessly (integer depths in 0.01 mm) into a 36 GB zarr store on a
1024 × 832 km window that contains every pixel ever covered by radar
([store.py](../nowcast/store.py)). Pixels without radar coverage (outages leave
between 1 and 17 radars) are masked in inputs and losses; no sample is
discarded.

| split | seasons | use |
|---|---|---|
| train | 2006–2020 | fitting |
| validation | 2021–2022 | model selection; includes the July 2021 floods |
| test | 2023–2024 | locked until the method is frozen; includes the June 2024 floods |

## Model

For lead time k (5 to 120 min) and every pixel x,

    y_k(x) = ψ_k(X_k(x)),     ψ_k = y_0 + S_k,     y = log(1 + R / (1 mm/h))

* X_k(x) are departure points: where the precipitation arriving at x after k
  steps was at the analysis time. They are integrated backwards with the
  midpoint rule in a velocity field u = u₀ + Δu on a 16 km grid, and the
  analysis-time field is interpolated once per lead
  ([advection.py](../nowcast/advection.py)).
* u₀ is a variational optical flow of the observed past: brightness constancy
  along the scheme's own trajectories plus a Horn–Schunck smoothness term,
  minimized on 4 km means of the last 30 min (first one uniform velocity,
  then the field; Adam with a decaying step). It is a fixed function of the
  inputs: the data-loader workers compute it during training, the model
  itself at inference. Δu is a
  learned correction, bounded by a tanh.
* The motion is kept nearly rigid locally. A stationary field with strain
  rate e stretches material lines like exp(e t), so strain that is harmless
  over the fitted 30 min deforms the 2 h trajectory map and draws isotropic
  features as streaks. A weakly regularized flow has such strain because it
  explains growth and decay as divergence and convergence; in this model
  that is the role of S_k. Two measures: a Horn–Schunck weight of 1.0 for
  u₀ (median stretch anisotropy of the 2 h map in rain 1.1; it was 2.4–2.8
  with the initial 0.01), and a penalty on the shear (s₁ − s₂)² of the
  common trajectory map (without the member perturbations) at the last lead, where s₁, s₂ are the singular values of
  ∂X/∂x. For J = [[a, b], [c, d]] the shear is (a − d)² + (b + c)²: zero
  exactly for rotations with uniform scaling.
* The deformation is given a lifetime instead of being discarded. The
  difference d between a detailed optical flow (Horn–Schunck weight 0.1, the
  best for pure extrapolation) and u₀ is the part of the observed motion
  that fits the last 30 min but does not persist. The trajectories are
  integrated in the time-dependent velocity u(t) = u₀ + Δu + w(t) d (plus the
  member perturbation), with w falling linearly from 1 at the analysis time
  to 0 at +60 min: the detail displaces the rain while it lasts, and that
  displacement stays. A trajectory arriving at lead k is traced back through
  the velocities of steps k, …, 1. This is an option of the forecast operator
  (`--motion-detail 60` at evaluation); the network shown in the README was
  trained without it, and the physics terms of the loss act on the motion
  without d. On the 120 validation cases it lowers the CRPS at 1 km by 6.8 /
  3.3 / 1.7 / 0.6 % at +5 / +30 / +60 / +120 min and at 16 km by 2.5 / 4.5 /
  2.7 / 0.8 %; the coherence at +2 h is 0.36 (observed 0.33; 0.40 with the
  detailed flow at all leads).
* S_k is the change of log rain rate that a parcel undergoes by lead k,
  predicted where the precipitation is at the analysis time, i.e. where the
  network can see it.
* Δu and S_k come from one U-Net ([model.py](../nowcast/model.py)). The encoder
  sees the last 30 min of radar data, the coverage mask, u₀, and the time of
  day and year (convection has a strong diurnal cycle). S_k is predicted on a
  4 km grid and interpolated to 1 km, so that the fine structure of a member
  comes from the displaced observation, not from the network.
* **Ensemble.** Most of the uncertainty of a nowcast is where the rain will be,
  not how strong it is. Each member m therefore moves with its own velocity
  u = u₀ + Δu + a ξ_m: ξ_m is white noise interpolated bicubically from a
  512 km grid (a smooth perturbation of the whole system's motion), and the
  amplitude a = a₀ exp(h) is predicted by a zero-initialized head of the
  encoder, within a factor e³ of a₀ = 1.5 px per step. The perturbation enters
  the trajectories only; motion consistency and the shear penalty see the
  common motion. A Gaussian noise field with 32 km correlation additionally
  modulates every decoder block (FiLM), so that growth and decay differ
  between members. Each noise draw is one ensemble member.
* Why displacement: the CRPS scores the distribution at each pixel and of area
  means, not the arrangement of rain within a member. With noise fields that
  act locally (an earlier version used FiLM noise at 32 and 4 km and no
  velocity perturbation), the network realized its spread as local amplitude
  noise: good scores, mottled members that erased the observed structure.
  Displacing the observation produces members that are radar fields by
  construction at nearly the same scores.
* With zero-initialized output layers and without the velocity perturbation,
  the untrained network is exactly optical-flow extrapolation; with it, the
  untrained network is a displacement ensemble of the analysis.

## Training objective

([losses.py](../nowcast/losses.py))

* **Multiscale almost-fair CRPS.** The fair CRPS of an M-member ensemble
  (Ferro 2014) is an unbiased estimate of the CRPS of the distribution the
  members are drawn from, so two members per sample suffice. With two members,
  though, any pair that brackets the observation scores the same however far
  apart the members are. The almost-fair CRPS (α = 0.95, as in AIFS-CRPS) adds
  a small spread penalty that removes this degeneracy, at the cost of a small
  downward bias of low probabilities; a comparison with α = 1 is planned. The
  score is applied to rain rates averaged over 1, 4, 16 and 64 km: each term is
  proper for its pooled quantity, so spatially incoherent noise is penalized at
  the coarse scales. In y, the CRPS equals the Brier score integrated over all
  rain-rate thresholds r with weight 1/(1 + r).
* **Motion consistency.** The velocity must also explain the observed past
  (brightness constancy along the trajectories), with a Horn–Schunck
  smoothness prior where no rain constrains the motion, and the shear
  penalty described above. Removing the consistency term made training
  unstable and the scores worse.
* **Sampling.** Crops of 512 km are drawn preferentially where it rains at the
  analysis time. Because the selection depends on the inputs only, the
  conditional distribution of the future is unchanged, and the proper score's
  optimum remains the true conditional distribution: no reweighting is needed.
* **Optimization.** Batch 4, two members per sample, AdamW, warm-up and cosine
  decay over a wall-clock budget, EMA of the weights, bf16. About 3.3 steps/s
  on an RTX 4070 Super (12 GB).

## Verification

([verify.py](../nowcast/verify.py), [evaluate.py](../nowcast/evaluate.py))

All methods are run on the full domain for the same 120 analysis times (full
hours with a domain-mean rain rate of at least 0.1 mm/h) and scored on identical
pixels: fair CRPS at 1–64 km, Brier skill and reliability, fractions skill
score of the members, frequency bias, spread–skill, rank histograms,
radially averaged power spectra and the coherence of the gradient structure
tensor (0 = locally isotropic texture, 1 = parallel streaks; to be compared
with the observed value), per lead time; and the fair CRPS of 1 h and 2 h
accumulations, which tests the temporal coherence of the members. Differences between methods
are tested per case, with calendar-day block bootstrap intervals.

Known gaps: missing radar data enter the inputs as zeros, which leaves an
artifact along coverage edges; the reference scores of STEPS for the two newest
diagnostics require a rerun of its evaluation.
