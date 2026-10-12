# Stochastic Lagrangian precipitation nowcasting over Germany

Probabilistic 0–2 h nowcasts of rain rate over the full German radar domain
(1 km, 5 min), learned from 15 warm seasons of the DWD radar climatology
RADKLIM-YW and verified against STEPS, the operational ensemble nowcast.

The model is built around the physics of the problem instead of around a
generic video-prediction network:

* **Advection is resampling, not something convolutions must learn.** Along
  trajectories of the steering flow, the continuity equation for log rain rate
  reads Dy/Dt = s. The network predicts a smooth velocity field and a
  *Lagrangian* source term (growth, decay, initiation), and the forecast is
  obtained by a differentiable semi-Lagrangian scheme.
* **Uncertainty is part of the forecast.** Beyond ~30 min the location and
  intensity of convective cells are not predictable at 1 km. A forecast that
  minimizes squared error converges to a smooth conditional mean and
  systematically underpredicts heavy rain. Here the network outputs an
  ensemble and is trained with a strictly proper scoring rule (CRPS), applied
  at several spatial scales so that the spatial structure of the uncertainty is
  rewarded too.

> Status: v2 rewrite in progress. Results below are filled in as the
> evaluation completes.

## Method

### Data

RADKLIM-YW 2017.002 (DWD): the climate-reprocessed national radar composite,
5-minute precipitation depth on a 1 km polar-stereographic grid. All 19 warm
seasons (April–October, 2006–2024, 1.17 million frames) are re-encoded
losslessly (integer depths in 0.01 mm) into a 36 GB zarr store on a
1024 × 832 km window that contains every pixel ever covered by radar.
Pixels without radar coverage (outages leave between 1 and 17 radars) are
masked in inputs and losses; no sample is discarded.

| split | seasons | use |
|---|---|---|
| train | 2006–2020 | fitting |
| validation | 2021–2022 | model selection; includes the July 2021 floods |
| test | 2023–2024 | locked until the method is frozen; includes the June 2024 floods |

### Model

For lead time k (5 to 120 min) and every pixel x,

    y_k(x) = ψ_k(X_k(x)),     ψ_k = y_0 + S_k,     y = log(1 + R / (1 mm/h))

* X_k(x) are departure points: where the precipitation arriving at x after k
  steps was at the analysis time. They are integrated backwards with the
  midpoint rule in a velocity field u predicted on a 16 km grid, and the
  analysis-time field is interpolated once per lead
  ([advection.py](nowcast/advection.py)).
* S_k is the change of log rain rate that a parcel undergoes by lead k,
  predicted where the precipitation is at the analysis time, i.e. where the
  network can see it.
* u and S_k come from one U-Net ([model.py](nowcast/model.py)). The encoder sees
  the last 30 min of radar data, the coverage mask, and the time of day and
  year (convection has a strong diurnal cycle). The decoder also receives
  Gaussian noise at 32, 16, 8 and 4 km. Every noise draw is one ensemble
  member. Only system-scale noise reaches the velocity grid, so motion
  uncertainty is coherent; finer noise acts on growth and decay.
* With zero-initialized heads the untrained network is exactly persistence.

### Training objective

([losses.py](nowcast/losses.py))

* **Multiscale almost-fair CRPS.** The fair CRPS of an M-member ensemble is an
  unbiased estimate of the CRPS of the distribution the members are drawn
  from, so two members per sample suffice. It is applied to the rain rates
  averaged over 1, 4, 16 and 64 km: each term is proper for its pooled
  quantity, so an ensemble of spatially incoherent noise is penalized at
  the coarse scales. In y, the CRPS equals the Brier score integrated over
  all rain-rate thresholds r with weight 1/(1 + r): every decade of
  intensity above 1 mm/h counts equally.
* **Motion consistency.** The velocity must also explain the observed past:
  frame t0 − j, sampled at the departure points X_j, should reproduce frame t0.
  This is the brightness-constancy objective of variational optical flow,
  evaluated with the forecast's own advection operator, plus a Horn–Schunck
  smoothness prior for regions where no rain constrains the motion.
* **Sampling.** Crops are drawn preferentially where it rains at the analysis
  time. Because the selection depends on the inputs only, the conditional
  distribution of the future is unchanged, and the proper score's optimum
  remains the true conditional distribution: no reweighting is needed.

### Verification

([verify.py](nowcast/verify.py), [evaluate.py](nowcast/evaluate.py))

All methods are run on the full 1024 × 832 km domain for the same 120
analysis times (full hours in 2021–2022 with a domain-mean rain rate of at
least 0.1 mm/h) and scored on identical pixels: fair CRPS at 1–64 km, Brier
skill and reliability, fractions skill score of the members, frequency bias,
spread–skill, rank histograms and power spectra, per lead time.

## Results

*To be added.*

## Reproduce

```bash
conda env create -f environment.yml && conda activate radar-ml
python scripts/build_store.py                         # download + build store (~2 h, 36 GB)
python -m pytest tests                                # scientific contracts (~5 s)
python -m nowcast.train --run my_run --hours 8 --compile
python -m nowcast.evaluate --method steps --cases 120 # classical baselines (CPU)
python -m nowcast.evaluate --method runs/my_run/best.pt --cases 120
```

## Code

| module | content |
|---|---|
| [config.py](nowcast/config.py) | domain, time axis, splits |
| [radklim.py](nowcast/radklim.py) | DWD archive download and decoding with strict unit checks |
| [store.py](nowcast/store.py) | lossless zarr frame store |
| [transforms.py](nowcast/transforms.py) | rain rate ↔ model space |
| [advection.py](nowcast/advection.py) | differentiable semi-Lagrangian advection |
| [model.py](nowcast/model.py) | stochastic Lagrangian network |
| [losses.py](nowcast/losses.py) | CRPS family, motion consistency, smoothness |
| [data.py](nowcast/data.py) | sampling of training windows |
| [train.py](nowcast/train.py) | training loop |
| [baselines.py](nowcast/baselines.py) | persistence, extrapolation, STEPS (pysteps) |
| [verify.py](nowcast/verify.py) | verification scores |
| [evaluate.py](nowcast/evaluate.py) | full-domain evaluation |
| [plots.py](nowcast/plots.py) | maps and score curves |
