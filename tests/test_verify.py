"""Verification scores behave as their definitions require."""

import torch

from nowcast.verify import Scores, anisotropy_sums, radial_spectrum


def rain(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.relu(torch.randn(shape, generator=g) * 8 - 4)


def test_perfect_forecast_scores():
    obs = rain((3, 64, 64), 0)
    obs[:, :5] = float("nan")  # no radar coverage
    scores = Scores(leads=3, members=4, device="cpu")
    scores.add(torch.nan_to_num(obs)[None].expand(4, -1, -1, -1), obs)
    s = scores.summary()
    assert max(s["crps_y_1km"]) < 1e-6 and max(s["crps_mmh"]) < 1e-6
    assert all(abs(v - 1) < 1e-6 for v in s["fss_5.0_11km"])
    assert all(abs(v - 1) < 1e-6 for v in s["freq_bias_1.0"])
    assert all(abs(v - 1) < 1e-6 for v in s["bss_1.0"])


def test_calibrated_ensemble_has_flat_rank_histogram_and_matching_spread():
    # Observation and members are exchangeable draws from one distribution.
    draws = rain((9, 4, 128, 128), 1) + 0.01  # strictly wet, no ties at zero
    scores = Scores(leads=4, members=8, device="cpu")
    scores.add(draws[1:], draws[0])
    s = scores.summary()
    hist = torch.tensor(s["rank_histogram"])
    assert float((hist - 1 / 9).abs().max()) < 0.01
    assert all(abs(sp / rm - 1) < 0.03 for sp, rm in zip(s["spread_y"], s["rmse_mean_y"]))


def test_fss_hand_example():
    obs = torch.zeros(1, 8, 8)
    obs[0, 2, 2] = 10.0
    fcst = torch.zeros(1, 1, 8, 8)
    fcst[0, 0, 2, 3] = 10.0  # one pixel displaced
    scores = Scores(leads=1, members=1, device="cpu")
    scores.add(fcst, obs)
    s = scores.summary()
    assert abs(s["fss_5.0_1km"][0]) < 1e-9          # no overlap at grid scale
    assert s["fss_5.0_5km"][0] > 0.7                 # nearly right at 5 km


def test_spectrum_of_single_wave():
    n = 64
    x = torch.arange(n, dtype=torch.float32)
    field = torch.sin(2 * torch.pi * 8 * x / n)[None].expand(n, n)
    spectrum = radial_spectrum(field)
    assert int(spectrum.argmax()) == 8


def test_spectra_accumulate_despite_coverage_gaps():
    obs = rain((2, 1024, 832), 3)
    obs[:, 300:310, 300:400] = float("nan")  # an outage inside the spectrum window
    scores = Scores(leads=2, members=1, device="cpu")
    scores.add(torch.nan_to_num(obs)[None], obs)
    s = scores.summary()
    ratio = torch.tensor(s["spectrum_fcst"][0][1:]) / torch.tensor(s["spectrum_obs"][0][1:])
    assert torch.allclose(ratio, torch.ones_like(ratio))  # perfect forecast: identical masked spectra


def test_spectrum_ignores_the_corners_beyond_nyquist():
    n = 64
    x = torch.arange(n, dtype=torch.float32)
    diagonal = torch.sin(2 * torch.pi * 30 * (x[:, None] + x[None]) / n)  # |k| = 42 > n / 2
    spectrum = radial_spectrum(diagonal + 1e-3 * torch.sin(2 * torch.pi * 8 * x / n)[None])
    assert int(spectrum.argmax()) == 8  # the diagonal wave belongs to no ring (it used to land in the last one)


def test_anisotropy_separates_streaks_from_isotropic_texture():
    torch.manual_seed(0)
    smooth = lambda t, k: torch.nn.functional.avg_pool2d(t[None, None], k, stride=1, padding=k // 2)[0, 0]
    valid = torch.ones(128, 128, dtype=torch.bool)
    coherence = lambda field: float(torch.divide(*anisotropy_sums(field, valid)))
    isotropic = smooth(torch.randn(128, 128), 5)
    streaks = torch.nn.functional.avg_pool2d(torch.randn(1, 1, 128, 168), (1, 41), stride=1)[0, 0]  # smeared along x
    assert coherence(isotropic) < 0.35 < 0.8 < coherence(streaks)


def test_accumulation_score_vanishes_for_a_perfect_forecast():
    obs = rain((24, 64, 64), 5)
    scores = Scores(leads=24, members=3, device="cpu")
    scores.add(obs[None].expand(3, -1, -1, -1), obs)
    s = scores.summary()
    assert s["crps_acc60_y_1km"] < 1e-6 and s["crps_acc120_y_16km"] < 1e-6
    assert all(abs(f - o) < 1e-6 for f, o in zip(s["anisotropy_fcst"], s["anisotropy_obs"]))
