"""
Figures: case studies on the radar domain and lead-time score curves.

Maps are drawn in the native RADOLAN polar-stereographic projection
(spherical earth, R = 6370.04 km, true scale at 60 N, central meridian 10 E),
so pixels are shown exactly as stored.
"""

import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from .config import DOMAIN_ORIGIN, DOMAIN_SHAPE  # noqa: E402

RADOLAN_CRS = ccrs.Stereographic(central_latitude=90, central_longitude=10, true_scale_latitude=60,
                                 globe=ccrs.Globe(semimajor_axis=6370040, semiminor_axis=6370040))
GRID_ORIGIN_KM = (-443.462, -4758.645)  # RADOLAN 1100 x 900 grid, x and y of pixel (0, 0)
EXTENT = [1e3 * (GRID_ORIGIN_KM[0] + DOMAIN_ORIGIN[1]), 1e3 * (GRID_ORIGIN_KM[0] + DOMAIN_ORIGIN[1] + DOMAIN_SHAPE[1]),
          1e3 * (GRID_ORIGIN_KM[1] + DOMAIN_ORIGIN[0]), 1e3 * (GRID_ORIGIN_KM[1] + DOMAIN_ORIGIN[0] + DOMAIN_SHAPE[0])]

RAIN_LEVELS = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]  # mm/h
RAIN_CMAP = ListedColormap(["#c6dbef", "#6baed6", "#2171b5", "#08306b", "#41ab5d", "#fed976", "#fd8d3c", "#e31a1c", "#800026"])
RAIN_CMAP.set_under("white")
RAIN_CMAP.set_bad("#e5e5e5")  # no radar data
RAIN_NORM = BoundaryNorm(RAIN_LEVELS + [1000], RAIN_CMAP.N)

METHOD_STYLE = {
    "persistence": dict(color="#999999", ls=":"),
    "extrapolation": dict(color="#555555", ls="--"),
    "steps": dict(color="#d95f02", ls="-"),
}


def _map(ax, title=None):
    ax.set_extent(EXTENT, crs=RADOLAN_CRS)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.4, edgecolor="#333333")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.4, edgecolor="#333333")
    if title:
        ax.set_title(title, fontsize=8)


def show_rain(ax, rate, title=None, window=None):
    """rate: (H, W) mm/h on the model domain (NaN = no data)."""
    ax.imshow(np.ma.masked_invalid(rate), origin="lower", extent=EXTENT, transform=RADOLAN_CRS,
              cmap=RAIN_CMAP, norm=RAIN_NORM, interpolation="nearest")
    _map(ax, title)


PROB_CMAP = matplotlib.colormaps["Purples"].copy()
PROB_CMAP.set_under("white")


def show_probability(ax, probability, observed_event, title=None):
    """Exceedance probability (H, W) with the observed exceedance outlined."""
    ax.imshow(np.ma.masked_less(probability, 0.05), origin="lower", extent=EXTENT, transform=RADOLAN_CRS,
              cmap=PROB_CMAP, vmin=0.05, vmax=1, interpolation="nearest")
    ax.contour(observed_event.astype(float), levels=[0.5], origin="lower", extent=EXTENT, transform=RADOLAN_CRS,
               colors="#d95f02", linewidths=0.5)
    _map(ax, title)


def case_figure(time: str, history: np.ndarray, obs: np.ndarray, forecasts: dict[str, np.ndarray],
                leads=(6, 12, 24), velocity: np.ndarray | None = None, path: Path | None = None,
                probability_threshold: float | None = 5.0):
    """
    Rows: observation; two members of each forecast method; for ensembles also the
    probability of exceeding `probability_threshold` (observed exceedance outlined).
    Columns: lead times.
    history, obs: (T, H, W) mm/h;  forecasts: name -> (M, K, H, W) mm/h;  velocity: (2, h, w) px per step.
    """
    rows = [("observed", lambda ax, k, title: show_rain(ax, obs[k], title))]
    for name, f in forecasts.items():
        for m in range(min(2, f.shape[0])):
            rows.append((f"{name}" + (f" #{m + 1}" if f.shape[0] > 1 else ""),
                         lambda ax, k, title, f=f, m=m: show_rain(ax, f[m, k], title)))
        if f.shape[0] > 1 and probability_threshold is not None:
            rows.append((f"{name}\nP(R > {probability_threshold:g} mm/h)",
                         lambda ax, k, title, f=f: show_probability(ax, (f[:, k] > probability_threshold).mean(0),
                                                                    np.nan_to_num(obs[k]) > probability_threshold, title)))
    cols = len(leads) + 1
    fig, axes = plt.subplots(len(rows), cols, figsize=(2.2 * cols, 2.6 * len(rows)),
                             subplot_kw={"projection": RADOLAN_CRS}, squeeze=False)
    for r, (label, get) in enumerate(rows):  # get(ax, lead index, title) draws one panel
        if r == 0:
            show_rain(axes[0, 0], history[-1], f"analysis {time} UTC")
            if velocity is not None:
                h, w = velocity.shape[-2:]
                km = DOMAIN_SHAPE[1] / w  # grid spacing of the velocity field
                step = max(1, round(32 / km))
                ys, xs = np.mgrid[0:h:step, 0:w:step]
                kx = EXTENT[0] + (xs + 0.5) * km * 1e3
                ky = EXTENT[2] + (ys + 0.5) * km * 1e3
                axes[0, 0].quiver(kx, ky, velocity[0, ::step, ::step], velocity[1, ::step, ::step],
                                  transform=RADOLAN_CRS, scale=60, width=0.004, color="k")
        else:
            axes[r, 0].axis("off")
            axes[r, 0].text(0.5, 0.5, label, ha="center", va="center", transform=axes[r, 0].transAxes, fontsize=10)
        for c, k in enumerate(leads, start=1):
            get(axes[r, c], k - 1, f"+{5 * k} min" if r == 0 else None)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=RAIN_NORM, cmap=RAIN_CMAP), ax=axes, shrink=0.4, pad=0.02)
    cbar.set_label("rain rate (mm/h)")
    if path:
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    return fig


def scores_figure(results: dict[str, dict], path: Path | None = None):
    """Lead-time curves of the main scores; results: name -> summary dict of verify.Scores."""
    panels = [
        ("crps_y_1km", "CRPS of log(1+R), 1 km", False),
        ("crps_y_16km", "CRPS of log(1+R), 16 km mean", False),
        ("crps_mmh", "CRPS (mm/h), 1 km", False),
        ("fss_5.0_21km", "FSS, 5 mm/h, 21 km (members)", True),
        ("bss_5.0", "Brier skill, P(R > 5 mm/h)", True),
        ("freq_bias_5.0", "frequency bias, 5 mm/h", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (key, title, _) in zip(axes.flat, panels):
        for name, summary in results.items():
            values = summary.get(key)
            if values is None:
                continue
            leads = 5 * np.arange(1, len(values) + 1)
            ax.plot(leads, values, label=name, lw=1.6, **METHOD_STYLE.get(name, {}))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("lead time (min)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    return fig


def load_results(directory: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text())["summary"] for p in sorted(Path(directory).glob("*.json"))}


def calibration_figure(results: dict[str, dict], lead: int = 12, threshold: float = 5.0, path: Path | None = None):
    """
    Calibration and sharpness at one lead time (index `lead` 1-based, default +60 min):
    reliability of P(R > threshold), rank histogram, spread vs. error over leads,
    and the power spectrum of member 0 relative to the observations.
    """
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    k = lead - 1
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="k", lw=0.8)
    for name, s in results.items():
        rel = s.get(f"reliability_{threshold}")
        if rel is None or s["members"] == 1:
            continue
        freq = np.array(rel["observed_frequency"][k], dtype=float)
        n = np.array(rel["pixels"][k], dtype=float)
        prob = np.arange(len(freq)) / (len(freq) - 1)
        ok = n > 1000
        ax.plot(prob[ok], freq[ok], marker="o", ms=3, label=name, **METHOD_STYLE.get(name, {}))
    ax.set(title=f"reliability, P(R > {threshold:g} mm/h), +{5 * lead} min", xlabel="forecast probability", ylabel="observed frequency")

    ax = axes[1]
    for name, s in results.items():
        if "rank_histogram" in s:
            hist = np.array(s["rank_histogram"][k])
            ax.plot(np.linspace(0, 1, len(hist)), hist * len(hist), label=name, **METHOD_STYLE.get(name, {}))
    ax.axhline(1, color="k", lw=0.8)
    ax.set(title=f"rank histogram, +{5 * lead} min (flat = calibrated)", xlabel="normalized rank", ylabel="relative frequency")

    ax = axes[2]
    for name, s in results.items():
        if "spread_y" in s:
            ratio = np.array(s["spread_y"]) / np.array(s["rmse_mean_y"])
            ax.plot(5 * np.arange(1, len(ratio) + 1), ratio, label=name, **METHOD_STYLE.get(name, {}))
    ax.axhline(1, color="k", lw=0.8)
    ax.set(title="spread / error of the ensemble mean (1 = calibrated)", xlabel="lead time (min)")

    ax = axes[3]
    for name, s in results.items():
        obs, fcst = np.array(s["spectrum_obs"][k]), np.array(s["spectrum_fcst"][k])
        if obs.sum() == 0:
            continue
        wavelength = 512 / np.arange(1, len(obs))  # km, on the 512 km spectrum window
        ax.loglog(wavelength, fcst[1:] / obs[1:], label=name, **METHOD_STYLE.get(name, {}))
    ax.axhline(1, color="k", lw=0.8)
    ax.invert_xaxis()
    ax.set(title=f"power spectrum / observed, +{5 * lead} min", xlabel="wavelength (km)")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    return fig
