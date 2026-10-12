#!/usr/bin/env python
"""
Summary figures for an evaluation population.

    python scripts/figures.py runs/eval/val_120 --out docs/figures [--motion runs/motion_study.json]

1_crps_by_lead      CRPS at 1 and 16 km per lead time, mean over the cases common to all methods
2_paired_vs_steps   per-case CRPS difference to STEPS with 95% day-block bootstrap intervals
3_heavy_rain        Brier skill and frequency bias for R > 5 mm/h (complete evaluations only)
4_calibration       spread/error, rank histogram and reliability of the ensembles
5_spectra           power spectrum of forecasts relative to observations
6_motion            advection error of different motion fields (scripts/motion_study.py)

Evaluations still in progress (<method>.partial.pt) enter figures 1 and 2,
which are computed from per-case scores on common cases.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from report import paired_difference  # noqa: E402  (scripts/report.py)

# One fixed colour and line style per method in every figure (colour-blind checked;
# low-contrast hues are always paired with a line style and a label).
STYLE = {
    "persistence": dict(color="#898781", ls=":", label="persistence"),
    "extrapolation": dict(color="#1baf7a", ls="--", label="optical-flow extrapolation (pysteps)"),
    "flow": dict(color="#eda100", ls="-.", label="variational flow extrapolation"),
    "steps": dict(color="#eb6834", ls="-", label="STEPS, 20 members"),
    "lag_v2": dict(color="#2a78d6", ls="-", label="network lag_v2, 20 members"),
    "lag_v3_film": dict(color="#6250d6", ls="-", label="network lag_v3 (FiLM noise), 20 members"),
}
INK, INK_2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "sans-serif",
})


def style(ax, xlabel="lead time (min)", ylabel=None, title=None):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")


def line(ax, x, y, name, **kw):
    s = STYLE.get(name, dict(color=INK_2, ls="-", label=name))
    return ax.plot(x, y, color=s["color"], ls=s["ls"], lw=1.8, label=s["label"], **kw)


def end_labels(ax, ends: dict[str, float], x: float, min_gap: float = 0.045):
    """Direct labels at the right end of lines, in text ink, spread apart where lines end close together."""
    lo, hi = ax.get_ylim()
    items = sorted(ends.items(), key=lambda kv: kv[1])
    placed = []
    for name, y in items:
        frac = (y - lo) / (hi - lo)
        if placed and frac - placed[-1] < min_gap:
            frac = placed[-1] + min_gap
        placed.append(frac)
        ax.annotate(STYLE.get(name, {}).get("label", name).split(",")[0].split(" (")[0], (x, y),
                    xytext=(x + 2, lo + frac * (hi - lo)), textcoords="data", va="center", fontsize=7.5, color=INK_2)


def load(directory: Path):
    complete = {f.stem: json.loads(f.read_text()) for f in sorted(directory.glob("*.json"))}
    partial = {f.name.removesuffix(".partial.pt"): {"cases": torch.load(f, map_location="cpu", weights_only=False)["cases"]}
               for f in sorted(directory.glob("*.partial.pt"))}
    partial = {k: v for k, v in partial.items() if k not in complete}
    return complete, partial


def order(names):
    rank = {k: i for i, k in enumerate(STYLE)}
    return sorted(names, key=lambda n: rank.get(n, len(rank)))


def crps_by_lead(results, out):
    keys = [set((c["year"], c["t0"]) for c in r["cases"]) for r in results.values()]
    common = set.intersection(*keys)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in zip(axes, ("crps_y_1km", "crps_y_16km"),
                              ("CRPS of log(1 + R) at 1 km", "CRPS of log(1 + R), 16 km area means")):
        ends = {}
        for name in order(results):
            if name == "flow":
                continue  # indistinguishable from the pysteps extrapolation
            cases = [c for c in results[name]["cases"] if (c["year"], c["t0"]) in common]
            curve = np.mean([c[key] for c in cases], axis=0)
            leads = 5 * np.arange(1, len(curve) + 1)
            line(ax, leads, curve, name)
            ends[name] = curve[-1]
        style(ax, ylabel="CRPS (lower is better)", title=title)
        ax.set_xlim(0, 155)
        end_labels(ax, ends, 120)
    axes[0].legend(loc="upper left")
    fig.suptitle(f"Full-domain validation, {len(common)} analysis times common to all methods (2021-2022)",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    fig.savefig(out / "1_crps_by_lead.png", dpi=130)
    plt.close(fig)


def paired_vs_steps(results, out):
    if "steps" not in results:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, key, title in zip(axes, ("crps_y_1km", "crps_y_16km"), ("1 km", "16 km area means")):
        ax.axhline(0, color=INK_2, lw=0.8)
        ends = {}
        for name in order(results):
            if name in ("steps", "persistence", "flow"):
                continue
            stats = [paired_difference(results[name], results["steps"], key, k) for k in range(24)]
            mean = np.array([s[0] for s in stats])
            lo, hi = np.array([s[1][0] for s in stats]), np.array([s[1][1] for s in stats])
            leads = 5 * np.arange(1, 25)
            line(ax, leads, mean, name)
            ax.fill_between(leads, lo, hi, color=STYLE[name]["color"], alpha=0.15, lw=0)
            ends[name] = mean[-1]
        style(ax, ylabel="CRPS minus CRPS of STEPS", title=f"Paired difference to STEPS, {title}")
        ax.set_xlim(0, 155)
        end_labels(ax, ends, 120)
    n = stats[0][2]
    axes[0].legend(loc="upper left")
    fig.suptitle(f"Below zero: better than STEPS. Mean over {n} common cases; shaded: 95% interval (bootstrap over calendar days)",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    fig.savefig(out / "2_paired_vs_steps.png", dpi=130)
    plt.close(fig)


def heavy_rain(complete, out, threshold=5.0):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in order(complete):
        s = complete[name]["summary"]
        leads = 5 * np.arange(1, len(s["crps_y_1km"]) + 1)
        line(axes[0], leads, s[f"bss_{threshold}"], name)
        line(axes[1], leads, s[f"freq_bias_{threshold}"], name)
    axes[0].axhline(0, color=INK_2, lw=0.8)
    axes[1].axhline(1, color=INK_2, lw=0.8)
    style(axes[0], ylabel="Brier skill score (0 = climatology)", title=f"Probability of R > {threshold:g} mm/h at 1 km")
    style(axes[1], ylabel="forecast / observed area", title=f"Frequency bias, R > {threshold:g} mm/h (1 = unbiased)")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "3_heavy_rain.png", dpi=130)
    plt.close(fig)


def calibration(complete, out, lead=12, threshold=5.0):
    ensembles = [n for n in order(complete) if complete[n]["summary"]["members"] > 1]
    if not ensembles:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    k = lead - 1
    for name in ensembles:
        s = complete[name]["summary"]
        leads = 5 * np.arange(1, len(s["spread_y"]) + 1)
        line(axes[0], leads, np.array(s["spread_y"]) / np.array(s["rmse_mean_y"]), name)
        hist = np.array(s["rank_histogram"][k])
        line(axes[1], np.arange(len(hist)) + 1, hist * len(hist), name, marker="o", ms=3)
        rel = s[f"reliability_{threshold}"]
        freq, n = np.array(rel["observed_frequency"][k], dtype=float), np.array(rel["pixels"][k], dtype=float)
        prob = np.arange(len(freq)) / (len(freq) - 1)
        ok = n > 1000
        line(axes[2], prob[ok], freq[ok], name, marker="o", ms=3)
    axes[0].axhline(1, color=INK_2, lw=0.8)
    axes[1].axhline(1, color=INK_2, lw=0.8)
    axes[2].plot([0, 1], [0, 1], color=INK_2, lw=0.8)
    style(axes[0], ylabel="spread / RMSE of ensemble mean", title="Spread vs. error (1 = calibrated, < 1 = overconfident)")
    axes[1].set_xticks([1, 5, 10, 15, 21])
    style(axes[1], xlabel="rank of the observation among 20 members", ylabel="relative frequency",
          title=f"Rank histogram at +{5 * lead} min (flat = calibrated)")
    style(axes[2], xlabel="forecast probability", ylabel="observed frequency",
          title=f"Reliability of P(R > {threshold:g} mm/h), +{5 * lead} min")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "4_calibration.png", dpi=130)
    plt.close(fig)


def spectra(complete, out, leads=(6, 12, 24)):
    fig, axes = plt.subplots(1, len(leads), figsize=(15, 4), sharey=True)
    for ax, lead in zip(axes, leads):
        for name in order(complete):
            s = complete[name]["summary"]
            obs, fcst = np.array(s["spectrum_obs"][lead - 1]), np.array(s["spectrum_fcst"][lead - 1])
            if obs.sum() == 0:
                continue
            wavelength = 512 / np.arange(1, len(obs))
            line(ax, wavelength, fcst[1:] / obs[1:], name)
        ax.axhline(1, color=INK_2, lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([256, 100, 50, 20, 10, 5, 2], labels=["256", "100", "50", "20", "10", "5", "2"])
        ax.set_yticks([0.3, 0.5, 0.7, 1, 1.5], labels=["0.3", "0.5", "0.7", "1", "1.5"])
        ax.minorticks_off()
        ax.invert_xaxis()
        style(ax, xlabel="wavelength (km)", title=f"+{5 * lead} min")
    axes[0].set_ylabel("forecast power / observed power")
    axes[0].legend(loc="lower left")
    fig.suptitle("Power spectrum of one member relative to the observations (1 = realistic variability; < 1: too smooth)",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    fig.savefig(out / "5_spectra.png", dpi=130)
    plt.close(fig)


def motion(path: Path, out):
    study = json.loads(path.read_text())
    names = {"zero": ("no motion", "#898781", ":"), "lk": ("Lucas-Kanade (pysteps)", "#1baf7a", "--"),
             "flow": ("variational flow", "#eda100", "-."), "network": ("network: flow + learned correction", "#2a78d6", "-"),
             "oracle": ("oracle: fitted to the future", "#6250d6", (0, (1, 1)))}
    fig, ax = plt.subplots(figsize=(7, 4))
    leads = study["leads_min"]
    for key, (label, color, ls) in names.items():
        if key in study["mae_y"]:
            ax.plot(leads, study["mae_y"][key], color=color, ls=ls, lw=1.8, marker="o", ms=4, label=label)
    style(ax, ylabel="MAE of advected field in log(1 + R)",
          title=f"Advection only: which motion field forecasts best? ({study['samples']} rainy crops)")
    ax.set_xticks(leads)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "6_motion.png", dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--motion", type=Path, default=None)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    complete, partial = load(args.directory)
    every = {**complete, **partial}
    crps_by_lead(every, args.out)
    paired_vs_steps(every, args.out)
    heavy_rain(complete, args.out)
    calibration(complete, args.out)
    spectra(complete, args.out)
    if args.motion:
        motion(args.motion, args.out)
    print(f"figures in {args.out}; complete: {sorted(complete)}; partial: {sorted(partial)}")


if __name__ == "__main__":
    main()
