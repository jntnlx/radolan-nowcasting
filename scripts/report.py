#!/usr/bin/env python
"""
Summarize an evaluation population: score table, lead-time figure, paired differences.

    python scripts/report.py runs/eval/val_120 [--reference steps]

Paired differences use the per-case CRPS of methods scored on identical cases.
Their 95% intervals come from a bootstrap over calendar days (cases on the same
day share weather and are not independent). Evaluations still in progress
(<method>.partial.pt) enter the paired comparison only, on the cases they share.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nowcast.plots import calibration_figure, scores_figure

LEADS_SHOWN = (6, 12, 24)  # 30, 60, 120 min


def paired_difference(a: dict, b: dict, key: str, lead: int, n_boot: int = 2000, seed: int = 0):
    """Mean per-case difference a - b at one lead, with a day-block bootstrap 95% interval."""
    ca = {(c["year"], c["t0"]): c for c in a["cases"]}
    cb = {(c["year"], c["t0"]): c for c in b["cases"]}
    common = sorted(set(ca) & set(cb))
    days = np.array([ca[k]["time"][:10] for k in common])
    diff = np.array([ca[k][key][lead] - cb[k][key][lead] for k in common])
    unique = np.unique(days)
    by_day = [diff[days == d] for d in unique]
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([by_day[i] for i in rng.integers(0, len(by_day), len(by_day))]).mean() for _ in range(n_boot)]
    return diff.mean(), np.percentile(boots, [2.5, 97.5]), len(common)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", type=Path)
    p.add_argument("--reference", default="steps")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    results = {f.stem: json.loads(f.read_text()) for f in sorted(args.directory.glob("*.json"))}
    out = args.out or args.directory
    scores_figure({k: v["summary"] for k, v in results.items()}, out / "scores.png")
    calibration_figure({k: v["summary"] for k, v in results.items()}, path=out / "calibration.png")

    rows = [("CRPS y 1 km", "crps_y_1km"), ("CRPS y 16 km", "crps_y_16km"), ("CRPS mm/h", "crps_mmh"),
            ("FSS 5 mm/h 21 km", "fss_5.0_21km"), ("BSS 5 mm/h", "bss_5.0"), ("freq. bias 5 mm/h", "freq_bias_5.0")]
    lines = ["| method | " + " | ".join(f"{name} @{5 * k} min" for name, _ in rows[:2] for k in LEADS_SHOWN) + " |"]
    lines.append("|---" * (1 + 2 * len(LEADS_SHOWN)) + "|")
    for name, r in results.items():
        s = r["summary"]
        lines.append(f"| {name} | " + " | ".join(f"{s[key][k - 1]:.4f}" for _, key in rows[:2] for k in LEADS_SHOWN) + " |")
    print("\n".join(lines))
    print()
    for name, r in results.items():
        s = r["summary"]
        print(name, " ".join(f"{label}@60={s[key][11]:.3f}" for label, key in rows[2:]))
    print("\nStreakiness (structure-tensor coherence of y in 16 km windows; compare with the observed value) @30/60/120 min,")
    print("and CRPS of 1 h / 2 h accumulations (y of the mean rate; 1 km, 16 km). Only in evaluations made since 2026-11-10.")
    for name, r in results.items():
        s = r["summary"]
        if "anisotropy_fcst" in s:
            print(f"  {name:20s} anisotropy " + " / ".join(f"{s['anisotropy_fcst'][k - 1]:.3f}" for k in LEADS_SHOWN)
                  + "  (observed " + " / ".join(f"{s['anisotropy_obs'][k - 1]:.3f}" for k in LEADS_SHOWN) + ")"
                  + "  acc 1 h: " + ", ".join(f"{s.get(f'crps_acc60_y_{km}km', float('nan')):.4f}" for km in (1, 16))
                  + "  acc 2 h: " + ", ".join(f"{s.get(f'crps_acc120_y_{km}km', float('nan')):.4f}" for km in (1, 16)))

    paired = dict(results)
    for f in sorted(args.directory.glob("*.partial.pt")):
        name = f.name.removesuffix(".partial.pt")
        if name not in paired:
            paired[f"{name} (partial)"] = {"cases": torch.load(f, map_location="cpu", weights_only=False)["cases"]}
    reference = args.reference if args.reference in paired else f"{args.reference} (partial)"
    if reference in paired:
        print(f"\nPaired CRPS differences vs {reference} (negative = better), 95% day-block bootstrap:")
        for name, r in paired.items():
            if name == reference:
                continue
            for key in ("crps_y_1km", "crps_y_16km"):
                parts = []
                for k in LEADS_SHOWN:
                    mean, (lo, hi), n = paired_difference(r, paired[reference], key, k - 1)
                    parts.append(f"@{5 * k}: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]")
                print(f"  {name:20s} {key:12s} " + "  ".join(parts) + f"  (n={n})")


if __name__ == "__main__":
    main()
