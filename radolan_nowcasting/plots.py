"""Forecast plotting and diagnostic visualization via CLI."""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .baselines import optical_flow_nowcast, persistence_nowcast
from .config import PLOTS_DIR, PROCESSED_CACHE_DIR, SEQ_LEN_IN, SEQ_LEN_OUT
from .data import MemmapDataset
from .evaluate import default_device, load_checkpoint_model
from .metrics import BIAS_BINS_MMH
from .transforms import normalized_to_rate_mmh


# Precipitation colormap
# white (dry) → light blue → blue → dark blue → purple → red → yellow
# Based on DWD RADOLAN rain intensity display.
PRECIP_COLORS = [
    (1.0, 1.0, 1.0),    # no rain (dry)
    (0.72, 0.86, 1.0),  # trace
    (0.28, 0.58, 1.0),  # light rain
    (0.08, 0.28, 0.78), # moderate
    (0.48, 0.12, 0.68), # heavy
    (0.82, 0.1, 0.26),  # very heavy
    (1.0, 0.86, 0.05),  # extreme
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot forecasts from a checkpoint.")
    p.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path.")
    p.add_argument("--split", choices=("val_event", "val_all", "test_all"), required=True)
    p.add_argument("--index", type=int, default=0, help="Sample index.")
    p.add_argument("--out-dir", type=Path, default=PLOTS_DIR)
    p.add_argument("--include-optical-flow", action="store_true")
    p.add_argument("--skip-diagnostics", action="store_true")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device) if args.device else default_device()
    model, meta = load_checkpoint_model(args.ckpt, device=device)
    dataset = MemmapDataset(args.split, cache_dir=PROCESSED_CACHE_DIR)
    if args.index >= len(dataset):
        raise SystemExit(f"Index {args.index} out of range (split has {len(dataset)} samples).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = plot_sample(
        model=model, dataset=dataset, split=args.split, index=args.index,
        out_dir=args.out_dir, meta=meta, device=device,
        include_optical_flow=args.include_optical_flow,
        include_diagnostics=not args.skip_diagnostics,
    )
    for path in outputs:
        print(f"Saved: {path}")
    return 0


@torch.no_grad()
def plot_sample(
    *,
    model: torch.nn.Module,
    dataset: MemmapDataset,
    split: str,
    index: int,
    out_dir: Path,
    meta: dict[str, Any],
    device: torch.device,
    include_optical_flow: bool,
    include_diagnostics: bool,
) -> list[Path]:
    """Plot one forecast sample with optional physical diagnostics."""
    inputs, targets = dataset[index]
    inputs_b = inputs.unsqueeze(0).to(device)
    targets_b = targets.unsqueeze(0).to(device)
    pred_b = model(inputs_b)
    persist_b = persistence_nowcast(inputs_b)
    of_b = None
    if include_optical_flow:
        of_b = optical_flow_nowcast(inputs_b, mean=dataset.mean, std=dataset.std)

    # Convert everything to physical mm/h numpy arrays
    arrays = {
        "inputs": _to_mmh(inputs_b, dataset.mean, dataset.std),
        "targets": _to_mmh(targets_b, dataset.mean, dataset.std),
        "model": _to_mmh(pred_b, dataset.mean, dataset.std),
        "persistence": _to_mmh(persist_b, dataset.mean, dataset.std),
        "optical_flow": _to_mmh(of_b, dataset.mean, dataset.std) if of_b is not None else None,
    }

    base = f"{split}_idx{index:05d}"
    forecast_path = out_dir / f"forecast_{base}.png"
    _plot_forecast_panel(arrays, forecast_path, split, index, meta)
    outputs = [forecast_path]

    if include_diagnostics:
        diag_png = out_dir / f"diagnostics_{base}.png"
        diag_json = out_dir / f"diagnostics_{base}.json"
        diagnostics = _plot_diagnostics(arrays, diag_png)
        diagnostics["split"] = split
        diagnostics["index"] = index
        diagnostics["checkpoint"] = meta
        with open(diag_json, "w") as f:
            json.dump(_json_safe(diagnostics), f, indent=2, sort_keys=True)
            f.write("\n")
        outputs.extend([diag_png, diag_json])

    return outputs


def _to_mmh(tensor: torch.Tensor, mean: float, std: float) -> np.ndarray:
    return normalized_to_rate_mmh(tensor.detach().cpu(), mean, std).numpy()[0]


def _plot_forecast_panel(
    arrays: dict[str, np.ndarray | None],
    path: Path,
    split: str,
    index: int,
    meta: dict[str, Any],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    precip_cmap = LinearSegmentedColormap.from_list("precip", PRECIP_COLORS, N=256)
    targets = arrays["targets"]
    model_pred = arrays["model"]
    persistence = arrays["persistence"]
    of_pred = arrays.get("optical_flow")

    # Build row layout
    rows = [
        ("Input (mm/h)", arrays["inputs"], "input"),
        ("Target (mm/h)", targets, "forecast"),
        ("Model (mm/h)", model_pred, "forecast"),
        ("Persistence (mm/h)", persistence, "forecast"),
    ]
    if of_pred is not None:
        rows.append(("Farneback (mm/h)", of_pred, "forecast"))

    error = model_pred - targets
    rows.append(("Model error (mm/h)", error, "error"))

    # Dynamic color scales
    rain_vals = np.concatenate([arrays["inputs"].ravel(), targets.ravel(),
                                model_pred.ravel(), persistence.ravel()])
    rainy = rain_vals[rain_vals > 0.1]
    vmax = float(np.percentile(rainy, 98)) if rainy.size else 1.0
    vmax = max(0.5, min(vmax, 50.0))
    error_abs = max(0.1, float(np.nanpercentile(np.abs(error), 98)))

    # Accumulation panels
    accum = {}
    for key, arr in arrays.items():
        if arr is not None and key != "inputs":
            accum[key] = arr.sum(axis=0) / 12.0

    accum_items = [
        ("Target accum (mm)", accum["targets"], "rain"),
        ("Model accum (mm)", accum["model"], "rain"),
        ("Persistence accum (mm)", accum["persistence"], "rain"),
        ("Model accum error", accum["model"] - accum["targets"], "error"),
    ]
    if of_pred is not None:
        accum_items.insert(3, ("Farneback accum (mm)", accum["optical_flow"], "rain"))

    n_rows = len(rows) + 1
    fig, axes = plt.subplots(n_rows, SEQ_LEN_OUT, figsize=(18, 2.4 * n_rows),
                             constrained_layout=True)
    if n_rows == 1:
        axes = np.asarray([axes])

    title = f"RADOLAN-YW nowcast | {split} index {index}"
    step = meta.get("step")
    if step is not None:
        title += f" | step {step}"
    fig.suptitle(title, fontsize=13)

    # Lead time panels
    for row_idx, (label, data, kind) in enumerate(rows):
        for lead in range(SEQ_LEN_OUT):
            ax = axes[row_idx, lead]
            frame = data[lead]
            if kind == "input":
                frame_label = f"T-{(SEQ_LEN_IN - lead) * 5} min"
            else:
                frame_label = f"T+{(lead + 1) * 5} min"

            if kind == "error":
                norm = TwoSlopeNorm(vmin=-error_abs, vcenter=0.0, vmax=error_abs)
                ax.imshow(frame, cmap="RdBu_r", norm=norm, interpolation="nearest")
            else:
                ax.imshow(np.clip(frame, 0.0, vmax), cmap=precip_cmap,
                          vmin=0.0, vmax=vmax, interpolation="nearest")
            ax.set_title(frame_label, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if lead == 0:
                ax.set_ylabel(label, fontsize=9)

    # Accumulation row
    accum_vmax = max(0.1, float(np.nanmax([
        accum["targets"].max(), accum["model"].max(), accum["persistence"].max()
    ])))
    accum_err_abs = max(0.1, float(np.nanmax(np.abs(accum["model"] - accum["targets"]))))

    for col in range(SEQ_LEN_OUT):
        ax = axes[-1, col]
        ax.set_xticks([])
        ax.set_yticks([])
        if col >= len(accum_items):
            ax.axis("off")
            continue
        label, frame, kind = accum_items[col]
        if kind == "error":
            norm = TwoSlopeNorm(vmin=-accum_err_abs, vcenter=0.0, vmax=accum_err_abs)
            ax.imshow(frame, cmap="RdBu_r", norm=norm, interpolation="nearest")
        else:
            ax.imshow(np.clip(frame, 0.0, accum_vmax), cmap=precip_cmap,
                      vmin=0.0, vmax=accum_vmax, interpolation="nearest")
        ax.set_title(label, fontsize=8)
        if col == 0:
            ax.set_ylabel("30-min panels", fontsize=9)

    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_diagnostics(arrays: dict, path: Path) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = arrays["targets"]
    sources = {"model": arrays["model"], "persistence": arrays["persistence"]}
    if arrays.get("optical_flow") is not None:
        sources["farneback"] = arrays["optical_flow"]

    dry_summary = _dry_false_alarm_summary(sources, targets)
    bias = _conditional_bias_summary(sources, targets)
    psd = _psd_summary(sources, targets)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), constrained_layout=True)

    # 1. False alarms below threshold
    thresholds = [0.1, 1.0, 5.0]
    x = np.arange(len(thresholds))
    width = 0.24
    for idx, (name, values) in enumerate(dry_summary.items()):
        offsets = x + (idx - (len(dry_summary) - 1) / 2.0) * width
        axes[0].bar(offsets, [values[str(t)] for t in thresholds], width, label=name)
    axes[0].set_title("Below-threshold false alarms")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{t:g}" for t in thresholds])
    axes[0].set_xlabel("Threshold (mm/h)")
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(bottom=0.0)
    axes[0].legend(fontsize=8)

    # 2. Conditional bias
    bin_names = [name for name, _, _ in BIAS_BINS_MMH]
    x_bias = np.arange(len(bin_names))
    for idx, (name, values) in enumerate(bias.items()):
        offsets = x_bias + (idx - (len(bias) - 1) / 2.0) * 0.24
        axes[1].bar(offsets, [values[b] for b in bin_names], 0.24, label=name)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Conditional bias")
    axes[1].set_xticks(x_bias)
    axes[1].set_xticklabels(bin_names, rotation=30, ha="right")
    axes[1].set_ylabel("Forecast − target (mm/h)")

    # 3. Radial PSD of 30 minute accumulation
    for name, values in psd.items():
        r = np.asarray(values["radius"])
        p = np.asarray(values["power"])
        mask = r > 0
        axes[2].loglog(r[mask], p[mask], label=name)
    axes[2].set_title("Radial PSD of 30-min accumulation")
    axes[2].set_xlabel("Radial wavenumber")
    axes[2].set_ylabel("Power")
    axes[2].legend(fontsize=8)

    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "below_threshold_false_alarm_rate": dry_summary,
        "conditional_bias_mmh": bias,
        "radial_psd": psd,
    }


def _dry_false_alarm_summary(sources, targets):
    out = {}
    for name, forecast in sources.items():
        out[name] = {}
        for t in (0.1, 1.0, 5.0):
            below = targets <= t
            denom = int(below.sum())
            rate = float(((forecast > t) & below).sum()) / denom if denom > 0 else float("nan")
            out[name][str(t)] = rate
    return out


def _conditional_bias_summary(sources, targets):
    out = {}
    for src_name, forecast in sources.items():
        out[src_name] = {}
        for bin_name, lo, hi in BIAS_BINS_MMH:
            if math.isinf(hi):
                mask = targets > lo
            elif lo == 0.0:
                mask = targets <= hi
            else:
                mask = (targets > lo) & (targets <= hi)
            out[src_name][bin_name] = (
                float((forecast[mask] - targets[mask]).mean())
                if mask.any() else float("nan")
            )
    return out


def _psd_summary(sources, targets):
    accum = {"target": targets.sum(axis=0) / 12.0}
    accum.update({n: v.sum(axis=0) / 12.0 for n, v in sources.items()})
    return {
        name: {"radius": r.tolist(), "power": p.tolist()}
        for name, field in accum.items()
        for r, p in [_radial_psd(field)]
    }


def _radial_psd(field: np.ndarray):
    centered = field.astype(np.float64) - float(np.mean(field))
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    yy, xx = np.indices(power.shape)
    cy, cx = (power.shape[0] - 1) / 2.0, (power.shape[1] - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int64)
    total = np.bincount(radius.ravel(), weights=power.ravel())
    count = np.bincount(radius.ravel())
    radial = total / np.maximum(count, 1)
    return np.arange(radial.size, dtype=np.float64), radial


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
