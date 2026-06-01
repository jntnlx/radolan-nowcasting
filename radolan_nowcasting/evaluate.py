"""
Eval CLI setup.
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .baselines import CV2_AVAILABLE, optical_flow_nowcast, persistence_nowcast
from .config import BATCH_SIZE, NUM_WORKERS, PROCESSED_CACHE_DIR
from .data import get_loader
from .metrics import ForecastMetricAccumulator
from .model import UNet


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(
    model,
    loader,
    *,
    device: torch.device | str | None = None,
    include_optical_flow: bool = False,
    mean: float | None = None,
    std: float | None = None,
) -> dict[str, Any]:
    """
    Evaluate model and baselines on one split.
    Note: Set `model=None` to compute only persistence/Farneback.
    """
    if device is None:
        device = default_device()
    else:
        device = torch.device(device)

    # Resolve normalization stats from loader dataset or explicit args
    dataset = getattr(loader, "dataset", None)
    if mean is None:
        mean = getattr(dataset, "mean", None)
    if std is None:
        std = getattr(dataset, "std", None)
    if mean is None or std is None:
        raise ValueError("Evaluation requires mean and std.")
    mean, std = float(mean), float(std)

    if model is not None:
        model = model.to(device)
        model.eval()

    model_acc = ForecastMetricAccumulator(mean, std) if model is not None else None
    persist_acc = ForecastMetricAccumulator(mean, std)
    of_acc = ForecastMetricAccumulator(mean, std) if include_optical_flow else None
    of_batches = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        persist_acc.update(persistence_nowcast(inputs), targets)

        if model is not None:
            predictions = model(inputs)
            if predictions.shape != targets.shape:
                raise ValueError(
                    f"Model output shape {predictions.shape} != target shape {targets.shape}"
                )
            model_acc.update(predictions, targets)

        if include_optical_flow:
            of_pred = optical_flow_nowcast(inputs, mean=mean, std=std)
            if of_pred is not None:
                of_batches += 1
                of_acc.update(of_pred, targets)

    results: dict[str, Any] = {}
    if model_acc is not None:
        results.update(model_acc.finalize(reference=persist_acc))

    persist_results = persist_acc.finalize(prefix="persist")
    results.update(persist_results)
    if "persist_rmse_log_norm" in persist_results:
        results["persistence_rmse_log_norm"] = persist_results["persist_rmse_log_norm"]

    if include_optical_flow and of_acc is not None and of_batches > 0:
        results.update(of_acc.finalize(prefix="of", reference=persist_acc))
        results["optical_flow_available"] = True
        results["optical_flow_batches"] = of_batches
    else:
        results["optical_flow_available"] = bool(CV2_AVAILABLE)
        results["optical_flow_batches"] = of_batches
        for key in ("of_rmse_log_norm", "of_rmse_mmh", "of_mae_mmh",
                     "of_skill_score", "of_accum30_rmse_mm"):
            results[key] = float("nan")

    return _clean_metrics(results)


def load_checkpoint_model(
    ckpt_path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> tuple[UNet, dict[str, Any]]:
    """
    Load U-Net checkpoint from explicit path.
    """
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if device is None:
        device = default_device()
    else:
        device = torch.device(device)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        meta = {
            "checkpoint_path": str(path),
            "step": checkpoint.get("step"),
            "epoch": checkpoint.get("epoch"),
            "run_config": checkpoint.get("run_config", {}),
        }
    else:
        state_dict = checkpoint if isinstance(checkpoint, dict) else {}
        meta = {"checkpoint_path": str(path), "step": None}

    # Handle torch.compile prefix
    prefix = "_orig_mod."
    if any(str(k).startswith(prefix) for k in state_dict):
        state_dict = {
            str(k)[len(prefix):] if str(k).startswith(prefix) else str(k): v
            for k, v in state_dict.items()
        }

    model = UNet()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, meta


# CLI

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a nowcasting checkpoint.")
    p.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path.")
    p.add_argument("--split", choices=("val_event", "val_all", "test_all"), required=True)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument("--include-optical-flow", action="store_true")
    p.add_argument("--out", type=Path, default=None, help="Metrics JSON path.")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device) if args.device else default_device()
    model, meta = load_checkpoint_model(args.ckpt, device=device)
    loader = get_loader(
        args.split, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, cache_dir=PROCESSED_CACHE_DIR,
    )
    metrics = evaluate(
        model, loader, device=device,
        include_optical_flow=args.include_optical_flow,
    )

    payload = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "checkpoint": meta,
        "include_optical_flow": args.include_optical_flow,
        "metrics": metrics,
    }
    suffix = "with_of" if args.include_optical_flow else "model"
    out_path = args.out or (Path(args.ckpt).parent / f"evaluation_{args.split}_{suffix}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")

    print(f"Saved: {out_path}")
    print(f"  rmse_log_norm={metrics.get('rmse_log_norm', float('nan')):.6f}")
    print(f"  persist_rmse_log_norm={metrics.get('persist_rmse_log_norm', float('nan')):.6f}")
    return 0


# Helper Functions

def _clean_metrics(metrics: dict) -> dict:
    return {k: (v if not isinstance(v, float) or math.isfinite(v) else float("nan"))
            for k, v in metrics.items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
