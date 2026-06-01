"""Training loop and CLI."""

import argparse
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import (
    BATCH_SIZE,
    DATA_VERSION,
    GRADIENT_CLIP,
    LEARNING_RATE,
    MAX_STEPS,
    NUM_WORKERS,
    PROCESSED_CACHE_DIR,
    RUNS_DIR,
    SEED,
    TIME_BUDGET_SEC,
    VAL_INTERVAL_STEPS,
    WEIGHT_DECAY,
)
from .data import get_dataloaders
from .evaluate import evaluate
from .losses import ForecastLoss
from .model import UNet
from .transforms import load_normalization_stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the RADOLAN-YW nowcasting baseline.")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--epochs", type=int, default=0,
                    help="Number of epochs. 0 = train until max-steps or time budget.")
    p.add_argument("--time-budget", type=int, default=TIME_BUDGET_SEC,
                    help="Wall-clock budget in seconds. 0 = disable.")
    p.add_argument("--val-interval-steps", type=int, default=VAL_INTERVAL_STEPS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--no-compile", action="store_true",
                    help="Disable torch.compile even on CUDA.")
    p.add_argument("--include-optical-flow-final-eval", action="store_true",
                    help="Evaluate Farneback baseline during final evaluation.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        torch.set_float32_matmul_precision("medium")
    except AttributeError:
        pass
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_id = args.run_id or f"raw_unet_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    config = {
        "run_id": run_id, "data_version": DATA_VERSION,
        "max_steps": args.max_steps, "epochs": args.epochs,
        "time_budget": args.time_budget, "batch_size": args.batch_size,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "seed": args.seed, "device": str(device),
        "compile": not args.no_compile and device.type == "cuda",
    }
    _write_json(run_dir / "run_config.json", config)

    # Data
    mean, std = load_normalization_stats(PROCESSED_CACHE_DIR / "stats.json")
    loaders = get_dataloaders(
        batch_size=args.batch_size, num_workers=NUM_WORKERS,
        splits=("train_event", "val_event", "val_all"),
        cache_dir=PROCESSED_CACHE_DIR,
    )
    train_loader = loaders["train_event"]
    if len(train_loader) == 0:
        raise RuntimeError("train_event has zero batches. Reduce --batch-size or rebuild cache.")

    # Model and optimizer
    model = UNet().to(device)
    use_compile = not args.no_compile and device.type == "cuda" and hasattr(torch, "compile")
    train_model = torch.compile(model) if use_compile else model
    criterion = ForecastLoss(mean=mean, std=std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    started = time.monotonic()
    best_score = float("inf")
    best_step = 0
    best_val_event = None
    best_val_all = None
    step = 0
    epoch = 0
    time_budget = None if args.time_budget <= 0 else args.time_budget

    log_path = run_dir / "train_log.tsv"
    with open(log_path, "w") as log:
        log.write("step\tepoch\tloss\thuber_norm\tphysical_mae_mmh\t"
                  "threshold_mae_mmh\telapsed_sec\tval_event_rmse\tval_all_rmse\n")

        while step < args.max_steps:
            if args.epochs > 0 and epoch >= args.epochs:
                break
            epoch += 1

            for inputs, targets in train_loader:
                step += 1
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with _autocast(device):
                    predictions = train_model(inputs)
                    loss, parts = criterion(predictions, targets)

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(f"Non-finite gradient at step {step}")
                optimizer.step()

                elapsed = time.monotonic() - started
                val_event_rmse = ""
                val_all_rmse = ""

                # Validation
                should_validate = (
                    step == 1
                    or step % args.val_interval_steps == 0
                    or step >= args.max_steps
                )
                if should_validate:
                    val_event = evaluate(model, loaders["val_event"],
                                         device=device, mean=mean, std=std)
                    val_all = evaluate(model, loaders["val_all"],
                                       device=device, mean=mean, std=std)
                    val_event_rmse = f"{val_event['rmse_log_norm']:.8g}"
                    val_all_rmse = f"{val_all['rmse_log_norm']:.8g}"

                    score = float(val_event["rmse_log_norm"])
                    if score < best_score:
                        best_score = score
                        best_step = step
                        best_val_event = val_event
                        best_val_all = val_all
                        _save_checkpoint(
                            run_dir / "checkpoint_best.pt",
                            model=model, optimizer=optimizer, config=config,
                            step=step, epoch=epoch,
                            val_event=val_event, val_all=val_all,
                        )
                    train_model.train()

                # Log
                log.write(
                    f"{step}\t{epoch}\t{float(loss.detach()):.8g}\t"
                    f"{_part(parts, 'huber_norm'):.8g}\t"
                    f"{_part(parts, 'physical_mae_mmh'):.8g}\t"
                    f"{_part(parts, 'threshold_weighted_mae_mmh'):.8g}\t"
                    f"{elapsed:.3f}\t{val_event_rmse}\t{val_all_rmse}\n"
                )
                log.flush()

                if step >= args.max_steps:
                    break
                if time_budget is not None and elapsed >= time_budget:
                    break

            if time_budget is not None and (time.monotonic() - started) >= time_budget:
                break

    # Finalize
    final_val_event = evaluate(
        model, loaders["val_event"], device=device, mean=mean, std=std,
        include_optical_flow=args.include_optical_flow_final_eval,
    )
    final_val_all = evaluate(
        model, loaders["val_all"], device=device, mean=mean, std=std,
        include_optical_flow=args.include_optical_flow_final_eval,
    )

    if best_val_event is None:
        best_step = step
        best_val_event = final_val_event
        best_val_all = final_val_all
        best_score = float(final_val_event["rmse_log_norm"])
        _save_checkpoint(
            run_dir / "checkpoint_best.pt",
            model=model, optimizer=optimizer, config=config,
            step=step, epoch=epoch,
            val_event=final_val_event, val_all=final_val_all,
        )

    _save_checkpoint(
        run_dir / "checkpoint_final.pt",
        model=model, optimizer=optimizer, config=config,
        step=step, epoch=epoch,
        val_event=final_val_event, val_all=final_val_all,
    )

    metrics = {
        "run_id": run_id, "data_version": DATA_VERSION,
        "best_step": best_step, "final_step": step,
        "best_val_event": best_val_event, "best_val_all": best_val_all,
        "final_val_event": final_val_event, "final_val_all": final_val_all,
    }
    _write_json(run_dir / "metrics.json", metrics)

    print(
        f"Training complete: final_step={step} best_step={best_step} "
        f"best_val_event_rmse={best_val_event.get('rmse_log_norm', float('nan')):.6f}"
    )
    return 0


# Helpers functiosn

def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _part(parts: dict[str, torch.Tensor], key: str) -> float:
    v = parts.get(key)
    return float(v.detach().cpu()) if v is not None else float("nan")


def _save_checkpoint(path: Path, *, model, optimizer, config, step, epoch,
                     val_event, val_all) -> None:
    torch.save({
        "step": step, "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "run_config": config,
        "val_event": val_event, "val_all": val_all,
    }, path)


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(_json_safe(data), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


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
