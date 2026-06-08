"""

RADOLAN-YW data acquisition, cache building and dataset loading.

Data pipeline:

    1. Download raw monthly RADOLAN-YW (5min) archives from DWD (open data)
    2. Read frames from tar archives by using wradlib
    3. Convert raw to rain rate (mm/h) with strict unit validation
    4. Extract patch sequences while rejecting any with non-finite pixels (e.g. NaN)
    5. Apply event filtering/reservoir sampling per split
    6. Store as .npy files for subsequent trainign/evaluation

Missing data handling: 

    Sequences containing NaN/Inf in any input and/or target pixel are excluded. No zero-fill/imputation. 
    More conservative than usual but simple Ansatz that ensures no training on corrupted or missing data presently. 

Memory management (required due to previously encountered memroy limitations on local machine): 

    Event splits flush accepted samples to temp. chunk files, then subsequently concatenate via memory mapping approach.  
    Representative splits employ online reservoir sampling with bounded memory, i.e. only limited amount of items are stored in RAM simultaneously.

"""

import io
import json
import math
import shutil
import tarfile
import time
import numpy as np

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from urllib.error import URLError
from urllib.request import urlretrieve

from .config import (

    ACTIVITY_PERCENTILE,
    ACTIVITY_THRESHOLD_MMH,
    CROP_SIZE,
    DATA_VERSION,
    DWD_BASE_URL,
    FRAME_INTERVAL_SEC,
    GAP_TOLERANCE_SEC,
    MAX_RAIN_RATE_MMH,
    MAX_TEST_ALL_SAMPLES,
    MAX_VAL_ALL_SAMPLES,
    PATCH_STRIDE,
    PROCESSED_CACHE_DIR,
    RADOLAN_SHAPE,
    RAW_CACHE_DIR,
    REPRESENTATIVE_SEED,
    SEQ_LEN_IN,
    SEQ_LEN_OUT,
    SEQ_LEN_TOTAL,
    TEMPORAL_STRIDE,
    TEST_YEAR,
    TRAIN_YEARS,
    VAL_YEAR,
    WARM_SEASON_MONTHS,

)

from .transforms import load_normalization_stats, rate_mmh_to_log


# Archive Access

# Download and validate RADOLAN-YW monthly archives from DWD.
# Reprocessed dataset (2017.002) stored as (nested) tarballs: monthly.tar → daily.tar.gz → individual 5min frames


def month_archive_path(year: int, month: int, raw_dir: Path = RAW_CACHE_DIR) -> Path:
    """
    Expected local path for RADOLAN-YW monthly archive.
    """
    return raw_dir / f"{year:04d}" / f"YW2017.002_{year:04d}{month:02d}.tar"


def month_archive_url(year: int, month: int) -> str:
    """
    URL for one RADOLAN-YW monthly archive from DWD.
    """
    filename = f"YW2017.002_{year:04d}{month:02d}.tar"
    return f"{DWD_BASE_URL}/{year:04d}/{filename}"


def ensure_month_archive(

    year: int,
    month: int,
    raw_dir: Path = RAW_CACHE_DIR,
    *,
    skip_download: bool = False,

) -> Path:
    """
    Return valid local archive and download it.
    """
    path = month_archive_path(year, month, raw_dir)
    if path.exists():
        try:
            with tarfile.open(path, "r:") as tar:
                tar.getmembers()
            return path
        except (tarfile.ReadError, tarfile.CompressionError, EOFError) as exc:
            if skip_download:
                raise RuntimeError(f"Corrupt archive: {path}") from exc
            path.unlink()

    if skip_download:
        raise FileNotFoundError(
            f"RADOLAN-YW archive not found: {path}. "
            "Run without --skip-download to fetch it from DWD."
        )

    # Download with retries
    path.parent.mkdir(parents=True, exist_ok=True)
    url = month_archive_url(year, month)
    for attempt in range(3):
        try:
            urlretrieve(url, path)
            with tarfile.open(path, "r:") as tar:
                tar.getmembers()
            return path
        except (URLError, OSError, tarfile.TarError, EOFError) as exc:
            if path.exists():
                path.unlink()
            if attempt == 2:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
            time.sleep(5 * (attempt + 1))

    raise RuntimeError(f"Failed to download {url}")


# Unit Conversion

# RADOLAN-YW with intervalunit=0 reports 5min accumulated depth. Multiply by 12 to get mm/h.
# Conversion path only via convert_radolan_to_rate_mmh() fct.

def _metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metadata_text(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key, "")
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def convert_radolan_to_rate_mmh(

    data: np.ndarray,
    metadata: dict[str, Any],
    *,
    strict_units: bool = True,

) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Convert raw RADOLAN-YW values to rain rate (mm/h).
    Return (rate_mmh, conversion_info). conversion_info dict documents unit interpretation.
    """
    product = _metadata_text(metadata, "producttype")
    if product != "YW" and strict_units:
        raise RuntimeError(f"Expected producttype='YW', got {product!r}")

    frame = np.asarray(data, dtype=np.float32).copy()
    frame[~np.isfinite(frame)] = np.nan
    frame[frame < -0.5] = np.nan  # RADOLAN sentinel values

    interval_unit = _metadata_int(metadata, "intervalunit")
    interval_sec = _metadata_int(metadata, "intervalseconds")

    if interval_sec is None or interval_sec <= 0:
        if strict_units:
            raise RuntimeError(
                "RADOLAN metadata missing valid 'intervalseconds'; "
                "cannot determine unit conversion."
            )
        interval_sec = 300

    if interval_unit == 0:
        # Accumulated depth over interval → rate (mm/h)
        factor = 3600.0 / float(interval_sec)
        status = "ok"
    elif interval_unit == 1:
        # Already in mm/h
        factor = 1.0
        status = "ok"
    else:
        if strict_units:
            raise RuntimeError(
                f"Unknown intervalunit={interval_unit!r}; "
                "cannot determine if values are depth or rate."
            )
        factor = 3600.0 / float(interval_sec)
        status = "warning"

    conversion_info = {
        "intervalunit": interval_unit,
        "intervalseconds": interval_sec,
        "factor": float(factor),
        "status": status,
        "producttype": product,
    }
    return frame * factor, conversion_info


def _read_radolan_bytes(data_bytes: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Read one RADOLAN frame from raw via wradlib.
    """
    try:
        import wradlib
    except ImportError as exc:
        raise RuntimeError(
            "wradlib is required to read RADOLAN frames. "
            "Install it via: conda install -c conda-forge wradlib"
        ) from exc
    data, metadata = wradlib.io.read_radolan_composite(io.BytesIO(data_bytes))
    return data, dict(metadata)


def iter_daily_frames(

    archive_path: Path,
    *,
    max_days: int | None = None,
    max_frames_per_day: int | None = None,
    strict_units: bool = True,

):
    """
    Yield (day_name, {timestamp: rate_mmh}) for each day in a monthly archive.
    All unit conversion via convert_radolan_to_rate_mmh().
    """
    days_yielded = 0
    with tarfile.open(archive_path, "r:") as outer_tar:
        daily_members = sorted(
            (m for m in outer_tar.getmembers() if m.isfile() and m.name.endswith(".tar.gz")),
            key=lambda m: m.name,
        )
        for daily_member in daily_members:
            if max_days is not None and days_yielded >= max_days:
                break
            daily_file = outer_tar.extractfile(daily_member)
            if daily_file is None:
                continue

            frames: dict[datetime, np.ndarray] = {}
            with tarfile.open(fileobj=daily_file, mode="r|gz") as daily_tar:
                for frame_member in daily_tar:
                    if not frame_member.isfile():
                        continue
                    file_obj = daily_tar.extractfile(frame_member)
                    if file_obj is None:
                        continue
                    data, metadata = _read_radolan_bytes(file_obj.read())
                    ts = metadata.get("datetime")
                    if not isinstance(ts, datetime):
                        continue

                    # Single conversion path
                    rate_mmh, _ = convert_radolan_to_rate_mmh(
                        data, metadata, strict_units=strict_units,
                    )
                    frames[ts] = rate_mmh

            if max_frames_per_day is not None:
                sorted_times = sorted(frames)[:max_frames_per_day]
                frames = {t: frames[t] for t in sorted_times}

            days_yielded += 1
            yield daily_member.name, frames


# Unit Audit

# Verify RADOLAN-YW unit interpretation by inspection of raw frame metadata.

def audit_radolan_units(

    year_months: list[tuple[int, int]],
    *,
    frames_per_archive: int = 1,
    raw_dir: Path = RAW_CACHE_DIR,
    skip_download: bool = False,
    strict_units: bool = True,

) -> dict[str, Any]:
    """
    Check RADOLAN-YW frame metadata w.r.t. unit consistency.
    """
    records = []
    for year, month in year_months:
        archive = ensure_month_archive(year, month, raw_dir, skip_download=skip_download)
        n_read = 0
        with tarfile.open(archive, "r:") as outer_tar:
            daily_members = sorted(
                (m for m in outer_tar.getmembers() if m.isfile() and m.name.endswith(".tar.gz")),
                key=lambda m: m.name,
            )
            for daily_member in daily_members:
                if n_read >= frames_per_archive:
                    break
                daily_file = outer_tar.extractfile(daily_member)
                if daily_file is None:
                    continue
                daily_bytes = daily_file.read()
                with tarfile.open(fileobj=io.BytesIO(daily_bytes), mode="r:gz") as daily_tar:
                    for frame_member in sorted(daily_tar.getmembers(), key=lambda m: m.name):
                        if not frame_member.isfile() or n_read >= frames_per_archive:
                            break
                        file_obj = daily_tar.extractfile(frame_member)
                        if file_obj is None:
                            continue
                        data, metadata = _read_radolan_bytes(file_obj.read())
                        rate_mmh, conv_info = convert_radolan_to_rate_mmh(
                            data, metadata, strict_units=strict_units
                        )
                        valid = rate_mmh[np.isfinite(rate_mmh)]
                        records.append({
                            "archive": str(archive),
                            "frame": frame_member.name,
                            "datetime": metadata.get("datetime", "").isoformat()
                            if isinstance(metadata.get("datetime"), datetime) else "",
                            "producttype": _metadata_text(metadata, "producttype"),
                            "intervalunit": _metadata_int(metadata, "intervalunit"),
                            "intervalseconds": _metadata_int(metadata, "intervalseconds"),
                            "factor": conv_info["factor"],
                            "status": conv_info["status"],
                            "shape": list(rate_mmh.shape),
                            "valid_min_mmh": float(valid.min()) if valid.size else None,
                            "valid_max_mmh": float(valid.max()) if valid.size else None,
                            "missing_fraction": float((~np.isfinite(rate_mmh)).mean()),
                        })
                        n_read += 1

    all_expected = all(
        r["producttype"] == "YW"
        and r["intervalunit"] == 0
        and r["intervalseconds"] == 300
        and abs(r["factor"] - 12.0) < 1e-9
        for r in records
    )
    return {
        "created": datetime.now(timezone.utc).isoformat(),
        "data_version": DATA_VERSION,
        "summary": {
            "n_archives": len(year_months),
            "n_frames": len(records),
            "all_expected_yw_depth_300s": all_expected,
            "all_unit_status_ok": all(r["status"] == "ok" for r in records),
            "intervalunit_values": sorted({r["intervalunit"] for r in records}),
            "intervalseconds_values": sorted({r["intervalseconds"] for r in records}),
            "source_to_model_factors": sorted({r["factor"] for r in records}),
        },
        "frames": records,
    }


# Cache Building

# Build processed cache: 
#   - Extract patch sequences from default RADOLAN-YW frames that provide the entire grid.
#   - Reject any with non-finite pixels.
#   - Apply event filtering/reservoir sampling.
#   - Store as numpy arrays mapped to memory.
#
# Memory management:
#   - Event splits: flush accepted samples to temp. chunk files every `chunk_size` samples (Peak RAM ≈ chunk_size x sample_size).
#   - Representative splits: online reservoir sampling (Peak RAM ≈ max_samples x sample_size, independent of how many candidates exist).

def _patch_positions() -> list[tuple[int, int]]:
    """
    Crop origins on RADOLAN grid without overlappign.
    """
    return [
        (row, col)
        for row in range(0, RADOLAN_SHAPE[0] - CROP_SIZE + 1, PATCH_STRIDE)
        for col in range(0, RADOLAN_SHAPE[1] - CROP_SIZE + 1, PATCH_STRIDE)
    ]


def _find_contiguous_runs(
    sorted_times: list[datetime],
) -> list[list[datetime]]:
    """
    Find timestamp sequences with expected 5min cadence.
    """
    if not sorted_times:
        return []
    runs: list[list[datetime]] = []
    current = [sorted_times[0]]
    for ts in sorted_times[1:]:
        delta = (ts - current[-1]).total_seconds()
        if abs(delta - FRAME_INTERVAL_SEC) <= GAP_TOLERANCE_SEC:
            current.append(ts)
        else:
            if len(current) >= SEQ_LEN_TOTAL:
                runs.append(current)
            current = [ts]
    if len(current) >= SEQ_LEN_TOTAL:
        runs.append(current)
    return runs


def _save_meta_npz(metas: list[dict[str, Any]], path: Path) -> None:
    """
    Save sample metadata as compressed numpy arrays.
    """
    if not metas:
        np.savez_compressed(path)
        return
    arrays = {}
    for key in metas[0]:
        values = [m[key] for m in metas]
        first = values[0]
        if isinstance(first, str):
            w = max(1, max(len(str(v)) for v in values))
            arrays[key] = np.asarray(values, dtype=f"<U{w}")
        elif isinstance(first, (bool, np.bool_)):
            arrays[key] = np.asarray(values, dtype=np.bool_)
        elif isinstance(first, (int, np.integer)):
            arrays[key] = np.asarray(values, dtype=np.int64)
        else:
            arrays[key] = np.asarray(values, dtype=np.float64)
    np.savez_compressed(path, **arrays)


def _concat_meta_npz(meta_batches: list[dict[str, np.ndarray]], path: Path) -> None:
    """
    Concatenate metadata arrays from multiple chunks.
    """
    meta_batches = [b for b in meta_batches if b]
    if not meta_batches:
        np.savez_compressed(path)
        return
    keys = list(meta_batches[0].keys())
    arrays = {
        key: np.concatenate([batch[key] for batch in meta_batches], axis=0)
        for key in keys
    }
    np.savez_compressed(path, **arrays)


def _sequence_metadata(
    times: list[datetime], row: int, col: int, patches_mmh: np.ndarray,
) -> dict[str, Any]:
    """
    Build per-sample metadata for one sequence.
    """
    target = patches_mmh[SEQ_LEN_IN:]
    return {
        "start_time": times[0].strftime("%Y-%m-%dT%H:%M:%S"),
        "target_end_time": times[-1].strftime("%Y-%m-%dT%H:%M:%S"),
        "year": int(times[0].year),
        "month": int(times[0].month),
        "patch_row": int(row),
        "patch_col": int(col),
        "target_max_mmh": float(np.max(target)),
        "target_mean_mmh": float(np.mean(target)),
        "target_p90_mmh": float(np.percentile(target, 90)),
        "target_rain_frac_0_1_mmh": float(np.mean(target > 0.1)),
        "target_rain_frac_1_0_mmh": float(np.mean(target > 1.0)),
        "target_rain_frac_5_0_mmh": float(np.mean(target > 5.0)),
    }


def _extract_sequences(

    frames: dict[datetime, np.ndarray],
    *,
    split_mode: str,
    activity_threshold: float,
    samples: list[np.ndarray],
    metas: list[dict[str, Any]],
    stats: dict[str, int],
    normalizer: dict[str, float] | None,

) -> None:
    """
    Extract valid sequences from a block of frames, appending to lists.

    Represents inner loop of cache building. 
    For each run of frames / spatial patch position, check for non-finite pixels, apply event filtering and store accepted sequences.
    """
    sorted_times = sorted(frames)
    if len(sorted_times) < SEQ_LEN_TOTAL:
        return

    positions = _patch_positions()
    runs = _find_contiguous_runs(sorted_times)

    for run in runs:
        for start in range(0, len(run) - SEQ_LEN_TOTAL + 1, TEMPORAL_STRIDE):
            seq_times = run[start:start + SEQ_LEN_TOTAL]
            source = [frames[t] for t in seq_times]

            for row, col in positions:
                stats["total_candidates"] += 1
                patches = np.stack([
                    f[row:row + CROP_SIZE, col:col + CROP_SIZE] for f in source
                ]).astype(np.float32)

                # Strict rejection: any non-finite pixel → reject
                if not np.isfinite(patches).all():
                    stats["nonfinite_rejected"] += 1
                    continue

                # Clip extreme rates before log transform
                clip_mask = patches > MAX_RAIN_RATE_MMH
                if clip_mask.any():
                    stats["rate_clipped_samples"] += 1
                    stats["rate_clipped_pixels"] += int(clip_mask.sum())
                    patches = np.clip(patches, None, MAX_RAIN_RATE_MMH)

                # Event filtering: only keep sequences with significant rain
                if split_mode == "event":
                    target_p90 = float(np.percentile(patches[SEQ_LEN_IN:], ACTIVITY_PERCENTILE))
                    if target_p90 < activity_threshold:
                        stats["inactive"] += 1
                        continue
                    stats["active"] += 1

                # Convert to log-rain and store
                log_sample = rate_mmh_to_log(patches)
                stored = log_sample.astype(np.float16)
                samples.append(stored)
                metas.append(_sequence_metadata(seq_times, row, col, patches))

                # Accumulate running statistics (train_event only)
                if normalizer is not None:
                    vals = stored.astype(np.float64)
                    normalizer["sum"] += float(vals.sum())
                    normalizer["sq_sum"] += float((vals * vals).sum())
                    normalizer["n"] += int(vals.size)


# Chunk "Flushing" (Event Splits)

# Flush accepted samples to temporary chunk files periodically to bound peak RAM usage.
# Final concatenation via output array mapped to memory.

def _flush_chunk(

    samples: list[np.ndarray],
    metas: list[dict[str, Any]],
    temp_dir: Path,
    chunk_files: list[tuple[Path, Path, int]],

) -> None:
    """
    Write one chunk of samples and metas to temp. files.
    """
    if not samples:
        return
    idx = len(chunk_files)
    npy_path = temp_dir / f"chunk_{idx:05d}.npy"
    meta_path = temp_dir / f"chunk_{idx:05d}_meta.npz"
    arr = np.stack(samples).astype(np.float16)
    np.save(npy_path, arr)
    _save_meta_npz(metas, meta_path)
    chunk_files.append((npy_path, meta_path, int(arr.shape[0])))


def _finalize_chunks(

    chunk_files: list[tuple[Path, Path, int]],
    output_prefix: Path,

) -> tuple[Path, Path, int]:
    """
    Concatenate chunk files into final .npy + metadata helper.
    """
    npy_path = Path(str(output_prefix) + ".npy")
    meta_path = Path(str(output_prefix) + "_meta.npz")

    if not chunk_files:
        empty = np.empty((0, SEQ_LEN_TOTAL, CROP_SIZE, CROP_SIZE), dtype=np.float16)
        np.save(npy_path, empty)
        _save_meta_npz([], meta_path)
        return npy_path, meta_path, 0

    total = sum(count for _, _, count in chunk_files)
    sample_shape = (SEQ_LEN_TOTAL, CROP_SIZE, CROP_SIZE)

    # Concatenate arrays via memory-mapped output
    final = np.lib.format.open_memmap(
        npy_path, mode="w+", dtype=np.float16, shape=(total,) + sample_shape,
    )
    offset = 0
    meta_batches = []
    for chunk_npy, chunk_meta, count in chunk_files:
        chunk = np.load(chunk_npy, mmap_mode="r")
        final[offset:offset + count] = chunk
        offset += count
        meta_batches.append(dict(np.load(chunk_meta, allow_pickle=False)))
    final.flush()
    del final

    _concat_meta_npz(meta_batches, meta_path)

    # Clean up temp files
    for chunk_npy, chunk_meta, _ in chunk_files:
        chunk_npy.unlink(missing_ok=True)
        chunk_meta.unlink(missing_ok=True)

    return npy_path, meta_path, total


# Simple Write (Small Splits)

def _write_split(

    samples: list[np.ndarray],
    metas: list[dict[str, Any]],
    output_prefix: Path,

) -> tuple[Path, Path, int]:
    """
    Write small split to disk as .npy + metadata helper.
    """
    npy_path = Path(str(output_prefix) + ".npy")
    meta_path = Path(str(output_prefix) + "_meta.npz")

    if samples:
        arr = np.stack(samples).astype(np.float16)
    else:
        arr = np.empty((0, SEQ_LEN_TOTAL, CROP_SIZE, CROP_SIZE), dtype=np.float16)

    np.save(npy_path, arr)
    _save_meta_npz(metas, meta_path)

    return npy_path, meta_path, int(arr.shape[0])


# Split Processing

def _split_months() -> dict[str, list[tuple[int, int]]]:
    """
    Month assignments for each split.
    """
    return {
        "train_event": [(y, m) for y in TRAIN_YEARS for m in WARM_SEASON_MONTHS],
        "val_event": [(VAL_YEAR, m) for m in WARM_SEASON_MONTHS],
        "val_all": [(VAL_YEAR, m) for m in WARM_SEASON_MONTHS],
        "test_all": [(TEST_YEAR, m) for m in WARM_SEASON_MONTHS],
    }


def _process_split(

    split_name: str,
    year_months: list[tuple[int, int]],
    *,
    split_mode: str,
    output_dir: Path,
    raw_dir: Path,
    skip_download: bool,
    activity_threshold: float,
    normalizer: dict[str, float] | None = None,
    max_days: int | None = None,
    max_frames_per_day: int | None = None,
    max_samples: int | None = None,
    reservoir_seed: int = REPRESENTATIVE_SEED,
    chunk_size: int = 512,
    optional: bool = False,
    
) -> dict[str, Any] | None:
    """
    Build one cache split with bounded memory.

    Event splits ("event"): 
        Samples flushed to temporary chunk files every `chunk_size` samples.  
        Peak RAM is O(chunk_size x sample_size).

    Representative splits ("representative"): 
        Online reservoir sampling keeps at most `max_samples` items in memory.
        Every candidate has equal probability of being in final set regardless of arrival order (Vitter's Algorithm R).
    """
    stats = {
        "total_candidates": 0, "active": 0, "inactive": 0,
        "nonfinite_rejected": 0, "rate_clipped_samples": 0,
        "rate_clipped_pixels": 0, "frames_read": 0, "archives_read": 0,
    }
    processed_months = []

    # Event mode: streaming chunk writes
    chunk_files: list[tuple[Path, Path, int]] = []
    temp_dir = output_dir / ".tmp_chunks" / split_name

    # Representative mode: online reservoir
    rng = np.random.RandomState(reservoir_seed) if split_mode == "representative" else None
    reservoir_samples: list[np.ndarray] = []
    reservoir_metas: list[dict[str, Any]] = []
    reservoir_seen = 0

    # Extraction buffer: flushed/folded into reservoir after each day
    buffer_samples: list[np.ndarray] = []
    buffer_metas: list[dict[str, Any]] = []

    if split_mode == "event":
        temp_dir.mkdir(parents=True, exist_ok=True)

    for year, month in year_months:
        try:
            archive = ensure_month_archive(year, month, raw_dir, skip_download=skip_download)
        except FileNotFoundError:
            if optional:
                continue
            raise

        processed_months.append((year, month))
        stats["archives_read"] += 1
        prev_tail: dict[datetime, np.ndarray] = {}

        for _, day_frames in iter_daily_frames(
            archive,
            max_days=max_days,
            max_frames_per_day=max_frames_per_day,
            strict_units=True,
        ):
            stats["frames_read"] += len(day_frames)

            # Merge tail of previous day for sequences across boundary
            merged = {**prev_tail, **day_frames}
            _extract_sequences(
                merged,
                split_mode=split_mode,
                activity_threshold=activity_threshold,
                samples=buffer_samples,
                metas=buffer_metas,
                stats=stats,
                normalizer=normalizer,
            )

            # Manage memory after each day
            if split_mode == "event":
                # Flush buffer to disk when it exceeds chunk_size
                if len(buffer_samples) >= chunk_size:
                    _flush_chunk(buffer_samples, buffer_metas, temp_dir, chunk_files)
                    buffer_samples = []
                    buffer_metas = []

            elif split_mode == "representative" and max_samples is not None:
                # Fold new samples into online reservoir
                for sample, meta in zip(buffer_samples, buffer_metas):
                    reservoir_seen += 1
                    if len(reservoir_samples) < max_samples:
                        reservoir_samples.append(sample)
                        reservoir_metas.append(meta)
                    else:
                        j = int(rng.randint(0, reservoir_seen))
                        if j < max_samples:
                            reservoir_samples[j] = sample
                            reservoir_metas[j] = meta
                buffer_samples = []
                buffer_metas = []

            # Keep tail frames for cross-day continuity
            day_times = sorted(day_frames)
            tail_n = min(len(day_times), SEQ_LEN_TOTAL - 1)
            prev_tail = {t: day_frames[t] for t in day_times[-tail_n:]}

    if optional and not processed_months:
        return None

    # Finalize
    output_prefix = output_dir / split_name

    if split_mode == "event":
        # Flush any remaining buffer
        if buffer_samples:
            _flush_chunk(buffer_samples, buffer_metas, temp_dir, chunk_files)
        npy_path, meta_path, n = _finalize_chunks(chunk_files, output_prefix)
        # Clean up temp directory
        try:
            temp_dir.rmdir()
        except OSError:
            pass

    elif split_mode == "representative":
        # Reservoir is already bounded → Write directly
        npy_path, meta_path, n = _write_split(
            reservoir_samples, reservoir_metas, output_prefix,
        )
        stats["reservoir_seen"] = reservoir_seen

    else:
        raise ValueError(f"Unknown split_mode: {split_mode!r}")

    stats["saved_samples"] = n
    return {
        "split_name": split_name,
        "npy_path": str(npy_path),
        "meta_path": str(meta_path),
        "samples": n,
        "stats": stats,
        "year_months": [f"{y:04d}-{m:02d}" for y, m in processed_months],
        "mode": split_mode,
    }


def build_strict_cache(
    *,
    output_dir: Path = PROCESSED_CACHE_DIR,
    raw_dir: Path = RAW_CACHE_DIR,
    skip_download: bool = False,
    test_mode: bool = False,
    test_year: int = 2021,
    test_month: int = 7,
    test_days: int = 1,
    test_frames_per_day: int = 36,
    test_sample_cap: int = 64,
    activity_threshold: float = ACTIVITY_THRESHOLD_MMH,
    reservoir_seed: int = REPRESENTATIVE_SEED,
    chunk_size: int = 512,
) -> dict[str, Any]:
    """
    Build processed cache with all splits.

    Use single month with bounded reads for fast testing in "test mode".
    """
    # Clean prior artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train_event", "val_event", "val_all", "test_all"):
        for suffix in (".npy", "_meta.npz"):
            (output_dir / f"{name}{suffix}").unlink(missing_ok=True)
    for f in ("stats.json", "metadata.json", "unit_audit.json"):
        (output_dir / f).unlink(missing_ok=True)
    temp = output_dir / ".tmp_chunks"
    if temp.exists():
        shutil.rmtree(temp)

    # Determine months per split
    if test_mode:
        month_list = [(test_year, test_month)]
        all_months = {k: month_list for k in ("train_event", "val_event", "val_all", "test_all")}
    else:
        all_months = _split_months()

    max_days_arg = test_days if test_mode else None
    max_fpd = test_frames_per_day if test_mode else None
    event_cap = test_sample_cap if test_mode else None
    rep_cap = test_sample_cap if test_mode else None

    results = {}

    # Train event (also compute normalization statistics)
    normalizer = {"sum": 0.0, "sq_sum": 0.0, "n": 0}
    train = _process_split(
        "train_event", all_months["train_event"],
        split_mode="event", output_dir=output_dir, raw_dir=raw_dir,
        skip_download=skip_download, activity_threshold=activity_threshold,
        normalizer=normalizer, max_days=max_days_arg,
        max_frames_per_day=max_fpd, max_samples=event_cap,
        chunk_size=chunk_size,
    )
    if train is None or train["samples"] == 0:
        raise RuntimeError("train_event produced zero samples.")
    results["train_event"] = train

    # Compute and write normalization stats
    n = normalizer["n"]
    mean = normalizer["sum"] / n
    variance = max(normalizer["sq_sum"] / n - mean * mean, 0.0)
    norm_stats = {
        "mean": float(mean),
        "std": float(math.sqrt(variance)),
        "transform": "log_rain_mmh",
        "unit": "mm/h",
        "source_split": "train_event",
        "n_values": int(n),
    }

    # Validate stats count matches stored array
    train_arr = np.load(train["npy_path"], mmap_mode="r")
    if int(train_arr.size) != norm_stats["n_values"]:
        raise RuntimeError(
            f"Normalization stats mismatch: n_values={norm_stats['n_values']}, "
            f"train_event.npy size={train_arr.size}"
        )
    del train_arr

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)

    # Validation and test splits
    for name, mode, max_s in [
        ("val_event", "event", event_cap),
        ("val_all", "representative", rep_cap or MAX_VAL_ALL_SAMPLES),
        ("test_all", "representative", rep_cap or MAX_TEST_ALL_SAMPLES),
    ]:
        result = _process_split(
            name, all_months[name],
            split_mode=mode, output_dir=output_dir, raw_dir=raw_dir,
            skip_download=skip_download, activity_threshold=activity_threshold,
            max_days=max_days_arg, max_frames_per_day=max_fpd,
            max_samples=max_s,
            reservoir_seed=reservoir_seed, chunk_size=chunk_size,
            optional=(not test_mode),
        )
        if result is not None:
            results[name] = result

    # Unit audit
    audit_months = (
        [(test_year, test_month)] if test_mode
        else [
            (TRAIN_YEARS[0], WARM_SEASON_MONTHS[0]),
            (VAL_YEAR, WARM_SEASON_MONTHS[0]),
            (TEST_YEAR, WARM_SEASON_MONTHS[0]),
        ]
    )
    audit = audit_radolan_units(
        audit_months, raw_dir=raw_dir, skip_download=skip_download,
    )
    audit_path = output_dir / "unit_audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)

    # Cache metadata
    metadata = {
        "created": datetime.now(timezone.utc).isoformat(),
        "data_version": DATA_VERSION,
        "strict_policy": "reject_any_nonfinite_input_or_target",
        "nan_fill_policy": "none",
        "grid_shape": list(RADOLAN_SHAPE),
        "patch_size": CROP_SIZE,
        "patch_stride": PATCH_STRIDE,
        "seq_len_in": SEQ_LEN_IN,
        "seq_len_out": SEQ_LEN_OUT,
        "temporal_stride": TEMPORAL_STRIDE,
        "activity_threshold_mmh": activity_threshold,
        "test_mode": test_mode,
        "split_independence": not test_mode,
        "normalization": norm_stats,
        "splits": {
            name: {
                "samples": r["samples"],
                "mode": r["mode"],
                "year_months": r["year_months"],
                "stats": r["stats"],
            }
            for name, r in results.items()
        },
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "output_dir": str(output_dir),
        "stats_path": str(stats_path),
        "metadata_path": str(meta_path),
        "unit_audit_path": str(audit_path),
        "splits": {
            name: {"samples": r["samples"], "stats": r["stats"]}
            for name, r in results.items()
        },
    }


# Dataset Loading

# Memory-mapped PyTorch dataset and dataloader creation.

class MemmapDataset:
    """
    Nowcasting dataset from strict cache split.

    Stored samples are log-rain arrays of shape (12, 128, 128).
    `__getitem__` normalizes and splits into (inputs, targets) tensors:
      - `inputs`:  (6, 128, 128), i.e. last 30min of observations.
      - `targets`: (6, 128, 128), i.e. next 30min to predict.
    """

    def __init__(
        self,
        split_name: str,
        *,
        cache_dir: Path = PROCESSED_CACHE_DIR,
        mean: float | None = None,
        std: float | None = None,
    ):
        self.split_name = split_name
        self.cache_dir = Path(cache_dir)
        npy_path = self.cache_dir / f"{split_name}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(
                f"Split not found: {npy_path}. Run scripts/build_cache.py first."
            )

        if mean is None or std is None:
            mean, std = load_normalization_stats(self.cache_dir / "stats.json")
        self.mean = float(mean)
        self.std = float(std)

        self.data = np.load(npy_path, mmap_mode="r")
        if self.data.ndim != 4 or self.data.shape[1] != SEQ_LEN_TOTAL:
            raise ValueError(f"Unexpected shape for {npy_path}: {self.data.shape}")

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, index: int):
        import torch

        sample = self.data[index].astype(np.float32)
        normalized = (sample - self.mean) / self.std
        inputs = torch.from_numpy(normalized[:SEQ_LEN_IN].copy())
        targets = torch.from_numpy(normalized[SEQ_LEN_IN:].copy())
        return inputs, targets

    def load_metadata(self) -> dict[str, np.ndarray]:
        """
        Load the per-sample metadata helper.
        """
        meta_path = self.cache_dir / f"{self.split_name}_meta.npz"
        return dict(np.load(meta_path, allow_pickle=False))


def get_loader(
    split_name: str,
    *,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 2,
    cache_dir: Path = PROCESSED_CACHE_DIR,
    mean: float | None = None,
    std: float | None = None,
):
    """
    Create a PyTorch DataLoader for one cache split.
    """
    import torch
    from torch.utils.data import DataLoader

    dataset = MemmapDataset(split_name, cache_dir=cache_dir, mean=mean, std=std)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=split_name.startswith("train"),
        persistent_workers=num_workers > 0,
    )


def get_dataloaders(
    *,
    batch_size: int = 16,
    num_workers: int = 2,
    splits: tuple[str, ...] = ("train_event", "val_event"),
    cache_dir: Path = PROCESSED_CACHE_DIR,
) -> dict[str, Any]:
    """
    Create DataLoaders for multiple splits with shared normalization stats.
    """
    mean, std = load_normalization_stats(Path(cache_dir) / "stats.json")
    return {
        name: get_loader(
            name,
            batch_size=batch_size,
            shuffle=name.startswith("train"),
            num_workers=num_workers,
            cache_dir=cache_dir,
            mean=mean,
            std=std,
        )
        for name in splits
    }