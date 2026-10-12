"""
RADKLIM-YW archive access and decoding.

DWD publishes the reprocessed 5-minute composite as monthly tar archives of
daily tar.gz archives of single binary RADOLAN frames. Each frame holds the
precipitation depth accumulated over 5 minutes, in multiples of 0.01 mm.

The frame store keeps those integers losslessly: depth = DEPTH_QUANTUM * value,
with NODATA marking pixels without radar coverage. `depth_to_rate` is the
single place where stored integers become physical rain rates.
"""

import io
import tarfile
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from .config import DOMAIN_ORIGIN, DOMAIN_SHAPE, DWD_URL, RADOLAN_SHAPE, RAW_DIR, STEP

DEPTH_QUANTUM_MM = 0.01  # resolution of the YW product
NODATA = np.uint16(65535)


def depth_to_rate(depth: np.ndarray) -> np.ndarray:
    """Stored 5-minute depth integers -> rain rate in mm/h (NaN where no data)."""
    rate = depth.astype(np.float32) * np.float32(DEPTH_QUANTUM_MM * 12)
    rate[depth == NODATA] = np.nan
    return rate


# Archives

def archive_path(year: int, month: int, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"{year:04d}" / f"YW2017.002_{year:04d}{month:02d}.tar"


def download_archive(year: int, month: int, raw_dir: Path = RAW_DIR) -> Path:
    """Fetch one monthly archive from DWD open data unless it is already present."""
    path = archive_path(year, month, raw_dir)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{DWD_URL}/{year:04d}/{path.name}"
    partial = path.with_suffix(".part")
    for attempt in range(3):
        try:
            urlretrieve(url, partial)
            with tarfile.open(partial, "r:") as tar:
                tar.getmembers()  # integrity check before the file is accepted
            partial.rename(path)
            return path
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(10 * (attempt + 1))
    raise AssertionError("unreachable")


# Frame decoding

def decode_frame(raw: bytes) -> tuple[datetime, np.ndarray, int]:
    """
    Decode one binary YW frame.

    Returns (timestamp, depth integers on the model domain, number of radars).
    Raises if the metadata contradict the assumed physical units.
    """
    import wradlib

    data, meta = wradlib.io.read_radolan_composite(io.BytesIO(raw))
    if meta["producttype"] != "YW":
        raise ValueError(f"expected product YW, got {meta['producttype']!r}")
    if meta["intervalseconds"] != 300 or meta["intervalunit"] != 0:
        raise ValueError(f"expected 5-minute depth, got {meta['intervalseconds']} s / unit {meta['intervalunit']}")
    if not np.isclose(meta["precision"], DEPTH_QUANTUM_MM):
        raise ValueError(f"expected precision {DEPTH_QUANTUM_MM} mm, got {meta['precision']}")
    if data.shape != RADOLAN_SHAPE:
        raise ValueError(f"expected grid {RADOLAN_SHAPE}, got {data.shape}")

    valid = data >= 0  # wradlib marks missing data with -9999
    depth = np.full(RADOLAN_SHAPE, NODATA, dtype=np.uint16)
    depth[valid] = np.rint(data[valid] / DEPTH_QUANTUM_MM).astype(np.uint16)

    r0, c0 = DOMAIN_ORIGIN
    h, w = DOMAIN_SHAPE
    outside = valid.sum() - valid[r0:r0 + h, c0:c0 + w].sum()
    if outside:
        raise ValueError(f"{outside} valid pixels fall outside the model domain")
    return meta["datetime"], depth[r0:r0 + h, c0:c0 + w], len(meta["radarlocations"])


def iter_month(year: int, month: int, raw_dir: Path = RAW_DIR):
    """Yield (timestamp, depth, n_radars) for every frame of one monthly archive."""
    with tarfile.open(archive_path(year, month, raw_dir), "r:") as outer:
        days = sorted((m for m in outer.getmembers() if m.name.endswith(".tar.gz")), key=lambda m: m.name)
        for day in days:
            with tarfile.open(fileobj=outer.extractfile(day), mode="r|gz") as inner:
                for member in inner:
                    if member.isfile():
                        timestamp, depth, n_radars = decode_frame(inner.extractfile(member).read())
                        if (timestamp - datetime(timestamp.year, 1, 1)) % STEP:
                            raise ValueError(f"{member.name}: timestamp {timestamp} is off the 5-minute grid")
                        yield timestamp, depth, n_radars
