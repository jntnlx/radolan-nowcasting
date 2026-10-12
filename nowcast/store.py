"""
Lossless frame store.

One zarr group per warm season, `yw_<year>.zarr`, on a complete 5-minute time
axis (missing frames are all-NODATA and flagged in `present`):

    depth       (T, 1024, 832) uint16   5-minute depth in 0.01 mm, NODATA outside coverage
    present     (T,)           bool     frame exists in the DWD archive
    n_radars    (T,)           uint8    radars contributing to the composite
    tile_rain   (T, 16, 13)    float16  mean rain rate (mm/h) over valid pixels of 64 x 64 km tiles
    tile_valid  (T, 16, 13)    float16  fraction of valid pixels per tile

Frames are stored whole (one compressed chunk each, ~45 kB) and grouped into
one shard file per day. Whole-frame chunks are what both training (large
crops) and evaluation (full domain) read, and day shards let parallel
workers write disjoint files.
"""

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec

from .config import DOMAIN_SHAPE, RAW_DIR, SEASON_MONTHS, STEPS_PER_HOUR, STORE_DIR, season_length, time_index
from .radklim import NODATA, depth_to_rate, iter_month

TILE = 64
TILES = (DOMAIN_SHAPE[0] // TILE, DOMAIN_SHAPE[1] // TILE)
STEPS_PER_DAY = 24 * STEPS_PER_HOUR


def store_path(year: int, store_dir: Path = STORE_DIR) -> Path:
    return store_dir / f"yw_{year}.zarr"


def tile_statistics(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-tile mean rain rate over valid pixels and valid fraction of one frame."""
    rate = depth_to_rate(depth).reshape(TILES[0], TILE, TILES[1], TILE)
    valid = np.isfinite(rate)
    n_valid = valid.sum(axis=(1, 3))
    total = np.where(valid, rate, 0).sum(axis=(1, 3))
    mean = np.divide(total, n_valid, out=np.zeros_like(total), where=n_valid > 0)
    return mean, n_valid / TILE**2


# Building

def create_season(year: int, store_dir: Path = STORE_DIR) -> zarr.Group:
    T, (H, W) = season_length(year), DOMAIN_SHAPE
    root = zarr.open_group(store_path(year, store_dir), mode="w")
    root.create_array(
        "depth", shape=(T, H, W), chunks=(1, H, W), shards=(STEPS_PER_DAY, H, W),
        dtype="uint16", fill_value=int(NODATA),
        compressors=BloscCodec(cname="zstd", clevel=5, shuffle="shuffle"),
    )
    root.create_array("present", shape=(T,), dtype="bool", fill_value=False)
    root.create_array("n_radars", shape=(T,), dtype="uint8", fill_value=0)
    root.create_array("tile_rain", shape=(T, *TILES), chunks=(STEPS_PER_DAY, *TILES), dtype="float16", fill_value=0)
    root.create_array("tile_valid", shape=(T, *TILES), chunks=(STEPS_PER_DAY, *TILES), dtype="float16", fill_value=0)
    root.attrs.update({
        "product": "RADKLIM-YW 2017.002 (DWD)",
        "depth_unit": "0.01 mm per 5 minutes, interval ending at the timestamp (UTC)",
        "nodata": int(NODATA),
        "season_start": f"{year}-{SEASON_MONTHS[0]:02d}-01T00:00",
        "step_minutes": 5,
        "tile_km": TILE,
        "months_complete": [],
    })
    return root


def _fill_month(year: int, month: int, store_dir: Path, raw_dir: Path) -> dict:
    """Decode one monthly archive and write its days into the season store (worker)."""
    depth = zarr.open_array(store_path(year, store_dir) / "depth", mode="r+")
    indices, n_radars, rain, valid = [], [], [], []
    skipped = []

    day, block = None, None

    def flush():
        if day is not None:
            depth[day * STEPS_PER_DAY:(day + 1) * STEPS_PER_DAY] = block

    for timestamp, frame, radars in iter_month(year, month, raw_dir):
        if timestamp.month != month:
            skipped.append(timestamp.isoformat())  # frames that belong to a neighbouring archive
            continue
        _, index = time_index(timestamp)
        if index // STEPS_PER_DAY != day:
            flush()
            day = index // STEPS_PER_DAY
            block = np.full((STEPS_PER_DAY, *DOMAIN_SHAPE), NODATA, dtype=np.uint16)
        block[index % STEPS_PER_DAY] = frame
        tile_rain, tile_valid = tile_statistics(frame)
        indices.append(index)
        n_radars.append(radars)
        rain.append(tile_rain)
        valid.append(tile_valid)
    flush()

    return {
        "year": year, "month": month, "skipped": skipped,
        "indices": np.array(indices), "n_radars": np.array(n_radars),
        "tile_rain": np.array(rain), "tile_valid": np.array(valid),
    }


def build(years, *, workers: int = 12, store_dir: Path = STORE_DIR, raw_dir: Path = RAW_DIR) -> None:
    """Build (or resume) the season stores for `years` from the raw archives."""
    store_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for year in years:
        path = store_path(year, store_dir)
        root = zarr.open_group(path, mode="r+") if path.exists() else create_season(year, store_dir)
        done = set(root.attrs["months_complete"])
        tasks += [(year, m) for m in SEASON_MONTHS if m not in done]

    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(_fill_month, y, m, store_dir, raw_dir) for y, m in tasks]
        for future in futures:
            r = future.result()
            root = zarr.open_group(store_path(r["year"], store_dir), mode="r+")
            i = r["indices"]
            if len(np.unique(i)) != len(i):
                raise ValueError(f"{r['year']}-{r['month']:02d}: duplicate timestamps")
            root["present"][i] = True
            root["n_radars"][i] = r["n_radars"]
            root["tile_rain"][i] = r["tile_rain"]
            root["tile_valid"][i] = r["tile_valid"]
            root.attrs["months_complete"] = sorted(set(root.attrs["months_complete"]) | {r["month"]})
            print(f"{r['year']}-{r['month']:02d}: {len(i)} frames, {len(r['skipped'])} skipped", flush=True)


# Reading

class Season:
    """Read access to one season store. Cheap metadata is held in memory."""

    def __init__(self, year: int, store_dir: Path = STORE_DIR):
        self.year = year
        self.path = store_path(year, store_dir)
        root = zarr.open_group(self.path, mode="r")
        self.present = root["present"][:]
        self.n_radars = root["n_radars"][:]
        self.tile_rain = root["tile_rain"][:]    # float16 as stored; cast slices before arithmetic
        self.tile_valid = root["tile_valid"][:]
        self._depth = None  # opened lazily so the object can be sent to DataLoader workers

    def __len__(self) -> int:
        return len(self.present)

    @property
    def depth(self) -> zarr.Array:
        if self._depth is None:
            self._depth = zarr.open_array(self.path / "depth", mode="r")
        return self._depth

    def __getstate__(self):
        return {**self.__dict__, "_depth": None}

    def frames(self, start: int, stop: int, rows: slice = slice(None), cols: slice = slice(None)) -> np.ndarray:
        """Stored depth integers for time steps [start, stop) on a spatial window."""
        return self.depth[start:stop, rows, cols]

    def time(self, index: int) -> datetime:
        return datetime(self.year, SEASON_MONTHS[0], 1) + index * timedelta(minutes=5)

    def complete_windows(self, before: int, after: int) -> np.ndarray:
        """Indices t0 such that frames t0-before+1 .. t0+after are all present."""
        n = before + after
        run = np.convolve(self.present.astype(np.int32), np.ones(n, dtype=np.int32), mode="valid")
        starts = np.flatnonzero(run == n)
        return starts + before - 1
