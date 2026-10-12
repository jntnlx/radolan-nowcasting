"""
Paths, domain geometry, time axis and data splits.

Everything that defines *what problem is being solved* lives here; model and
training hyperparameters live with the code that uses them.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

# Paths

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"

DATA_ROOT = Path(os.environ.get("NOWCAST_DATA", Path.home() / ".cache" / "radolan_nowcasting"))
RAW_DIR = DATA_ROOT / "raw_yw"      # monthly DWD archives, as downloaded
STORE_DIR = DATA_ROOT / "store_v3"  # lossless frame store built from RAW_DIR

# Source product: RADKLIM-YW 2017.002, the climate-reprocessed 5-minute
# radar composite of DWD (1 km polar-stereographic grid, 1100 x 900).

DWD_URL = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/5_minutes/radolan/reproc/2017_002/bin"
RADOLAN_SHAPE = (1100, 900)

# Model domain: the smallest 64-divisible window of the RADOLAN grid that
# contains every pixel that ever carries data (rows = south -> north).
# 1 pixel = 1 km.

DOMAIN_ORIGIN = (38, 35)    # (row, col) of the domain's south-west corner
DOMAIN_SHAPE = (1024, 832)

# Time axis. Timestamps mark the END of each 5-minute accumulation interval (UTC).

STEP = timedelta(minutes=5)
STEPS_PER_HOUR = 12
SEASON_MONTHS = (4, 5, 6, 7, 8, 9, 10)  # warm season: the convective regime

# Year-blocked splits. Later years are held out so that evaluation never
# benefits from information that lies in the future of the training data.

TRAIN_YEARS = tuple(range(2006, 2021))  # 15 seasons
VAL_YEARS = (2021, 2022)                # model selection; includes the July 2021 floods
TEST_YEARS = (2023, 2024)               # locked until the method is frozen


def season_start(year: int) -> datetime:
    return datetime(year, SEASON_MONTHS[0], 1)


def season_length(year: int) -> int:
    """Number of 5-minute steps in a season (the time axis of a yearly store)."""
    end = datetime(year, SEASON_MONTHS[-1] + 1, 1)
    return int((end - season_start(year)) / STEP)


def time_index(t: datetime) -> tuple[int, int]:
    """Map a timestamp to (year, index along that season's time axis)."""
    offset = t - season_start(t.year)
    index, remainder = divmod(offset, STEP)
    if remainder or not 0 <= index < season_length(t.year):
        raise ValueError(f"{t} is not on the 5-minute grid of the {t.year} season")
    return t.year, int(index)


def index_time(year: int, index: int) -> datetime:
    return season_start(year) + index * STEP
