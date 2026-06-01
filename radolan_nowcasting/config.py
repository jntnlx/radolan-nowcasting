"""
Configuration constants.

Paths, physical units, split definitions and training defaults.
"""

from pathlib import Path


# Paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
PLOTS_DIR = PROJECT_ROOT / "plots"

CACHE_DIR = Path.home() / ".cache" / "radolan_nowcasting"
RAW_CACHE_DIR = CACHE_DIR / "raw_yw"
DATA_VERSION = "radolan_yw_mmh_v2_strict_128"
PROCESSED_CACHE_DIR = CACHE_DIR / DATA_VERSION


# DWD RADOLAN-YW source

DWD_BASE_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/"
    "grids_germany/5_minutes/radolan/reproc/2017_002/bin"
)


# Grid and sequence geometry

RADOLAN_SHAPE = (1100, 900)   # RADOLAN grid (rows, cols)
CROP_SIZE = 128               # spatial patch size
PATCH_STRIDE = 128            # non-overlapping crops
TEMPORAL_STRIDE = 3           # frames to skip between sequences

SEQ_LEN_IN = 6                # 30 min of input  (6*5 min)
SEQ_LEN_OUT = 6               # 30 min forecast  (6*5 min)
SEQ_LEN_TOTAL = SEQ_LEN_IN + SEQ_LEN_OUT

FRAME_INTERVAL_SEC = 300      # 5 minute cadence
GAP_TOLERANCE_SEC = 60        # allowed deviation from exact 5min spacing


# Unit conversion

# Reprocessed RADOLAN-YW data with intervalunit=0 reports 5 minute
# accumulated precipitation depth. Convert to rate in mm/h:
#     rate_mmh = (depth_mm_per_5min)*12
# After conversion: Model works in physical units.

RADOLAN_MM5_TO_MMH = 12.0
MIN_RAIN_RATE_MMH = 0.01     # log transform floor: avoids log(0)
MAX_RAIN_RATE_MMH = 300.0    # ceiling of physical clipping


# Warm-season months only splits
# Reasoning: Convective precipitation dominates in tehse months

WARM_SEASON_MONTHS = (4, 5, 6, 7, 8, 9, 10)  # seasonal warm convective regime
TRAIN_YEARS = tuple(range(2017, 2023))   # total of 6 years of data 
VAL_YEAR = 2023
TEST_YEAR = 2024

# Event filtering: sequence is considered "active" if assigned target rain-rate 
# field percentile exceeds threshold.
ACTIVITY_THRESHOLD_MMH = 0.5  # threshold
ACTIVITY_PERCENTILE = 90  # 90th percentile

# Capped representative splits
MAX_VAL_ALL_SAMPLES = 5000
MAX_TEST_ALL_SAMPLES = 5000
REPRESENTATIVE_SEED = 42  # seed: 42


# Verification thresholds based on standard in literature

RAIN_THRESHOLD_MMH = 0.1     # distinguish "rainy" vs "dry" pixels
THRESHOLDS_MMH = (0.1, 1.0, 5.0, 10.0, 20.0)
FSS_THRESHOLDS_MMH = (1.0, 5.0, 10.0)
FSS_NEIGHBORHOODS = (9, 17)
HEAVY_RAIN_THRESHOLDS_MMH = (1.0, 5.0, 10.0, 20.0)


# Training defaults 
# Local hardware reference: RTX 4070 Super 12 GB, AMD 7800X3D, 32 GB DDR5

BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-2
GRADIENT_CLIP = 0.5
MAX_STEPS = 2000
TIME_BUDGET_SEC = 1800        # currently for testing: 30min limit
VAL_INTERVAL_STEPS = 250
SEED = 42
