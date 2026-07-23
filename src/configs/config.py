"""
Central configuration for the project.
Edit paths and hyperparameters here; 
all other modules import from this file.
"""

from pathlib import Path

#
# Data Paths
#
BASE_DIR_1   = Path.cwd() / "DataSet" / "DataSet1"
BASE_DIR_2   = Path.cwd() / "DataSet" / "DataSet2"
SORTED_DIR_1 = BASE_DIR_1 / "sortedDataSet"
SORTED_DIR_2 = BASE_DIR_2 / "sortedDataSet"
OUTPUT_DIR_1 = SORTED_DIR_1
OUTPUT_DIR_2 = SORTED_DIR_2

#
# Project Root & Generated-Plot Output Directories
#
# config.py lives at src/configs/config.py, so two levels up is the
# project root (the folder that contains "src", "DataSet", etc.).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

GENERATED_PLOTS_DIR = PROJECT_ROOT / "generated_plots"
PLOTS_DIR_1         = GENERATED_PLOTS_DIR / "dataset1_plots"
PLOTS_DIR_2         = GENERATED_PLOTS_DIR / "dataset2_plots"

#
# Which dataset / target days this run covers.
#
DATASET_NO   = 2
TARGET_DAYS  = (33, 37, 40, 41)

#
# Known calendar dates for specific target days.
# Used to name plot files/labels as real dates instead of "dayN".
# Add more entries as more mappings become known — anything missing
# here just falls back to a "dayN" label automatically.
#
DATASET_DATES = {
    2: {
        33: "02_May_2024_G3_Storm",
        37: "06_May_2024_G2_Storm",
        40: "09_May_2024",
        41: "10_May_2024_G5_Storm",
    },
}


def get_date_label(dataset_id: int, day: int):
    """Return a filename-safe date label for (dataset_id, day), or None if unknown."""
    return DATASET_DATES.get(dataset_id, {}).get(day)


#
# Publication-consistent plot styling
#
PLOT_LABEL_FONTSIZE    = 30
PLOT_LABEL_FONTWEIGHT  = "normal"
PLOT_TICK_FONTSIZE     = 28
PLOT_TICK_FONTWEIGHT   = "normal"
PLOT_LEGEND_FONTSIZE   = 25
PLOT_LEGEND_FONTWEIGHT = "normal"
PLOT_GRID_ALPHA        = 0.35
PLOT_GRID_LINEWIDTH    = 0.6
PLOT_DPI               = 300

#
# Plot Figure Sizes  (single source of truth — change a size here and
# every plot that uses it resizes, no need to touch individual plot files)
#
PLOT_FIGSIZE_SINGLE  = (9, 6)    # full-day series is wide, not tall
PLOT_FIGSIZE_STACKED = (9, 10)   # two-panel stacked plots: combined L1+L5 delay
PLOT_FIGSIZE_LOSS    = (9, 5)    # training/validation loss curves
PLOT_FIGSIZE_DIURNAL = (11, 5)   # diurnal hourly profile

#
# Legend placement (shared by every full-day time-series plot)
#
PLOT_LEGEND_ANCHOR       = (0.5, 1.02)     # above the axes, centered
PLOT_LEGEND_LOC          = "lower center"
PLOT_LEGEND_ANCHOR_RIGHT = (0.5, -0.25)    # below the x-axis label, horizontally centered
PLOT_LEGEND_LOC_RIGHT    = "upper center"

#
# Colors — actual vs. per-model prediction traces
#
COLOR_ACTUAL      = "steelblue"
COLOR_LSTM_PRED    = "tomato"
COLOR_TRANS_PRED   = "darkorange"

# Combined L1 + L5 delay plot colors
COLOR_L1_ACTUAL = "steelblue"
COLOR_L1_PRED   = "tomato"
COLOR_L5_ACTUAL = "navy"
COLOR_L5_PRED   = "firebrick"

#
# Time-of-Day X-Axis Ticks  (shared by every full-day plot)
# Ticks are placed every TIME_TICK_INTERVAL_HOURS, with the final tick
# forced to the real last minute of the day (23:59) instead of 24:00.
#
TIME_TICK_INTERVAL_HOURS = 12


def get_time_ticks(minutes_per_day: int = None, interval_hours: int = None):
    """Return (tick_positions, tick_labels) for a full-day x-axis in minutes.

    Ticks run every `interval_hours` starting at 00:00, and the final tick
    is always the last real minute of the day, labelled "23:59" (never
    "24:00", which doesn't correspond to an actual data point).
    """
    if minutes_per_day is None:
        minutes_per_day = MINUTES_PER_DAY
    if interval_hours is None:
        interval_hours = TIME_TICK_INTERVAL_HOURS

    step = interval_hours * 60
    positions = list(range(0, minutes_per_day, step))
    labels = [f"{p // 60:02d}:00" for p in positions]

    positions.append(minutes_per_day - 1)
    labels.append("23:59")

    return positions, labels


#
# Data Set Constants
#
FOLDER_RANGE_1   = (1290, 1690)   # iisc1290_TECU … iisc1690_TECU (inclusive)
FOLDER_RANGE_2   = (910,  1310)   # iisc0910_TECU … iisc1310_TECU (inclusive)
SATELLITES       = range(1, 33)   # G01 … G32
TOTAL_DAYS       = 41
MINUTES_PER_DAY  = 1440           # 00:00 – 23:59 UTC

# Day split (1-based day numbers)
TRAIN_TARGET_START = 1
TRAIN_TARGET_END   = 30           # 29 training pairs
VAL_TARGET_START   = 31
VAL_TARGET_END     = 41           # 11 validation pairs

#
# Physical Constants
#
F_L1     = 1575.42e6   # Hz
F_L2     = 1227.60e6   # Hz
F_L5     = 1176.45e6   # Hz
K_IONO   = 40.308      # m³/s²
TECU     = 1e16        # electrons/m²
R_EARTH  = 6371.0      # km
H_ION    = 350.0       # km

#
# LSTM Hyperparameters  (unchanged)
#
LSTM_UNITS_1   = 64
LSTM_UNITS_2   = 32
LSTM_EPOCHS    = 50
LSTM_BATCH     = 4
LSTM_OPTIMIZER = "adam"
LSTM_LOSS      = "mse"
LSTM_SEED      = 42

#
# Transformer Hyperparameters  (lightened for CPU / low-VRAM GPU)
# 
TRANS_D_MODEL       = 64     # was 128
TRANS_NUM_HEADS     = 4      # was 8
TRANS_FF_DIM        = 128    # was 512
TRANS_NUM_LAYERS    = 2      # was 4
TRANS_DROPOUT       = 0.1
TRANS_EPOCHS        = 50
TRANS_BATCH         = 4
TRANS_LR_INIT       = 1e-4
TRANS_LR_ALPHA      = 1e-6
TRANS_WARMUP_EPOCHS = 10
TRANS_SEED          = 42

#
# Event Threshold
#
EVENT_THRESHOLD_PERCENTILE  = 65.0
DELAY_EVENT_THRESHOLDS_M = (3.0, 8.0, 15.0)