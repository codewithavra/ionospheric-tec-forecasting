"""
Central configuration for the project.
Edit paths and hyperparameters here; 
all other modules import from this file.
"""

from pathlib import Path
# 
# Data Paths
#
BASE_DIR = Path.cwd() / "TEC_data"
SORTED_DIR = BASE_DIR / "sortedDataSet"
OUTPUT_DIR = SORTED_DIR


#
# Data Set Constants
#

FOLDER_RANGE   = (1290, 1690)    # iisc1290_TECU … iisc1690_TECU (inclusive)
SATELLITES     = range(1, 33)    # G01 … G32
TOTAL_DAYS     = 41
MINUTES_PER_DAY = 1440           # 00:00 – 23:59 UTC
 
# Day split (1-based day numbers)
TRAIN_TARGET_START = 2
TRAIN_TARGET_END   = 30          # 29 pairs
VAL_TARGET_START   = 31
VAL_TARGET_END     = 41          # 11 pairs

#
# Other Constants
#

F_L1     = 1575.42e6   # Hz
F_L2     = 1227.60e6   # Hz
F_L5     = 1176.45e6   # Hz
K_IONO   = 40.308      # m³/s²
TECU     = 1e16        # electrons/m²
R_EARTH  = 6371.0      # km
H_ION    = 350.0       # km  (thin-shell ionosphere height)

#
# LSTM Hyperparameters
#

LSTM_UNITS_1   = 64
LSTM_UNITS_2   = 32
LSTM_EPOCHS    = 30
LSTM_BATCH     = 4
LSTM_OPTIMIZER = "adam"
LSTM_LOSS      = "mse"
LSTM_SEED      = 42

#
# Transformer Hyperparameters
#

TRANS_D_MODEL     = 128   # was 64  — more capacity for trend + spikes
TRANS_NUM_HEADS   = 8     # was 4   — finer attention granularity
TRANS_FF_DIM      = 512   # was 256 — wider FFN to match larger d_model
TRANS_NUM_LAYERS  = 4     # was 2   — deeper = learns both local & global patterns
TRANS_WINDOW_SIZE = 60    # was 120 — 1-hour windows instead of 2-hour (catches spikes better)
TRANS_DROPOUT     = 0.1   # was 0.0 — light regularisation (only 29 training samples)
TRANS_EPOCHS      = 150   # was 80  — more room to converge with warmup LR
TRANS_BATCH       = 4
TRANS_LR_INIT     = 1e-4  # was 5e-4 — lower peak LR; warmup handles the ramp
TRANS_LR_ALPHA    = 1e-6  # was 1e-5 — decay to a smaller floor
TRANS_WARMUP_EPOCHS = 10  # NEW — linear warmup before cosine decay kicks in
TRANS_SEED        = 42




