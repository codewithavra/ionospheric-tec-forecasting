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