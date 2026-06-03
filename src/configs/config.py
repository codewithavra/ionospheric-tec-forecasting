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

TRANS_D_MODEL     = 64
TRANS_NUM_HEADS   = 4
TRANS_FF_DIM      = 256
TRANS_NUM_LAYERS  = 2
TRANS_WINDOW_SIZE = 120
TRANS_DROPOUT     = 0.0
TRANS_EPOCHS      = 80
TRANS_BATCH       = 4
TRANS_LR_INIT     = 5e-4
TRANS_LR_ALPHA    = 1e-5
TRANS_SEED        = 42



