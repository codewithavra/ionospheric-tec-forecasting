# """
# Loads sorted CSVs into a (days × minutes) numpy matrix, 
# builds supervised input/target pairs, and 
# normalises using training stats only.
# """

# #
# # Imports
# #

# import csv
# from pathlib import Path
# from datetime import datetime
 
# import numpy as np
 
# from src.configs.config import (
#     SORTED_DIR_1, SORTED_DIR_2, TOTAL_DAYS, MINUTES_PER_DAY,
#     TRAIN_TARGET_START, TRAIN_TARGET_END,
#     VAL_TARGET_START,   VAL_TARGET_END,
# )

# # Minute Label Helper

# MINUTE_LABELS = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(60)]
# MINUTE_INDEX  = {t: i for i, t in enumerate(MINUTE_LABELS)}
 
 
# def load_day_vector(csv_path: Path) -> np.ndarray:
#     """
#     Load one sorted CSV → float32 array of length MINUTES_PER_DAY.
#     Gaps are filled with linear interpolation.
#     """
#     vec = np.full(MINUTES_PER_DAY, np.nan, dtype=np.float32)
#     with csv_path.open("r", newline="", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             t = (row.get("Time(UTC)") or "").strip()
#             v = (row.get("TEC(TECU)")  or "").strip()
#             if not t or not v:
#                 continue
#             try:
#                 dt  = datetime.strptime(t, "%d-%m-%y %H:%M")
#                 tec = float(v)
#             except ValueError:
#                 continue
#             key = dt.strftime("%H:%M")
#             idx = MINUTE_INDEX.get(key)
#             if idx is not None:
#                 vec[idx] = tec
 
#     if np.isnan(vec).any():
#         x     = np.arange(len(vec))
#         valid = ~np.isnan(vec)
#         if valid.sum() == 0:
#             raise ValueError(f"No usable TEC data in: {csv_path.name}")
#         vec = np.interp(x, x[valid], vec[valid]).astype(np.float32)
 
#     return vec
 
 
# def build_daily_matrix(data_set_no: int = 1) -> tuple[np.ndarray, list[Path]]:
#     """
#     Load the first TOTAL_DAYS sorted CSVs into a matrix of shape
#     (TOTAL_DAYS, MINUTES_PER_DAY).  Returns (matrix, list_of_paths).
#     """
#     if data_set_no == 1:
#         sorted_dir = SORTED_DIR_1
#     elif data_set_no == 2:
#         sorted_dir = SORTED_DIR_2
#     all_csv = sorted(sorted_dir.glob("*_sorted.csv"))
#     if len(all_csv) < TOTAL_DAYS:
#         raise ValueError(
#             f"Need at least {TOTAL_DAYS} sorted CSV files, found {len(all_csv)}"
#         )
#     daily_files  = all_csv[:TOTAL_DAYS]
#     daily_matrix = np.stack([load_day_vector(p) for p in daily_files], axis=0)
#     return daily_matrix, daily_files
 
 
# def build_splits(
#     daily_matrix: np.ndarray,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Build (X_train, y_train, X_val, y_val) raw (un-normalised) arrays.
#     Each row is a 1440-minute day vector.
#     """
#     X_all = daily_matrix[:-1]            # days 1..40
#     y_all = daily_matrix[1:]             # days 2..41
#     target_day_num = np.arange(2, TOTAL_DAYS + 1)
 
#     train_mask = (
#         (target_day_num >= TRAIN_TARGET_START) &
#         (target_day_num <= TRAIN_TARGET_END)
#     )
#     val_mask = (
#         (target_day_num >= VAL_TARGET_START) &
#         (target_day_num <= VAL_TARGET_END)
#     )
 
#     return (
#         X_all[train_mask], y_all[train_mask],
#         X_all[val_mask],   y_all[val_mask],
#     )
 
 
# def normalise(
#     X_train_raw: np.ndarray,
#     y_train_raw: np.ndarray,
#     X_val_raw: np.ndarray,
#     y_val_raw: np.ndarray,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
#     """
#     Z-score normalise using *training* statistics only.
#     Returns (X_train, y_train, X_val, y_val, stats_dict).
#     All arrays gain a trailing channel dim: shape (N, 1440, 1).
#     """
#     x_mean = float(X_train_raw.mean())
#     x_std  = float(X_train_raw.std()) + 1e-8
#     y_mean = float(y_train_raw.mean())
#     y_std  = float(y_train_raw.std()) + 1e-8
 
#     X_train = ((X_train_raw - x_mean) / x_std)[..., np.newaxis]
#     y_train = ((y_train_raw - y_mean) / y_std)[..., np.newaxis]
#     X_val   = ((X_val_raw   - x_mean) / x_std)[..., np.newaxis]
#     y_val   = ((y_val_raw   - y_mean) / y_std)[..., np.newaxis]
 
#     stats = dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
#     return X_train, y_train, X_val, y_val, stats
 
 
# def prepare_dataset(
    
#     sorted_dir: Path = SORTED_DIR,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray, list[Path]]:
#     """
#     End-to-end helper: load → split → normalise.
 
#     Returns
#     -------
#     X_train, y_train, X_val, y_val   — normalised, shape (N, 1440, 1)
#     stats                            — dict with x_mean, x_std, y_mean, y_std
#     daily_matrix                     — raw (41, 1440) float32
#     daily_files                      — list of Path objects (length 41)
#     """
#     daily_matrix, daily_files = build_daily_matrix(sorted_dir)
    
#     X_train_r, y_train_r, X_val_r, y_val_r = build_splits(daily_matrix)
    
#     X_train, y_train, X_val, y_val, stats = normalise(
#         X_train_r, y_train_r, X_val_r, y_val_r
#     )
#     return X_train, y_train, X_val, y_val, stats, daily_matrix, daily_files
"""
Loads sorted CSVs into a (days × minutes) numpy matrix, 
builds supervised input/target pairs, and 
normalises using training stats only.
"""

#
# Imports
#

import csv
from pathlib import Path
from datetime import datetime
 
import numpy as np
 
from src.configs.config import (
    SORTED_DIR_1, SORTED_DIR_2, TOTAL_DAYS, MINUTES_PER_DAY,
    TRAIN_TARGET_START, TRAIN_TARGET_END,
    VAL_TARGET_START,   VAL_TARGET_END,
)

# Minute Label Helper

MINUTE_LABELS = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(60)]
MINUTE_INDEX  = {t: i for i, t in enumerate(MINUTE_LABELS)}
 
 
def load_day_vector(csv_path: Path) -> np.ndarray:
    """
    Load one sorted CSV → float32 array of length MINUTES_PER_DAY.
    Gaps are filled with linear interpolation.
    """
    vec = np.full(MINUTES_PER_DAY, np.nan, dtype=np.float32)
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (row.get("Time(UTC)") or "").strip()
            v = (row.get("TEC(TECU)")  or "").strip()
            if not t or not v:
                continue
            try:
                dt  = datetime.strptime(t, "%d-%m-%y %H:%M")
                tec = float(v)
            except ValueError:
                continue
            key = dt.strftime("%H:%M")
            idx = MINUTE_INDEX.get(key)
            if idx is not None:
                vec[idx] = tec
 
    if np.isnan(vec).any():
        x     = np.arange(len(vec))
        valid = ~np.isnan(vec)
        if valid.sum() == 0:
            raise ValueError(f"No usable TEC data in: {csv_path.name}")
        vec = np.interp(x, x[valid], vec[valid]).astype(np.float32)
 
    return vec
 
 
def build_daily_matrix(data_set_no: int = 1) -> tuple[np.ndarray, list[Path]]:
    """
    Load the first TOTAL_DAYS sorted CSVs into a matrix of shape
    (TOTAL_DAYS, MINUTES_PER_DAY).  Returns (matrix, list_of_paths).
    """
    if data_set_no == 1:
        sorted_dir = SORTED_DIR_1
    elif data_set_no == 2:
        sorted_dir = SORTED_DIR_2
    else:
        raise ValueError(f"data_set_no must be 1 or 2, got {data_set_no}")

    all_csv = sorted(sorted_dir.glob("*_sorted.csv"))
    if len(all_csv) < TOTAL_DAYS:
        raise ValueError(
            f"Need at least {TOTAL_DAYS} sorted CSV files, found {len(all_csv)}"
        )
    daily_files  = all_csv[:TOTAL_DAYS]
    daily_matrix = np.stack([load_day_vector(p) for p in daily_files], axis=0)
    return daily_matrix, daily_files
 
 
def build_splits(
    daily_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X_train, y_train, X_val, y_val) raw (un-normalised) arrays.
    Each row is a 1440-minute day vector.
    """
    X_all = daily_matrix[:-1]            # days 1..40
    y_all = daily_matrix[1:]             # days 2..41
    target_day_num = np.arange(2, TOTAL_DAYS + 1)
 
    train_mask = (
        (target_day_num >= TRAIN_TARGET_START) &
        (target_day_num <= TRAIN_TARGET_END)
    )
    val_mask = (
        (target_day_num >= VAL_TARGET_START) &
        (target_day_num <= VAL_TARGET_END)
    )
 
    return (
        X_all[train_mask], y_all[train_mask],
        X_all[val_mask],   y_all[val_mask],
    )
 
 
def normalise(
    X_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    X_val_raw: np.ndarray,
    y_val_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Z-score normalise using *training* statistics only.
    Returns (X_train, y_train, X_val, y_val, stats_dict).
    All arrays gain a trailing channel dim: shape (N, 1440, 1).
    """
    x_mean = float(X_train_raw.mean())
    x_std  = float(X_train_raw.std()) + 1e-8
    y_mean = float(y_train_raw.mean())
    y_std  = float(y_train_raw.std()) + 1e-8
 
    X_train = ((X_train_raw - x_mean) / x_std)[..., np.newaxis]
    y_train = ((y_train_raw - y_mean) / y_std)[..., np.newaxis]
    X_val   = ((X_val_raw   - x_mean) / x_std)[..., np.newaxis]
    y_val   = ((y_val_raw   - y_mean) / y_std)[..., np.newaxis]
 
    stats = dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    return X_train, y_train, X_val, y_val, stats
 
 
def prepare_dataset(
    data_set_no: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray, list[Path]]:
    """
    End-to-end helper: load → split → normalise.
 
    Returns
    -------
    X_train, y_train, X_val, y_val   — normalised, shape (N, 1440, 1)
    stats                            — dict with x_mean, x_std, y_mean, y_std
    daily_matrix                     — raw (41, 1440) float32
    daily_files                      — list of Path objects (length 41)
    """
    daily_matrix, daily_files = build_daily_matrix(data_set_no)
    
    X_train_r, y_train_r, X_val_r, y_val_r = build_splits(daily_matrix)
    
    X_train, y_train, X_val, y_val, stats = normalise(
        X_train_r, y_train_r, X_val_r, y_val_r
    )
    return X_train, y_train, X_val, y_val, stats, daily_matrix, daily_files