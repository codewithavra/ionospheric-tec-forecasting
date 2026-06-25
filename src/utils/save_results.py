"""
Saves the final prediction + ionospheric delay table to a CSV file.
"""

import csv
from pathlib import Path

import numpy as np

from src.configs.config import OUTPUT_DIR_1, MINUTES_PER_DAY
from src.preprocessing.dataset import MINUTE_LABELS
from src.utils.iono_delay import tec_to_iono_delay


def save_delay_csv(
    actual     : np.ndarray,
    lstm_pred  : np.ndarray,
    trans_pred : np.ndarray,
    output_dir : Path = OUTPUT_DIR_1,
    filename   : str  = "day41_ionospheric_delay.csv",
) -> None:
    """
    Write a CSV with columns:
        Time(UTC), TEC_actual, Delay_actual_L1,
        TEC_LSTM, Delay_LSTM_L1,
        TEC_Transformer, Delay_Transformer_L1

    Parameters
    ----------
    actual, lstm_pred, trans_pred : (1440,) float arrays in TECU
    output_dir : directory to write into (OUTPUT_DIR_1 or OUTPUT_DIR_2)
    filename   : output file name (default: day41_ionospheric_delay.csv)
    """
    iono_actual = tec_to_iono_delay(actual)
    iono_lstm   = tec_to_iono_delay(lstm_pred)
    iono_trans  = tec_to_iono_delay(trans_pred)

    out_path = output_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Time(UTC)",
            "TEC_actual(TECU)",      "Delay_actual_L1(m)",
            "TEC_LSTM(TECU)",        "Delay_LSTM_L1(m)",
            "TEC_Transformer(TECU)", "Delay_Transformer_L1(m)",
        ])
        for i, t in enumerate(MINUTE_LABELS):
            writer.writerow([
                t,
                f"{actual[i]:.6f}",     f"{iono_actual[i]:.8f}",
                f"{lstm_pred[i]:.6f}",  f"{iono_lstm[i]:.8f}",
                f"{trans_pred[i]:.6f}", f"{iono_trans[i]:.8f}",
            ])

    print(f"Delay table saved → {out_path}")