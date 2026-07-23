"""
Day-N TEC prediction plots for LSTM and Transformer.

Every plot is automatically saved to disk under the dataset's
generated-plots folder (see config.PLOTS_DIR_1 / PLOTS_DIR_2).
Filenames encode the plot type, the model, the target day (or its
real calendar date when known), and an optional dataset label, e.g.:

    actual_vs_predicted_LSTM_10_May_2024_dataset2.png
    actual_vs_predicted_transformer_day41_dataset1.png
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from src.configs.config import (
    PLOT_DPI,
    PLOT_FIGSIZE_SINGLE,
    COLOR_ACTUAL, COLOR_LSTM_PRED, COLOR_TRANS_PRED,
)
from src.utils.plot_style import _style_timeseries_ax, _legend_above


def _day_or_date(target_day: int, date_label: str = None) -> str:
    """Prefer a real calendar date label; fall back to 'dayN'."""
    return date_label if date_label else f"day{target_day}"


def _make_filename(*parts: object) -> str:
    """Build a `part1_part2..._partN.png` filename, skipping empty parts."""
    clean = [str(p) for p in parts if p not in (None, "")]
    return "_".join(clean) + ".png"


def plot_lstm_prediction(
    actual: np.ndarray,
    lstm_pred: np.ndarray,
    target_day: int = 41,
    dataset_label: str = None,
    date_label: str = None,
    output_dir: str = "plots",
    save: bool = True,
) -> tuple[float, float]:
    """Plot LSTM predicted vs actual TEC for a full day."""
    minutes = np.arange(len(actual))
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SINGLE)

    ax.plot(minutes, actual, color=COLOR_ACTUAL, linewidth=1.8, label="Actual", zorder=3)
    ax.plot(minutes, lstm_pred, color=COLOR_LSTM_PRED, linewidth=1.6,
            linestyle="--", label="LSTM Predicted", zorder=3)
    ax.fill_between(minutes, actual, lstm_pred, color=COLOR_LSTM_PRED, alpha=0.12, zorder=1)

    ax.set_xlim(0, len(minutes) - 1)
    _style_timeseries_ax(ax, "TEC (TECU)")
    _legend_above(ax, ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(
            output_dir,
            _make_filename("actual_vs_predicted_LSTM", _day_or_date(target_day, date_label), dataset_label),
        )
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.show()

    rmse_v = float(np.sqrt(np.mean((lstm_pred - actual) ** 2)))
    mae_v = float(np.mean(np.abs(lstm_pred - actual)))
    print(f"LSTM Day-{target_day} RMSE : {rmse_v:.4f} TECU")
    print(f"LSTM Day-{target_day} MAE  : {mae_v:.4f} TECU")
    return rmse_v, mae_v


def plot_transformer_prediction(
    actual: np.ndarray,
    trans_pred: np.ndarray,
    target_day: int = 41,
    dataset_label: str = None,
    date_label: str = None,
    output_dir: str = "plots",
    save: bool = True,
) -> tuple[float, float]:
    """Plot Transformer predicted vs actual TEC for a full day."""
    minutes = np.arange(len(actual))
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SINGLE)

    ax.plot(minutes, actual, color=COLOR_ACTUAL, linewidth=1.8, label="Actual", zorder=3)
    ax.plot(minutes, trans_pred, color=COLOR_TRANS_PRED, linewidth=1.6,
            linestyle="--", label="Transformer Predicted", zorder=3)
    ax.fill_between(minutes, actual, trans_pred, color=COLOR_TRANS_PRED, alpha=0.15, zorder=1)

    ax.set_xlim(0, len(minutes) - 1)
    _style_timeseries_ax(ax, "TEC (TECU)")
    _legend_above(ax, ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(
            output_dir,
            _make_filename("actual_vs_predicted_transformer", _day_or_date(target_day, date_label), dataset_label),
        )
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.show()

    rmse_v = float(np.sqrt(np.mean((trans_pred - actual) ** 2)))
    mae_v = float(np.mean(np.abs(trans_pred - actual)))
    print(f"Transformer Day-{target_day} RMSE : {rmse_v:.4f} TECU")
    print(f"Transformer Day-{target_day} MAE  : {mae_v:.4f} TECU")
    return rmse_v, mae_v