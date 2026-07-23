"""
Ionospheric delay plots.

Every plot is automatically saved to disk under the dataset's
generated-plots folder (see config.PLOTS_DIR_1 / PLOTS_DIR_2).
Filenames encode the plot type, the model, the target day (or its
real calendar date when known), and an optional dataset label, e.g.:

    ionospheric_delay_LSTM_L1_L5_combined_10_May_2024_dataset2.png
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from src.configs.config import (
    F_L1, F_L5,
    PLOT_DPI, PLOT_FIGSIZE_SINGLE,
    COLOR_LSTM_PRED, COLOR_TRANS_PRED,
    COLOR_L1_ACTUAL, COLOR_L1_PRED,
    COLOR_L5_ACTUAL, COLOR_L5_PRED,
)
from src.utils.iono_delay import tec_to_iono_delay
from src.utils.plot_style import (
    _style_timeseries_ax, _legend_right,
    _day_or_date, _make_filename,
)


# ── Combined L1 + L5 delay plots ─────────────────────────────────────────────

def _plot_combined_l1_l5_delay(
    actual: np.ndarray,
    pred: np.ndarray,
    pred_label: str,
    pred_color: str,
    target_day: int,
    dataset_label: str,
    date_label: str,
    output_dir: str,
    save: bool,
    filename_prefix: str,
) -> None:
    """Single-panel overlay of actual vs. predicted ionospheric delay for
    both L1 and L5 on the same axes, for the full day."""
    minutes = np.arange(len(actual))
    freq_styles = [
        (F_L1, "L1", COLOR_L1_ACTUAL, COLOR_L1_PRED, "-", "--"),
        (F_L5, "L5", COLOR_L5_ACTUAL, COLOR_L5_PRED, "-", "--"),
    ]

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SINGLE)

    for freq, fname, actual_color, pred_color_f, actual_ls, pred_ls in freq_styles:
        iono_actual = tec_to_iono_delay(actual, frequency=freq)
        iono_pred = tec_to_iono_delay(pred, frequency=freq)

        ax.plot(minutes, iono_actual, color=actual_color, linewidth=1.6,
                linestyle=actual_ls, label=f"Actual ({fname})", zorder=3)
        ax.plot(minutes, iono_pred, color=pred_color_f, linewidth=1.4,
                linestyle=pred_ls, label=f"{pred_label} ({fname})", zorder=3)
        ax.fill_between(minutes, iono_actual, iono_pred, alpha=0.10, color=pred_color_f, zorder=1)

        rmse_v = np.sqrt(np.mean((iono_pred - iono_actual) ** 2))
        mae_v = np.mean(np.abs(iono_pred - iono_actual))
        print(f"{pred_label} {fname} — RMSE: {rmse_v:.5f} m   MAE: {mae_v:.5f} m")

    ax.set_xlim(0, len(minutes) - 1)
    _style_timeseries_ax(ax, "Iono. Delay (m)")
    _legend_right(ax, ncol=1)   # 4 legend entries, stacked vertically beside the plot
    fig.tight_layout(rect=[0, 0, 0.82, 1])   # reserve right-side width for the legend

    if save:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(
            output_dir,
            _make_filename(
                filename_prefix, "L1_L5_combined",
                _day_or_date(target_day, date_label), dataset_label,
            ),
        )
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.show()


def plot_lstm_delay_l1_l5_combined(
    actual: np.ndarray,
    lstm_pred: np.ndarray,
    target_day: int = 41,
    dataset_label: str = None,
    date_label: str = None,
    output_dir: str = "plots",
    save: bool = True,
) -> None:
    """L1 + L5 ionospheric delay, actual vs. LSTM predicted, for the full day."""
    _plot_combined_l1_l5_delay(
        actual, lstm_pred, "LSTM", COLOR_LSTM_PRED,
        target_day, dataset_label, date_label, output_dir, save,
        "ionospheric_delay_LSTM",
    )


def plot_transformer_delay_l1_l5_combined(
    actual: np.ndarray,
    trans_pred: np.ndarray,
    target_day: int = 41,
    dataset_label: str = None,
    date_label: str = None,
    output_dir: str = "plots",
    save: bool = True,
) -> None:
    """L1 + L5 ionospheric delay, actual vs. Transformer predicted, for the full day."""
    _plot_combined_l1_l5_delay(
        actual, trans_pred, "Transformer", COLOR_TRANS_PRED,
        target_day, dataset_label, date_label, output_dir, save,
        "ionospheric_delay_transformer",
    )