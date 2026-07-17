"""
Ionospheric delay plots:
  - per-model, per-frequency full-day traces
  - diurnal hourly profile comparison

Every plot is automatically saved to disk under the dataset's
generated-plots folder (see config.PLOTS_DIR_1 / PLOTS_DIR_2).
Filenames encode the plot type, the model, the frequency band, the
target day (or its real calendar date when known), and an optional
dataset label, e.g.:

    ionospheric_delay_LSTM_L1_10_May_2024_dataset2.png
    diurnal_profile_L1_day41_dataset1.png
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

from src.configs.config import (
    F_L1, F_L2, F_L5,
    PLOT_LABEL_FONTSIZE, PLOT_LABEL_FONTWEIGHT,
    PLOT_TICK_FONTSIZE, PLOT_TICK_FONTWEIGHT,
    PLOT_LEGEND_FONTSIZE, PLOT_LEGEND_FONTWEIGHT,
    PLOT_GRID_ALPHA, PLOT_DPI,
    get_date_label,
)
from src.utils.iono_delay import tec_to_iono_delay


_TICK_POS    = np.arange(0, 1441, 120)
_TICK_LABELS = [f"{h:02d}:00" for h in range(0, 25, 2)]

_FREQ_NAMES = {F_L1: "L1", F_L2: "L2", F_L5: "L5"}


def _freq_name(freq) -> str:
    return _FREQ_NAMES.get(freq, str(freq))


def _day_or_date(target_day: int, date_label: str = None) -> str:
    """Prefer a real calendar date label; fall back to 'dayN'."""
    return date_label if date_label else f"day{target_day}"


def _apply_ax_style(ax, tick_pos, tick_labels, ylabel="Iono. Delay (m)"):
    ax.set_ylabel(ylabel, fontsize=PLOT_LABEL_FONTSIZE, fontweight=PLOT_LABEL_FONTWEIGHT)
    ax.set_xlabel("Time of Day (UTC)", fontsize=PLOT_LABEL_FONTSIZE, fontweight=PLOT_LABEL_FONTWEIGHT)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=PLOT_TICK_FONTSIZE, fontweight=PLOT_TICK_FONTWEIGHT)
    ax.tick_params(axis="y", labelsize=PLOT_TICK_FONTSIZE)
    for tick in ax.get_yticklabels():
        tick.set_fontweight(PLOT_TICK_FONTWEIGHT)
    ax.legend(prop={"weight": PLOT_LEGEND_FONTWEIGHT, "size": PLOT_LEGEND_FONTSIZE})
    ax.grid(True, alpha=PLOT_GRID_ALPHA)


# ── saving helpers ────────────────────────────────────────────────────────────

def _make_filename(*parts: object) -> str:
    """Build a `part1_part2..._partN.png` filename, skipping empty parts."""
    clean = [str(p) for p in parts if p not in (None, "")]
    return "_".join(clean) + ".png"


def _save_fig(fig, output_dir: str, *name_parts: object) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = _make_filename(*name_parts)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved: {path}")
    return path


# ── LSTM delay plots ──────────────────────────────────────────────────────────

def plot_lstm_delay(
    actual: np.ndarray,
    lstm_pred: np.ndarray,
    freq: Literal["F_L1", "F_L2", "F_L5"],
    target_day: int = 41,
    dataset_label: str = None,
    date_label: str = None,
    output_dir: str = "plots",
    save: bool = True,
) -> None:

    colors = {
        F_L1: ("steelblue", "tomato"),
        F_L2: ("seagreen", "darkorange"),
        F_L5: ("mediumpurple", "crimson")
    }

    actual_color, pred_color = colors.get(
        freq, ("steelblue", "tomato")
    )

    minutes = np.arange(len(actual))

    iono_actual = tec_to_iono_delay(actual, frequency=freq)

    iono_lstm   = tec_to_iono_delay(lstm_pred, frequency=freq)

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(
        minutes,
        iono_actual,
        color=actual_color,
        linewidth=1.4,
        label="Actual")

    ax.plot(
        minutes,
        iono_lstm,
        color=pred_color,
        linewidth=1.4,
        linestyle="--",
        label="LSTM Predicted")

    ax.fill_between(
        minutes,
        iono_actual,
        iono_lstm,
        alpha=0.15,
        color=pred_color,
        label="Error band")

    _apply_ax_style(ax, _TICK_POS, _TICK_LABELS)
    plt.tight_layout()

    if save:
        _save_fig(
            fig, output_dir,
            "ionospheric_delay_LSTM", _freq_name(freq),
            _day_or_date(target_day, date_label), dataset_label,
        )

    plt.show()
    rmse = np.sqrt(np.mean((iono_lstm - iono_actual) ** 2))
    mae  = np.mean(np.abs(iono_lstm - iono_actual))
    print(f"LSTM {freq} — RMSE: {rmse:.5f} m   MAE: {mae:.5f} m")


# ── Transformer delay plots ───────────────────────────────────────────────────

def plot_transformer_delay(
    actual: np.ndarray,
    trans_pred: np.ndarray,
    freq: Literal["F_L1", "F_L2", "F_L5"],
    target_day: int = 41,
    dataset_label: str = None,
    date_label: str = None,
    output_dir: str = "plots",
    save: bool = True,
) -> None:

    colors = {
        F_L1: ("steelblue", "tomato"),
        F_L2: ("seagreen", "darkorange"),
        F_L5: ("mediumpurple", "crimson")
    }

    actual_color, pred_color = colors.get(
        freq, ("steelblue", "tomato")
    )

    minutes = np.arange(len(actual))

    iono_actual = tec_to_iono_delay(actual, frequency=freq)

    iono_trans  = tec_to_iono_delay(trans_pred, frequency=freq)

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(
        minutes,
        iono_actual,
        color=actual_color,
        linewidth=1.4,
        label="Actual")

    ax.plot(
        minutes,
        iono_trans,
        color=pred_color,
        linewidth=1.4,
        linestyle="--",
        label="Transformer Predicted")

    ax.fill_between(
        minutes,
        iono_actual,
        iono_trans,
        alpha=0.15,
        color=pred_color,
        label="Error band")

    _apply_ax_style(ax, _TICK_POS, _TICK_LABELS)
    plt.tight_layout()

    if save:
        _save_fig(
            fig, output_dir,
            "ionospheric_delay_transformer", _freq_name(freq),
            _day_or_date(target_day, date_label), dataset_label,
        )

    plt.show()
    rmse = np.sqrt(np.mean((iono_trans - iono_actual) ** 2))
    mae  = np.mean(np.abs(iono_trans - iono_actual))
    print(f"Transformer {freq} — RMSE: {rmse:.5f} m   MAE: {mae:.5f} m")


# ── Diurnal hourly profile ────────────────────────────────────────────────────

def plot_diurnal_profile(
    actual: np.ndarray,
    lstm_pred: np.ndarray,
    trans_pred: np.ndarray,
    frequency: float = F_L1,
    target_day: int = 41,
    dataset_label: str = None,
    date_label: str = None,
    output_dir: str = "plots",
    save: bool = True,
) -> None:
    """Hourly-mean ionospheric delay for actual, LSTM, and Transformer."""
    iono_actual = tec_to_iono_delay(actual,     frequency=frequency)
    iono_lstm   = tec_to_iono_delay(lstm_pred,  frequency=frequency)
    iono_trans  = tec_to_iono_delay(trans_pred, frequency=frequency)

    def hourly_mean(arr):
        return arr.reshape(24, 60).mean(axis=1)

    hours    = np.arange(24)
    act_mu   = hourly_mean(iono_actual)
    lstm_mu  = hourly_mean(iono_lstm)
    tr_mu    = hourly_mean(iono_trans)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hours, act_mu,  "o-",  color="steelblue", linewidth=1.6, label="Actual")
    ax.plot(hours, lstm_mu, "s--", color="tomato",    linewidth=1.4, label="LSTM")
    ax.plot(hours, tr_mu,   "^--", color="darkorange",linewidth=1.4, label="Transformer")

    ax.set_xlabel("Hour (UTC)",       fontsize=PLOT_LABEL_FONTSIZE, fontweight=PLOT_LABEL_FONTWEIGHT)
    ax.set_ylabel("Iono. Delay (m)",  fontsize=PLOT_LABEL_FONTSIZE, fontweight=PLOT_LABEL_FONTWEIGHT)
    ax.set_xticks(hours)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_FONTSIZE)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight(PLOT_TICK_FONTWEIGHT)
    ax.legend(prop={"weight": PLOT_LEGEND_FONTWEIGHT, "size": PLOT_LEGEND_FONTSIZE})
    ax.grid(True, alpha=PLOT_GRID_ALPHA)
    plt.tight_layout()

    if save:
        _save_fig(
            fig, output_dir,
            "diurnal_profile", _freq_name(frequency),
            _day_or_date(target_day, date_label), dataset_label,
        )

    plt.show()