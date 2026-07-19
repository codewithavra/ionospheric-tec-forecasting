# """
# Ionospheric delay plots:
#   - per-model, per-frequency full-day traces
#   - combined L1 + L5 full-day traces (one figure, two stacked panels)
#   - diurnal hourly profile comparison

# Every plot is automatically saved to disk under the dataset's
# generated-plots folder (see config.PLOTS_DIR_1 / PLOTS_DIR_2).
# Filenames encode the plot type, the model, the frequency band, the
# target day (or its real calendar date when known), and an optional
# dataset label, e.g.:

#     ionospheric_delay_LSTM_L1_10_May_2024_dataset2.png
#     ionospheric_delay_LSTM_L1_L5_combined_10_May_2024_dataset2.png
#     diurnal_profile_L1_day41_dataset1.png
# """

# import os

# import numpy as np
# import matplotlib.pyplot as plt
# from typing import Literal

# from src.configs.config import (
#     F_L1, F_L2, F_L5,
#     PLOT_LABEL_FONTSIZE, PLOT_LABEL_FONTWEIGHT,
#     PLOT_TICK_FONTSIZE, PLOT_TICK_FONTWEIGHT,
#     PLOT_LEGEND_FONTSIZE, PLOT_LEGEND_FONTWEIGHT,
#     PLOT_GRID_ALPHA, PLOT_DPI,
#     PLOT_FIGSIZE_SINGLE, PLOT_FIGSIZE_STACKED, PLOT_FIGSIZE_DIURNAL,
#     get_date_label,
#     get_time_ticks,
# )
# from src.utils.iono_delay import tec_to_iono_delay


# _TICK_POS, _TICK_LABELS = get_time_ticks()

# _FREQ_NAMES = {F_L1: "L1", F_L2: "L2", F_L5: "L5"}


# def _freq_name(freq) -> str:
#     return _FREQ_NAMES.get(freq, str(freq))


# def _day_or_date(target_day: int, date_label: str = None) -> str:
#     """Prefer a real calendar date label; fall back to 'dayN'."""
#     return date_label if date_label else f"day{target_day}"


# def _apply_ax_style(ax, tick_pos, tick_labels, ylabel="Iono. Delay (m)"):
#     ax.set_ylabel(ylabel, fontsize=PLOT_LABEL_FONTSIZE, fontweight=PLOT_LABEL_FONTWEIGHT)
#     ax.set_xlabel("Time of Day (UTC)", fontsize=PLOT_LABEL_FONTSIZE, fontweight=PLOT_LABEL_FONTWEIGHT)
#     ax.set_xticks(tick_pos)
#     ax.set_xticklabels(tick_labels, rotation=45, fontsize=PLOT_TICK_FONTSIZE, fontweight=PLOT_TICK_FONTWEIGHT)
#     ax.tick_params(axis="y", labelsize=PLOT_TICK_FONTSIZE)
#     for tick in ax.get_yticklabels():
#         tick.set_fontweight(PLOT_TICK_FONTWEIGHT)
#     ax.legend(prop={"weight": PLOT_LEGEND_FONTWEIGHT, "size": PLOT_LEGEND_FONTSIZE})
#     ax.grid(True, alpha=PLOT_GRID_ALPHA)


# # ── saving helpers ────────────────────────────────────────────────────────────

# def _make_filename(*parts: object) -> str:
#     """Build a `part1_part2..._partN.png` filename, skipping empty parts."""
#     clean = [str(p) for p in parts if p not in (None, "")]
#     return "_".join(clean) + ".png"


# def _save_fig(fig, output_dir: str, *name_parts: object) -> str:
#     os.makedirs(output_dir, exist_ok=True)
#     filename = _make_filename(*name_parts)
#     path = os.path.join(output_dir, filename)
#     fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
#     print(f"Saved: {path}")
#     return path


# # ── LSTM delay plots ──────────────────────────────────────────────────────────

# def plot_lstm_delay(
#     actual: np.ndarray,
#     lstm_pred: np.ndarray,
#     freq: Literal["F_L1", "F_L2", "F_L5"],
#     target_day: int = 41,
#     dataset_label: str = None,
#     date_label: str = None,
#     output_dir: str = "plots",
#     save: bool = True,
# ) -> None:

#     colors = {
#         F_L1: ("steelblue", "tomato"),
#         F_L2: ("seagreen", "darkorange"),
#         F_L5: ("mediumpurple", "crimson")
#     }

#     actual_color, pred_color = colors.get(
#         freq, ("steelblue", "tomato")
#     )

#     minutes = np.arange(len(actual))

#     iono_actual = tec_to_iono_delay(actual, frequency=freq)

#     iono_lstm   = tec_to_iono_delay(lstm_pred, frequency=freq)

#     fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SINGLE)
#     ax.plot(
#         minutes,
#         iono_actual,
#         color=actual_color,
#         linewidth=1.4,
#         label="Actual")

#     ax.plot(
#         minutes,
#         iono_lstm,
#         color=pred_color,
#         linewidth=1.4,
#         linestyle="--",
#         label="LSTM Predicted")

#     ax.fill_between(
#         minutes,
#         iono_actual,
#         iono_lstm,
#         alpha=0.15,
#         color=pred_color,
#         label="Error band")

#     _apply_ax_style(ax, _TICK_POS, _TICK_LABELS)
#     plt.tight_layout()

#     if save:
#         _save_fig(
#             fig, output_dir,
#             "ionospheric_delay_LSTM", _freq_name(freq),
#             _day_or_date(target_day, date_label), dataset_label,
#         )

#     plt.show()
#     rmse = np.sqrt(np.mean((iono_lstm - iono_actual) ** 2))
#     mae  = np.mean(np.abs(iono_lstm - iono_actual))
#     print(f"LSTM {freq} — RMSE: {rmse:.5f} m   MAE: {mae:.5f} m")


# # ── Transformer delay plots ───────────────────────────────────────────────────

# def plot_transformer_delay(
#     actual: np.ndarray,
#     trans_pred: np.ndarray,
#     freq: Literal["F_L1", "F_L2", "F_L5"],
#     target_day: int = 41,
#     dataset_label: str = None,
#     date_label: str = None,
#     output_dir: str = "plots",
#     save: bool = True,
# ) -> None:

#     colors = {
#         F_L1: ("steelblue", "tomato"),
#         F_L2: ("seagreen", "darkorange"),
#         F_L5: ("mediumpurple", "crimson")
#     }

#     actual_color, pred_color = colors.get(
#         freq, ("steelblue", "tomato")
#     )

#     minutes = np.arange(len(actual))

#     iono_actual = tec_to_iono_delay(actual, frequency=freq)

#     iono_trans  = tec_to_iono_delay(trans_pred, frequency=freq)

#     fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SINGLE)
#     ax.plot(
#         minutes,
#         iono_actual,
#         color=actual_color,
#         linewidth=1.4,
#         label="Actual")

#     ax.plot(
#         minutes,
#         iono_trans,
#         color=pred_color,
#         linewidth=1.4,
#         linestyle="--",
#         label="Transformer Predicted")

#     ax.fill_between(
#         minutes,
#         iono_actual,
#         iono_trans,
#         alpha=0.15,
#         color=pred_color,
#         label="Error band")

#     _apply_ax_style(ax, _TICK_POS, _TICK_LABELS)
#     plt.tight_layout()

#     if save:
#         _save_fig(
#             fig, output_dir,
#             "ionospheric_delay_transformer", _freq_name(freq),
#             _day_or_date(target_day, date_label), dataset_label,
#         )

#     plt.show()
#     rmse = np.sqrt(np.mean((iono_trans - iono_actual) ** 2))
#     mae  = np.mean(np.abs(iono_trans - iono_actual))
#     print(f"Transformer {freq} — RMSE: {rmse:.5f} m   MAE: {mae:.5f} m")


# # ── Combined L1 + L5 delay plots ──────────────────────────────────────────────

# def _plot_combined_l1_l5_delay(
#     actual: np.ndarray,
#     pred: np.ndarray,
#     pred_label: str,
#     pred_color: str,
#     target_day: int,
#     dataset_label: str,
#     date_label: str,
#     output_dir: str,
#     save: bool,
#     filename_prefix: str,
# ) -> None:
#     """Shared implementation: stacks an L1 panel above an L5 panel,
#     each showing actual vs. predicted ionospheric delay for the full day."""
#     minutes = np.arange(len(actual))
#     freqs = [(F_L1, "L1"), (F_L5, "L5")]

#     fig, axes = plt.subplots(2, 1, figsize=PLOT_FIGSIZE_STACKED, sharex=True)

#     for ax, (freq, fname) in zip(axes, freqs):
#         iono_actual = tec_to_iono_delay(actual, frequency=freq)
#         iono_pred   = tec_to_iono_delay(pred,   frequency=freq)

#         ax.plot(
#             minutes, iono_actual,
#             color="steelblue", linewidth=1.4,
#             label=f"Actual ({fname})")
#         ax.plot(
#             minutes, iono_pred,
#             color=pred_color, linewidth=1.4, linestyle="--",
#             label=f"{pred_label} ({fname})")
#         ax.fill_between(
#             minutes, iono_actual, iono_pred,
#             alpha=0.15, color=pred_color)

#         ax.set_ylabel("Iono. Delay (m)", fontsize=PLOT_LABEL_FONTSIZE, fontweight=PLOT_LABEL_FONTWEIGHT)
#         ax.tick_params(axis="y", labelsize=PLOT_TICK_FONTSIZE)
#         for tick in ax.get_yticklabels():
#             tick.set_fontweight(PLOT_TICK_FONTWEIGHT)
#         ax.legend(prop={"weight": PLOT_LEGEND_FONTWEIGHT, "size": PLOT_LEGEND_FONTSIZE})
#         ax.grid(True, alpha=PLOT_GRID_ALPHA)

#         rmse = np.sqrt(np.mean((iono_pred - iono_actual) ** 2))
#         mae  = np.mean(np.abs(iono_pred - iono_actual))
#         print(f"{pred_label} {fname} — RMSE: {rmse:.5f} m   MAE: {mae:.5f} m")

#     axes[-1].set_xticks(_TICK_POS)
#     axes[-1].set_xticklabels(_TICK_LABELS, rotation=45, fontsize=PLOT_TICK_FONTSIZE, fontweight=PLOT_TICK_FONTWEIGHT)
#     axes[-1].set_xlabel("Time of Day (UTC)", fontsize=PLOT_LABEL_FONTSIZE, fontweight=PLOT_LABEL_FONTWEIGHT)

#     plt.tight_layout()

#     if save:
#         _save_fig(
#             fig, output_dir,
#             filename_prefix, "L1_L5_combined",
#             _day_or_date(target_day, date_label), dataset_label,
#         )

#     plt.show()


# def plot_lstm_delay_l1_l5_combined(
#     actual: np.ndarray,
#     lstm_pred: np.ndarray,
#     target_day: int = 41,
#     dataset_label: str = None,
#     date_label: str = None,
#     output_dir: str = "plots",
#     save: bool = True,
# ) -> None:
#     """L1 (top) and L5 (bottom) ionospheric delay, actual vs. LSTM predicted,
#     for the full day — one figure, stacked directly below the per-frequency
#     LSTM delay plots."""
#     _plot_combined_l1_l5_delay(
#         actual, lstm_pred, "LSTM Predicted", "tomato",
#         target_day, dataset_label, date_label, output_dir, save,
#         "ionospheric_delay_LSTM",
#     )


# def plot_transformer_delay_l1_l5_combined(
#     actual: np.ndarray,
#     trans_pred: np.ndarray,
#     target_day: int = 41,
#     dataset_label: str = None,
#     date_label: str = None,
#     output_dir: str = "plots",
#     save: bool = True,
# ) -> None:
#     """L1 (top) and L5 (bottom) ionospheric delay, actual vs. Transformer
#     predicted, for the full day — one figure, stacked directly below the
#     per-frequency Transformer delay plots."""
#     _plot_combined_l1_l5_delay(
#         actual, trans_pred, "Transformer Predicted", "darkorange",
#         target_day, dataset_label, date_label, output_dir, save,
#         "ionospheric_delay_transformer",
#     )


# # ── Diurnal hourly profile ────────────────────────────────────────────────────

# def plot_diurnal_profile(
#     actual: np.ndarray,
#     lstm_pred: np.ndarray,
#     trans_pred: np.ndarray,
#     frequency: float = F_L1,
#     target_day: int = 41,
#     dataset_label: str = None,
#     date_label: str = None,
#     output_dir: str = "plots",
#     save: bool = True,
# ) -> None:
#     """Hourly-mean ionospheric delay for actual, LSTM, and Transformer."""
#     iono_actual = tec_to_iono_delay(actual,     frequency=frequency)
#     iono_lstm   = tec_to_iono_delay(lstm_pred,  frequency=frequency)
#     iono_trans  = tec_to_iono_delay(trans_pred, frequency=frequency)

#     def hourly_mean(arr):
#         return arr.reshape(24, 60).mean(axis=1)

#     hours    = np.arange(24)
#     act_mu   = hourly_mean(iono_actual)
#     lstm_mu  = hourly_mean(iono_lstm)
#     tr_mu    = hourly_mean(iono_trans)

#     fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_DIURNAL)
#     ax.plot(hours, act_mu,  "o-",  color="steelblue", linewidth=1.6, label="Actual")
#     ax.plot(hours, lstm_mu, "s--", color="tomato",    linewidth=1.4, label="LSTM")
#     ax.plot(hours, tr_mu,   "^--", color="darkorange",linewidth=1.4, label="Transformer")

#     ax.set_xlabel("Hour (UTC)",       fontsize=PLOT_LABEL_FONTSIZE, fontweight=PLOT_LABEL_FONTWEIGHT)
#     ax.set_ylabel("Iono. Delay (m)",  fontsize=PLOT_LABEL_FONTSIZE, fontweight=PLOT_LABEL_FONTWEIGHT)
#     ax.set_xticks(hours)
#     ax.tick_params(axis="both", labelsize=PLOT_TICK_FONTSIZE)
#     for tick in ax.get_xticklabels() + ax.get_yticklabels():
#         tick.set_fontweight(PLOT_TICK_FONTWEIGHT)
#     ax.legend(prop={"weight": PLOT_LEGEND_FONTWEIGHT, "size": PLOT_LEGEND_FONTSIZE})
#     ax.grid(True, alpha=PLOT_GRID_ALPHA)
#     plt.tight_layout()

#     if save:
#         _save_fig(
#             fig, output_dir,
#             "diurnal_profile", _freq_name(frequency),
#             _day_or_date(target_day, date_label), dataset_label,
#         )

#     plt.show()


"""
Ionospheric delay plots:
  - per-model, per-frequency full-day traces
  - combined L1 + L5 full-day traces (one figure, two stacked panels)
  - diurnal hourly profile comparison

Every plot is automatically saved to disk under the dataset's
generated-plots folder (see config.PLOTS_DIR_1 / PLOTS_DIR_2).
Filenames encode the plot type, the model, the frequency band, the
target day (or its real calendar date when known), and an optional
dataset label, e.g.:

    ionospheric_delay_LSTM_L1_10_May_2024_dataset2.png
    ionospheric_delay_LSTM_L1_L5_combined_10_May_2024_dataset2.png
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
    PLOT_FIGSIZE_SINGLE, PLOT_FIGSIZE_STACKED, PLOT_FIGSIZE_DIURNAL,
    get_date_label,
    get_time_ticks,
)
from src.utils.iono_delay import tec_to_iono_delay


_TICK_POS, _TICK_LABELS = get_time_ticks()

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

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SINGLE)
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

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SINGLE)
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


# ── Combined L1 + L5 delay plots ──────────────────────────────────────────────

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
    """Shared implementation: single-panel overlay of L1 and L5 ionospheric
    delay (actual vs. predicted for both frequencies on the same axes) for
    the full day, so the L1/L5 traces can be compared directly."""
    minutes = np.arange(len(actual))

    # L1 in blue/pred_color tones, L5 in a darker/lighter shade of the same
    # families so the four lines stay visually grouped by frequency while
    # still being distinguishable from each other.
    freq_styles = [
        (F_L1, "L1", "steelblue", pred_color,  "-",  "--"),
        (F_L5, "L5", "navy",      "firebrick",  "-",  "--"),
    ]

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SINGLE)

    for freq, fname, actual_color, pred_color_f, actual_ls, pred_ls in freq_styles:
        iono_actual = tec_to_iono_delay(actual, frequency=freq)
        iono_pred   = tec_to_iono_delay(pred,   frequency=freq)

        ax.plot(
            minutes, iono_actual,
            color=actual_color, linewidth=1.4, linestyle=actual_ls,
            label=f"Actual ({fname})")
        ax.plot(
            minutes, iono_pred,
            color=pred_color_f, linewidth=1.4, linestyle=pred_ls,
            label=f"{pred_label} ({fname})")
        ax.fill_between(
            minutes, iono_actual, iono_pred,
            alpha=0.12, color=pred_color_f)

        rmse = np.sqrt(np.mean((iono_pred - iono_actual) ** 2))
        mae  = np.mean(np.abs(iono_pred - iono_actual))
        print(f"{pred_label} {fname} — RMSE: {rmse:.5f} m   MAE: {mae:.5f} m")

    _apply_ax_style(ax, _TICK_POS, _TICK_LABELS)
    plt.tight_layout()

    if save:
        _save_fig(
            fig, output_dir,
            filename_prefix, "L1_L5_combined",
            _day_or_date(target_day, date_label), dataset_label,
        )

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
    """L1 (top) and L5 (bottom) ionospheric delay, actual vs. LSTM predicted,
    for the full day — one figure, stacked directly below the per-frequency
    LSTM delay plots."""
    _plot_combined_l1_l5_delay(
        actual, lstm_pred, "LSTM Predicted", "tomato",
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
    """L1 (top) and L5 (bottom) ionospheric delay, actual vs. Transformer
    predicted, for the full day — one figure, stacked directly below the
    per-frequency Transformer delay plots."""
    _plot_combined_l1_l5_delay(
        actual, trans_pred, "Transformer Predicted", "darkorange",
        target_day, dataset_label, date_label, output_dir, save,
        "ionospheric_delay_transformer",
    )


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

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_DIURNAL)
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