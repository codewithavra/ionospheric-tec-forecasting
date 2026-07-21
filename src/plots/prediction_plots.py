"""
Day-N TEC prediction plots for LSTM and Transformer.

Every plot is automatically saved to disk under the dataset's
generated-plots folder (see config.PLOTS_DIR_1 / PLOTS_DIR_2).
Filenames encode the plot type, the model, the target day (or its
real calendar date when known), and an optional dataset label, e.g.:

    actual_vs_predicted_LSTM_10_May_2024_dataset2.png
    actual_vs_predicted_transformer_day41_dataset1.png

Pass `filename=` to override the auto-generated name, or `save=False`
to skip saving entirely.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from src.configs.config import (
    PLOT_LABEL_FONTSIZE, PLOT_LABEL_FONTWEIGHT,
    PLOT_TICK_FONTSIZE, PLOT_TICK_FONTWEIGHT,
    PLOT_LEGEND_FONTSIZE, PLOT_LEGEND_FONTWEIGHT,
    PLOT_GRID_ALPHA, PLOT_DPI,
    PLOT_FIGSIZE_SINGLE,
    get_time_ticks,
)

_TICK_POS, _TICK_LABELS = get_time_ticks()


def _day_or_date(target_day: int, date_label: str = None) -> str:
    """Prefer a real calendar date label; fall back to 'dayN'."""
    return date_label if date_label else f"day{target_day}"


# ── saving helpers ────────────────────────────────────────────────────────────

def _make_filename(*parts: object) -> str:
    """Build a `part1_part2..._partN.png` filename, skipping empty parts."""
    clean = [str(p) for p in parts if p not in (None, "")]
    return "_".join(clean) + ".png"


def _resolve_save_path(output_dir: str, filename: str, *default_name_parts: object) -> str:
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = _make_filename(*default_name_parts)
    return os.path.join(output_dir, filename)


def plot_lstm_prediction(
    actual: np.ndarray,
    lstm_pred: np.ndarray,
    actual_label: str = None,
    target_day: int = 41,
    dataset_label: str = None,
    date_label: str = None,
    legend_loc: str = "best",
    legend_size: int = PLOT_LEGEND_FONTSIZE,
    output_dir: str = "plots",
    filename: str = None,
    save: bool = True,
) -> tuple[float, float]:
    """Plot LSTM predicted vs actual TEC."""

    if actual_label is None:
        actual_label = "Actual Day"

    minutes = np.arange(len(actual))

    plt.figure(figsize=PLOT_FIGSIZE_SINGLE)

    plt.plot(
        minutes,
        actual,
        color="steelblue",
        linewidth=1.4,
        label=actual_label,
    )

    plt.plot(
        minutes,
        lstm_pred,
        color="tomato",
        linewidth=1.4,
        linestyle="--",
        label="Predicted Day",
    )

    plt.xlabel("Time of Day (UTC)", fontsize=PLOT_LABEL_FONTSIZE, fontweight=PLOT_LABEL_FONTWEIGHT)
    plt.ylabel("TEC (TECU)", fontsize=PLOT_LABEL_FONTSIZE, fontweight=PLOT_LABEL_FONTWEIGHT)

    plt.xticks(
        ticks=_TICK_POS,
        labels=_TICK_LABELS,
        rotation=45,
        fontsize=PLOT_TICK_FONTSIZE,
        fontweight=PLOT_TICK_FONTWEIGHT,
    )

    plt.yticks(fontsize=PLOT_TICK_FONTSIZE, fontweight=PLOT_TICK_FONTWEIGHT)

    plt.legend(
        loc=legend_loc,
        prop={"weight": PLOT_LEGEND_FONTWEIGHT, "size": legend_size},
    )

    plt.grid(True, alpha=PLOT_GRID_ALPHA)
    plt.tight_layout()

    # Save figure
    if save:
        path = _resolve_save_path(
            output_dir, filename,
            "actual_vs_predicted_LSTM", _day_or_date(target_day, date_label), dataset_label,
        )
        plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.show()
    plt.close()

    rmse = float(np.sqrt(np.mean((lstm_pred - actual) ** 2)))
    mae = float(np.mean(np.abs(lstm_pred - actual)))

    print(f"LSTM Day-{target_day} RMSE : {rmse:.4f} TECU")
    print(f"LSTM Day-{target_day} MAE  : {mae:.4f} TECU")

    return rmse, mae


def plot_transformer_prediction(
    actual: np.ndarray,
    trans_pred: np.ndarray,
    actual_label: str = None,
    target_day: int = 41,
    dataset_label: str = None,
    date_label: str = None,
    legend_loc: str = "best",
    legend_size: int = PLOT_LEGEND_FONTSIZE,
    output_dir: str = "plots",
    filename: str = None,
    save: bool = True,
) -> tuple[float, float]:
    """Plot Transformer predicted vs actual TEC."""

    if actual_label is None:
        actual_label = "Actual Day"

    minutes = np.arange(len(actual))

    plt.figure(figsize=PLOT_FIGSIZE_SINGLE)

    plt.plot(
        minutes,
        actual,
        color="steelblue",
        linewidth=1.4,
        label=actual_label,
    )

    plt.plot(
        minutes,
        trans_pred,
        color="darkorange",
        linewidth=1.4,
        linestyle="--",
        label="Predicted Day",
    )

    plt.xlabel("Time of Day (UTC)", fontsize=PLOT_LABEL_FONTSIZE, fontweight=PLOT_LABEL_FONTWEIGHT)
    plt.ylabel("TEC (TECU)", fontsize=PLOT_LABEL_FONTSIZE, fontweight=PLOT_LABEL_FONTWEIGHT)

    plt.xticks(
        ticks=_TICK_POS,
        labels=_TICK_LABELS,
        rotation=45,
        fontsize=PLOT_TICK_FONTSIZE,
        fontweight=PLOT_TICK_FONTWEIGHT,
    )

    plt.yticks(
        fontsize=PLOT_TICK_FONTSIZE,
        fontweight=PLOT_TICK_FONTWEIGHT,
    )

    plt.legend(
        loc=legend_loc,
        prop={
            "weight": PLOT_LEGEND_FONTWEIGHT,
            "size": legend_size,
        },
    )

    plt.grid(True, alpha=PLOT_GRID_ALPHA)
    plt.tight_layout()

    # Save figure
    if save:
        path = _resolve_save_path(
            output_dir, filename,
            "actual_vs_predicted_transformer", _day_or_date(target_day, date_label), dataset_label,
        )
        plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.show()
    plt.close()

    rmse = float(np.sqrt(np.mean((trans_pred - actual) ** 2)))
    mae = float(np.mean(np.abs(trans_pred - actual)))

    print(f"Transformer Day-{target_day} RMSE : {rmse:.4f} TECU")
    print(f"Transformer Day-{target_day} MAE  : {mae:.4f} TECU")

    return rmse, mae