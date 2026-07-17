"""
Day-N TEC prediction plots for LSTM and Transformer.

Every plot is automatically saved to disk. Filenames encode the plot
type, the model, the target day, an optional dataset label, and
today's date, e.g.:

    prediction_lstm_day41_dataset1_2026-07-17.png
    prediction_transformer_day40_dataset2_2026-07-17.png

Pass `filename=` to override the auto-generated name, or `save=False`
to skip saving entirely.
"""

import os
from datetime import date

import numpy as np
import matplotlib.pyplot as plt

_TICK_POS = np.arange(0, 1441, 120)
_TICK_LABELS = [f"{h:02d}:00" for h in range(0, 25, 2)]


# ── saving helpers ────────────────────────────────────────────────────────────

def _make_filename(*parts: object) -> str:
    """Build a `part1_part2..._YYYY-MM-DD.png` filename, skipping empty parts."""
    clean = [str(p) for p in parts if p not in (None, "")]
    clean.append(date.today().isoformat())
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
    legend_loc: str = "best",
    legend_size: int = 18,
    output_dir: str = "plots",
    filename: str = None,
    save: bool = True,
) -> tuple[float, float]:
    """Plot LSTM predicted vs actual TEC."""

    if actual_label is None:
        actual_label = f"Actual Day {target_day}"

    minutes = np.arange(len(actual))

    plt.figure(figsize=(9, 6))

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
        label=f"LSTM Predicted Day {target_day}",
    )

    plt.xlabel("Time of Day (UTC)", fontsize=18, fontweight="bold")
    plt.ylabel("TEC (TECU)", fontsize=18, fontweight="bold")

    plt.xticks(
        ticks=_TICK_POS,
        labels=_TICK_LABELS,
        rotation=45,
        fontsize=15,
    )

    plt.yticks(fontsize=15)

    plt.legend(
        loc=legend_loc,
        prop={"size": legend_size},
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    if save:
        path = _resolve_save_path(
            output_dir, filename,
            "prediction_lstm", f"day{target_day}", dataset_label,
        )
        plt.savefig(path, dpi=300, bbox_inches="tight")
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
    legend_loc: str = "best",
    legend_size: int = 11,
    output_dir: str = "plots",
    filename: str = None,
    save: bool = True,
) -> tuple[float, float]:
    """Plot Transformer predicted vs actual TEC."""

    if actual_label is None:
        actual_label = f"Actual Day {target_day}"

    minutes = np.arange(len(actual))

    plt.figure(figsize=(15, 5))

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
        label=f"Transformer Predicted Day {target_day}",
    )

    plt.xlabel("Time of Day (UTC)", fontsize=13, fontweight="bold")
    plt.ylabel("TEC (TECU)", fontsize=13, fontweight="bold")

    plt.xticks(
        ticks=_TICK_POS,
        labels=_TICK_LABELS,
        rotation=45,
        fontsize=11,
        fontweight="bold",
    )

    plt.yticks(
        fontsize=11,
        fontweight="bold",
    )

    plt.legend(
        loc=legend_loc,
        prop={
            "weight": "bold",
            "size": legend_size,
        },
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    if save:
        path = _resolve_save_path(
            output_dir, filename,
            "prediction_transformer", f"day{target_day}", dataset_label,
        )
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.show()
    plt.close()

    rmse = float(np.sqrt(np.mean((trans_pred - actual) ** 2)))
    mae = float(np.mean(np.abs(trans_pred - actual)))

    print(f"Transformer Day-{target_day} RMSE : {rmse:.4f} TECU")
    print(f"Transformer Day-{target_day} MAE  : {mae:.4f} TECU")

    return rmse, mae