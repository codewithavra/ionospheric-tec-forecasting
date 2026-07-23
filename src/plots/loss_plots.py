"""
Training vs validation loss curves for LSTM and Transformer.

Every plot is automatically saved to disk under the dataset's
generated-plots folder (see config.PLOTS_DIR_1 / PLOTS_DIR_2).
Filenames encode the plot type, the model, and an optional dataset
label, e.g.:

    loss_LSTM_dataset2.png
    loss_transformer_dataset2.png
"""

import os

import matplotlib.pyplot as plt

from src.configs.config import (
    PLOT_LABEL_FONTSIZE, PLOT_LABEL_FONTWEIGHT,
    PLOT_TICK_FONTSIZE, PLOT_TICK_FONTWEIGHT,
    PLOT_LEGEND_FONTSIZE, PLOT_LEGEND_FONTWEIGHT,
    PLOT_GRID_ALPHA, PLOT_DPI,
    PLOT_FIGSIZE_LOSS,
)


def _make_filename(*parts: object) -> str:
    """Build a `part1_part2..._partN.png` filename, skipping empty parts."""
    clean = [str(p) for p in parts if p not in (None, "")]
    return "_".join(clean) + ".png"


def plot_lstm_loss(
    history,
    dataset_label: str = None,
    output_dir: str = "plots",
    save: bool = True,
) -> None:
    """Plot LSTM training and validation MSE loss curves."""
    epochs_range = range(1, len(history["loss"]) + 1)
    # Thin the tick labels so they don't collide when there are many
    # epochs — always show epoch 1, then every 10th epoch.
    tick_positions = [e for e in epochs_range if e == 1 or e % 10 == 0]

    plt.figure(figsize=PLOT_FIGSIZE_LOSS)
    plt.plot(epochs_range, history["loss"], marker="o", linewidth=1.5, label="Training Loss")
    plt.plot(epochs_range, history["val_loss"], marker="s", linewidth=1.5,
             linestyle="--", label="Validation Loss")

    plt.xlabel("Epoch", fontsize=PLOT_LABEL_FONTSIZE, fontweight=PLOT_LABEL_FONTWEIGHT)
    plt.ylabel("MSE Loss", fontsize=PLOT_LABEL_FONTSIZE, fontweight=PLOT_LABEL_FONTWEIGHT)
    plt.xticks(ticks=tick_positions, labels=[str(e) for e in tick_positions],
               fontsize=PLOT_TICK_FONTSIZE, fontweight=PLOT_TICK_FONTWEIGHT)
    plt.yticks(fontsize=PLOT_TICK_FONTSIZE, fontweight=PLOT_TICK_FONTWEIGHT)
    plt.legend(prop={"weight": PLOT_LEGEND_FONTWEIGHT, "size": PLOT_LEGEND_FONTSIZE})
    plt.grid(True, alpha=PLOT_GRID_ALPHA)
    plt.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, _make_filename("loss_LSTM", dataset_label))
        plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.show()
    print(f"Final train loss : {history['loss'][-1]:.6f}")
    print(f"Final val   loss : {history['val_loss'][-1]:.6f}")


def plot_transformer_loss(
    history,
    dataset_label: str = None,
    output_dir: str = "plots",
    save: bool = True,
) -> None:
    """Plot Transformer training and validation MSE loss curves.

    Not called by the reference notebook (only the LSTM loss curve is
    plotted there), but kept here — identical styling — in case the
    project wants a matching Transformer loss plot.
    """
    epochs_range = range(1, len(history["loss"]) + 1)
    tick_positions = [e for e in epochs_range if e == 1 or e % 10 == 0]

    plt.figure(figsize=PLOT_FIGSIZE_LOSS)
    plt.plot(epochs_range, history["loss"], marker="o", linewidth=1.5, label="Training Loss")
    plt.plot(epochs_range, history["val_loss"], marker="s", linewidth=1.5,
             linestyle="--", label="Validation Loss")

    plt.xlabel("Epoch", fontsize=PLOT_LABEL_FONTSIZE, fontweight=PLOT_LABEL_FONTWEIGHT)
    plt.ylabel("MSE Loss", fontsize=PLOT_LABEL_FONTSIZE, fontweight=PLOT_LABEL_FONTWEIGHT)
    plt.xticks(ticks=tick_positions, labels=[str(e) for e in tick_positions],
               fontsize=PLOT_TICK_FONTSIZE, fontweight=PLOT_TICK_FONTWEIGHT)
    plt.yticks(fontsize=PLOT_TICK_FONTSIZE, fontweight=PLOT_TICK_FONTWEIGHT)
    plt.legend(prop={"weight": PLOT_LEGEND_FONTWEIGHT, "size": PLOT_LEGEND_FONTSIZE})
    plt.grid(True, alpha=PLOT_GRID_ALPHA)
    plt.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, _make_filename("loss_transformer", dataset_label))
        plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.show()
    print(f"Final train loss : {history['loss'][-1]:.6f}")
    print(f"Final val   loss : {history['val_loss'][-1]:.6f}")