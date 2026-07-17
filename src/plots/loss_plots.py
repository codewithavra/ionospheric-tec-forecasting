"""
Training vs validation loss curves for LSTM and Transformer.

Every plot is automatically saved to disk. Filenames encode the plot
type, the model, an optional dataset label, and today's date, e.g.:

    loss_lstm_dataset1_2026-07-17.png
    loss_transformer_dataset2_2026-07-17.png
"""

import os
from datetime import date

import matplotlib.pyplot as plt


# ── saving helpers ────────────────────────────────────────────────────────────

def _make_filename(*parts: object) -> str:
    """Build a `part1_part2..._YYYY-MM-DD.png` filename, skipping empty parts."""
    clean = [str(p) for p in parts if p not in (None, "")]
    clean.append(date.today().isoformat())
    return "_".join(clean) + ".png"


def _save_current_fig(output_dir: str, *name_parts: object) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = _make_filename(*name_parts)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    return path


# ── LSTM loss ──────────────────────────────────────────────────────────────────

def plot_lstm_loss(
    history,
    dataset_label: str = None,
    output_dir: str = "plots",
    save: bool = True,
) -> None:
    """Plot LSTM training and validation MSE loss curves."""
    epochs_range = range(1, len(history["loss"]) + 1)
    # Thin the tick labels so they don't collide when there are many
    # epochs (e.g. LSTM_EPOCHS=50) — always show epoch 1, then every
    # 10th epoch, same convention as plot_transformer_loss.
    tick_positions = [e for e in epochs_range if e == 1 or e % 10 == 0]

    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs_range,
        history["loss"],
        marker="o",
        linewidth=1.5,
        label="Training Loss")

    plt.plot(
        epochs_range,
        history["val_loss"],
        marker="s",
        linewidth=1.5,
        linestyle="--",
        label="Validation Loss")

    plt.xlabel(
        "Epoch",
        fontsize=13,
        fontweight="bold")
    plt.ylabel(
        "MSE Loss",
        fontsize=13,
        fontweight="bold")

    plt.xticks(
        ticks=tick_positions,
        labels=[str(e) for e in tick_positions],
        fontsize=11,
        fontweight="bold")

    plt.yticks(
        fontsize=11,
        fontweight="bold")

    plt.legend(prop={"weight": "bold", "size": 11})
    plt.grid(True, alpha=0.35)
    plt.tight_layout()

    if save:
        _save_current_fig(output_dir, "loss_lstm", dataset_label)

    plt.show()
    plt.close()

    print(f"Final train loss : {history['loss'][-1]:.6f}")
    print(f"Final val   loss : {history['val_loss'][-1]:.6f}")


def plot_transformer_loss(
    history,
    dataset_label: str = None,
    output_dir: str = "plots",
    save: bool = True,
) -> None:
    """Plot Transformer training and validation MSE loss curves."""
    epochs_range = range(1, len(history["loss"]) + 1)
    tick_positions = [e for e in epochs_range if e == 1 or e % 10 == 0]

    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs_range,
        history["loss"],
        marker="o",
        linewidth=1.5,
        label="Training Loss")

    plt.plot(
        epochs_range,
        history["val_loss"],
        marker="s",
        linewidth=1.5,
        linestyle="--",
        label="Validation Loss")

    plt.xlabel("Epoch", fontsize=13, fontweight="bold")

    plt.ylabel("MSE Loss", fontsize=13, fontweight="bold")

    plt.xticks(
        ticks=tick_positions,
        labels=[str(e) for e in tick_positions],
        fontsize=11,
        fontweight="bold")

    plt.yticks(
        fontsize=11,
        fontweight="bold")

    plt.legend(prop={"weight": "bold", "size": 11})
    plt.grid(True, alpha=0.35)
    plt.tight_layout()

    if save:
        _save_current_fig(output_dir, "loss_transformer", dataset_label)

    plt.show()
    plt.close()

    print(f"Final train loss : {history['loss'][-1]:.6f}")
    print(f"Final val   loss : {history['val_loss'][-1]:.6f}")