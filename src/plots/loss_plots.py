"""
Training vs validation loss curves for LSTM and Transformer.
"""
 
import matplotlib.pyplot as plt
 
 
def plot_lstm_loss(history) -> None:
    """Plot LSTM training and validation MSE loss curves."""
    epochs_range = range(1, len(history["loss"]) + 1)
 
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
        epochs_range, 
        fontsize=11, 
        fontweight="bold")
    
    plt.yticks(
        fontsize=11, 
        fontweight="bold")
    
    plt.legend(prop={"weight": "bold", "size": 11})
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()
 
    print(f"Final train loss : {history['loss'][-1]:.6f}")
    print(f"Final val   loss : {history['val_loss'][-1]:.6f}")
 
 
def plot_transformer_loss(history) -> None:
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
    plt.show()
 
    print(f"Final train loss : {history['loss'][-1]:.6f}")
    print(f"Final val   loss : {history['val_loss'][-1]:.6f}")