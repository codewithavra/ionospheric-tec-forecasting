"""
Day-41 TEC prediction plots for LSTM and Transformer.
"""
 
import numpy as np
import matplotlib.pyplot as plt
 
_TICK_POS    = np.arange(0, 1441, 120)
_TICK_LABELS = [f"{h:02d}:00" for h in range(0, 25, 2)]
 
 
def plot_lstm_prediction(actual: np.ndarray, lstm_pred: np.ndarray) -> None:
    """Plot LSTM predicted vs actual TEC for Day 41."""
    minutes = np.arange(len(actual))
 
    plt.figure(figsize=(15, 5))
    plt.plot(minutes, actual,    color="steelblue", linewidth=1.4,
             label="Actual Day 41 (iisc1690)")
    plt.plot(minutes, lstm_pred, color="tomato",    linewidth=1.4,
             linestyle="--", label="LSTM Predicted Day 41")
 
    plt.xlabel("Time of Day (UTC)", fontsize=13, fontweight="bold")
    plt.ylabel("TEC (TECU)",        fontsize=13, fontweight="bold")
    plt.xticks(ticks=_TICK_POS, labels=_TICK_LABELS, rotation=45,
               fontsize=11, fontweight="bold")
    plt.yticks(fontsize=11, fontweight="bold")
    plt.legend(prop={"weight": "bold", "size": 11})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
 
    rmse = float(np.sqrt(np.mean((lstm_pred - actual) ** 2)))
    mae  = float(np.mean(np.abs(lstm_pred - actual)))
    print(f"LSTM  Day-41 RMSE : {rmse:.4f} TECU")
    print(f"LSTM  Day-41 MAE  : {mae:.4f} TECU")
    return rmse, mae
 
 
def plot_transformer_prediction(actual: np.ndarray, trans_pred: np.ndarray) -> None:
    """Plot Transformer predicted vs actual TEC for Day 41."""
    minutes = np.arange(len(actual))
 
    plt.figure(figsize=(15, 5))
    plt.plot(minutes, actual,     color="steelblue",  linewidth=1.4,
             label="Actual Day 41 (iisc1690)")
    plt.plot(minutes, trans_pred, color="darkorange",  linewidth=1.4,
             linestyle="--", label="Transformer Predicted Day 41")
 
    plt.xlabel("Time of Day (UTC)", fontsize=13, fontweight="bold")
    plt.ylabel("TEC (TECU)",        fontsize=13, fontweight="bold")
    plt.xticks(ticks=_TICK_POS, labels=_TICK_LABELS, rotation=45,
               fontsize=11, fontweight="bold")
    plt.yticks(fontsize=11, fontweight="bold")
    plt.legend(prop={"weight": "bold", "size": 11})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
 
    rmse = float(np.sqrt(np.mean((trans_pred - actual) ** 2)))
    mae  = float(np.mean(np.abs(trans_pred - actual)))
    print(f"Transformer Day-41 RMSE : {rmse:.4f} TECU")
    print(f"Transformer Day-41 MAE  : {mae:.4f} TECU")
    return rmse, mae
