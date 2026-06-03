"""
Ionospheric delay plots:
  - per-model, per-frequency full-day traces
  - diurnal hourly profile comparison
"""
 
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
 
from src.configs.config import F_L1, F_L2, F_L5
from src.utils.iono_delay import tec_to_iono_delay

 
_TICK_POS    = np.arange(0, 1441, 120)
_TICK_LABELS = [f"{h:02d}:00" for h in range(0, 25, 2)]
 
 
def _apply_ax_style(ax, tick_pos, tick_labels, ylabel="Iono. Delay (m)"):
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time of Day (UTC)", fontsize=13, fontweight="bold")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.legend(prop={"weight": "bold", "size": 10})
    ax.grid(True, alpha=0.3)
 
 
# ── LSTM delay plots ──────────────────────────────────────────────────────────

def plot_lstm_delay(actual: np.ndarray, lstm_pred: np.ndarray, freq : Literal ["F_L1", "F_L2" , "F_L5"] ) -> None:
    
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
    plt.tight_layout(); plt.show()
    rmse = np.sqrt(np.mean((iono_lstm - iono_actual) ** 2))
    mae  = np.mean(np.abs(iono_lstm - iono_actual))
    print(f"LSTM {freq} — RMSE: {rmse:.5f} m   MAE: {mae:.5f} m")
 
 
# ── Transformer delay plots ───────────────────────────────────────────────────
 
def plot_transformer_delay(actual: np.ndarray, trans_pred: np.ndarray ,freq : Literal ["F_L1", "F_L2" , "F_L5"]) -> None:
    
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
    plt.tight_layout(); plt.show()
    rmse = np.sqrt(np.mean((iono_trans - iono_actual) ** 2))
    mae  = np.mean(np.abs(iono_trans - iono_actual))
    print(f"Transformer {freq} — RMSE: {rmse:.5f} m   MAE: {mae:.5f} m")
 
 
# ── Diurnal hourly profile ────────────────────────────────────────────────────
 
def plot_diurnal_profile(
    actual: np.ndarray,
    lstm_pred: np.ndarray,
    trans_pred: np.ndarray,
    frequency: float = F_L1,
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
 
    ax.set_xlabel("Hour (UTC)",       fontsize=13, fontweight="bold")
    ax.set_ylabel("Iono. Delay (m)",  fontsize=13, fontweight="bold")
    ax.set_xticks(hours)
    ax.tick_params(axis="both", labelsize=11)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.legend(prop={"weight": "bold", "size": 11})
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
