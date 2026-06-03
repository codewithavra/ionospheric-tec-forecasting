"""
Physical conversion utilities:
  - TEC (TECU) → ionospheric range delay (metres) at any GPS frequency
  - Thin-shell mapping function  M(E) for slant-path correction
  - Convenience functions for L1, L2, L5 vertical delays
"""
 
import numpy as np
 
from src.configs.config import (
    F_L1, F_L2, F_L5,
    K_IONO, TECU,
    R_EARTH, H_ION,
)
 
# Pre-computed scale factors (metres per TECU)
ALPHA_L1 = K_IONO * TECU / (F_L1 ** 2)
ALPHA_L2 = K_IONO * TECU / (F_L2 ** 2)
ALPHA_L5 = K_IONO * TECU / (F_L5 ** 2)
 
 
def mapping_function(elevation_deg: float | np.ndarray) -> np.ndarray:
    """
    Thin-shell single-layer mapping function M(E).
 
    Parameters
    ----------
    elevation_deg : satellite elevation angle in degrees (scalar or array)
 
    Returns
    -------
    M(E) : obliquity factor (≥ 1)
    """
    elev_rad = np.deg2rad(elevation_deg)
    sin_z_pp = (R_EARTH / (R_EARTH + H_ION)) * np.cos(elev_rad)
    sin_z_pp = np.clip(sin_z_pp, -1.0, 1.0)
    return 1.0 / np.sqrt(1.0 - sin_z_pp ** 2)
 
 
def tec_to_iono_delay(
    tec_tecu: np.ndarray,
    frequency: float = F_L1,
    elevation_deg: float | None = None,
) -> np.ndarray:
    """
    Convert TEC to ionospheric range delay.
 
    Parameters
    ----------
    tec_tecu      : TEC in TECU (array)
    frequency     : signal frequency in Hz  (default: GPS L1)
    elevation_deg : satellite elevation angle in degrees;
                    if None, vertical (nadir) delay is returned
 
    Returns
    -------
    delay in metres (same shape as *tec_tecu*)
    """
    alpha   = K_IONO * TECU / (frequency ** 2)
    delay_v = alpha * tec_tecu
    if elevation_deg is not None:
        return delay_v * mapping_function(elevation_deg)
    return delay_v
 
 
def compute_all_delays(
    actual: np.ndarray,
    lstm_pred: np.ndarray,
    trans_pred: np.ndarray,
    frequency: float = F_L1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch helper — convert actual + both predictions to delay at *frequency*.
 
    Returns (iono_actual, iono_lstm, iono_trans)
    """
    return (
        tec_to_iono_delay(actual,      frequency=frequency),
        tec_to_iono_delay(lstm_pred,   frequency=frequency),
        tec_to_iono_delay(trans_pred,  frequency=frequency),
    )
 
 
def delay_metrics(
    name: str,
    predicted: np.ndarray,
    actual: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    Compute RMSE, MAE, and Bias for a predicted delay array.
 
    Returns a dict with keys: name, rmse, mae, bias
    """
    err  = predicted - actual
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    if verbose:
        print(
            f"{name:<15}  RMSE = {rmse:.5f} m   "
            f"MAE = {mae:.5f} m   Bias = {bias:+.5f} m"
        )
    return dict(name=name, rmse=rmse, mae=mae, bias=bias)
