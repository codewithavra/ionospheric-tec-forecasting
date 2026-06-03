"""
Compiles and trains the LSTM model; returns the trained model and history.
"""
 
import numpy as np
import tensorflow as tf
 
from src.configs.config import (
    LSTM_EPOCHS, LSTM_BATCH,
    LSTM_OPTIMIZER, LSTM_LOSS, LSTM_SEED,
)
from src.models.lstm_model import build_lstm
 
 
def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = LSTM_EPOCHS,
    batch_size: int = LSTM_BATCH,
) -> tuple:
    """
    Train the LSTM model.
 
    Parameters
    ----------
    X_train, y_train : normalised training arrays, shape (N, 1440, 1)
    X_val,   y_val   : normalised validation arrays, shape (M, 1440, 1)
 
    Returns
    -------
    model   : trained Keras model
    history : Keras History object
    """
    tf.random.set_seed(LSTM_SEED)
    np.random.seed(LSTM_SEED)
 
    model = build_lstm()
    model.summary()
 
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )
    return model, history
