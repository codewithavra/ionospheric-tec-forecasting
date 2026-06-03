"""
Compiles and trains the CNN-Transformer model with cosine-decay LR schedule.
"""
 
import numpy as np
import tensorflow as tf
 
from src.configs.config import (
    TRANS_EPOCHS, TRANS_BATCH,
    TRANS_LR_INIT, TRANS_LR_ALPHA, TRANS_SEED,
)
from src.models.transformer_model import build_cnn_transformer
 
 
def train_transformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = TRANS_EPOCHS,
    batch_size: int = TRANS_BATCH,
) -> tuple:
    """
    Train the CNN-Transformer model.
 
    Parameters
    ----------
    X_train, y_train : normalised training arrays, shape (N, 1440, 1)
    X_val,   y_val   : normalised validation arrays, shape (M, 1440, 1)
 
    Returns
    -------
    model   : trained Keras model
    history : Keras History object
    """
    tf.random.set_seed(TRANS_SEED)
    np.random.seed(TRANS_SEED)
 
    model = build_cnn_transformer()
    model.summary()
 
    steps_per_epoch = max(1, len(X_train) // batch_size)
    total_steps     = epochs * steps_per_epoch
 
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=TRANS_LR_INIT,
        decay_steps=total_steps,
        alpha=TRANS_LR_ALPHA,
    )
 
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="mse",
    )
 
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )
    return model, history
