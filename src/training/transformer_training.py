"""
Compiles and trains the CNN-Transformer model.
Uses linear warmup for the first TRANS_WARMUP_EPOCHS, then cosine decay.
"""
 
import numpy as np
import tensorflow as tf
 
from src.configs.config import (
    TRANS_EPOCHS, TRANS_BATCH,
    TRANS_LR_INIT, TRANS_LR_ALPHA,
    TRANS_WARMUP_EPOCHS, TRANS_SEED,
)
from src.models.transformer_model import build_cnn_transformer
 
 
# ── Warmup + Cosine Decay schedule ───────────────────────────────────────────
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warmup from near-zero to *peak_lr* over *warmup_steps*,
    then cosine decay down to *alpha* over the remaining steps.
 
    Why warmup?
    With only 29 training samples the model is very sensitive to the
    initial weight updates. Starting at full LR causes large, destabilising
    gradient steps in epoch 1. Warmup lets the model orient itself first.
    """
 
    def __init__(self, peak_lr: float, warmup_steps: int, total_steps: int, alpha: float):
        super().__init__()
        self.peak_lr       = peak_lr
        self.warmup_steps  = warmup_steps
        self.total_steps   = total_steps
        self.alpha         = alpha
 
    def __call__(self, step):
        step      = tf.cast(step, tf.float32)
        warmup    = self.peak_lr * (step / tf.cast(self.warmup_steps, tf.float32))
        cos_steps = tf.cast(self.total_steps - self.warmup_steps, tf.float32)
        cos_step  = step - tf.cast(self.warmup_steps, tf.float32)
        cosine    = self.alpha + 0.5 * (self.peak_lr - self.alpha) * (
            1 + tf.cos(np.pi * cos_step / cos_steps)
        )
        return tf.where(step < self.warmup_steps, warmup, cosine)
 
    def get_config(self):
        return dict(
            peak_lr=self.peak_lr,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            alpha=self.alpha,
        )
 
 
def train_transformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int     = TRANS_EPOCHS,
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
    warmup_steps    = TRANS_WARMUP_EPOCHS * steps_per_epoch
 
    lr_schedule = WarmupCosineDecay(
        peak_lr      = TRANS_LR_INIT,
        warmup_steps = warmup_steps,
        total_steps  = total_steps,
        alpha        = TRANS_LR_ALPHA,
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
