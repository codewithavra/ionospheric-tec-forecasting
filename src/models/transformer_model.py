"""
Hybrid CNN + Windowed-Transformer model for day-ahead TEC forecasting.
 
Architecture
    Stage 1 : Multi-scale Conv1D (kernel 3 / 15 / 61) — spike + trend features
    Stage 2 : Linear projection to d_model + sinusoidal positional encoding
    Stage 3 : Stacked LocalAttentionEncoder (windowed MHA + FFN)
    Stage 4 : One GlobalAttentionEncoder — reconciles local features with full-day arc
    Stage 5 : Dense(1) output projection
"""
 
import tensorflow as tf
from tensorflow.keras import layers, Model
 
from src.configs.config import (
    TRANS_D_MODEL, TRANS_NUM_HEADS, TRANS_FF_DIM,
    TRANS_NUM_LAYERS, TRANS_WINDOW_SIZE, TRANS_DROPOUT,
    MINUTES_PER_DAY,
)
 
 
# ── Sinusoidal positional encoding ────────────────────────────────────────────
def positional_encoding(length: int, d_model: int) -> tf.Tensor:
    """Return a (1, length, d_model) sinusoidal positional encoding tensor."""
    positions   = tf.range(length, dtype=tf.float32)[:, tf.newaxis]
    dims        = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
    angle_rates = 1 / tf.pow(10000.0, (2 * (dims // 2)) / tf.cast(d_model, tf.float32))
    angle_rads  = positions * angle_rates
    sines       = tf.sin(angle_rads[:, 0::2])
    cosines     = tf.cos(angle_rads[:, 1::2])
    return tf.concat([sines, cosines], axis=-1)[tf.newaxis, ...]   # (1, L, d_model)
 
 
# ── Windowed self-attention encoder (local, fine-grained) ────────────────────
class LocalAttentionEncoder(layers.Layer):
    """
    Splits (B, L, d_model) into non-overlapping windows of *window_size*,
    runs MHA within each window, then restores (B, L, d_model).
    Captures short-range spike structure within each hour-long window.
    """
 
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        window_size: int = 60,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.d_model     = d_model
 
        self.mha   = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )
        self.ffn1  = layers.Dense(ff_dim, activation="gelu")
        self.ffn2  = layers.Dense(d_model)
        self.drop  = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
 
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        B  = tf.shape(x)[0]
        W  = self.window_size
        nW = tf.shape(x)[1] // W
 
        x_win = tf.reshape(x, [B, nW, W, self.d_model])
        x_win = tf.reshape(x_win, [B * nW, W, self.d_model])
 
        attn  = self.mha(x_win, x_win, training=training)
        x_win = self.norm1(x_win + self.drop(attn, training=training))
 
        ffn   = self.ffn2(self.ffn1(x_win))
        x_win = self.norm2(x_win + self.drop(ffn, training=training))
 
        return tf.reshape(x_win, [B, nW * W, self.d_model])
 
 
# ── Global attention encoder (full-sequence, coarse) ─────────────────────────
class GlobalAttentionEncoder(layers.Layer):
    """
    Standard transformer encoder block operating on the full sequence.
    Runs after all local layers — lets the model reconcile the diurnal arc
    (broad shape, evening decay) with the local spike features.
    Uses strided average-pooling to reduce the 1440-step sequence to
    *pool_factor* steps before attention, then upsamples back.
    This keeps the attention matrix manageable (144×144 instead of 1440×1440).
    """
 
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        pool_factor: int = 10,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pool_factor = pool_factor
        self.d_model     = d_model
 
        self.pool  = layers.AveragePooling1D(pool_size=pool_factor, strides=pool_factor, padding="same")
        self.mha   = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )
        self.ffn1  = layers.Dense(ff_dim, activation="gelu")
        self.ffn2  = layers.Dense(d_model)
        self.drop  = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        # Project pooled output back to d_model after upsampling
        self.proj  = layers.Dense(d_model)
 
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Downsample: (B, 1440, d) → (B, 144, d)
        x_pooled = self.pool(x)
 
        # Full-sequence attention on the compressed representation
        attn     = self.mha(x_pooled, x_pooled, training=training)
        x_pooled = self.norm1(x_pooled + self.drop(attn, training=training))
 
        ffn      = self.ffn2(self.ffn1(x_pooled))
        x_pooled = self.norm2(x_pooled + self.drop(ffn, training=training))
 
        # Upsample back: repeat each pooled step pool_factor times → (B, 1440, d)
        x_up = tf.repeat(x_pooled, repeats=self.pool_factor, axis=1)
        # Trim or pad to exactly match original length
        x_up = x_up[:, :tf.shape(x)[1], :]
 
        # Residual add with a projection (x_up may differ from x in scale)
        return x + self.proj(x_up)
 
 
# ── Model builder ─────────────────────────────────────────────────────────────
def build_cnn_transformer(
    input_shape: tuple = (MINUTES_PER_DAY, 1),
    d_model: int       = TRANS_D_MODEL,
    num_heads: int     = TRANS_NUM_HEADS,
    ff_dim: int        = TRANS_FF_DIM,
    num_layers: int    = TRANS_NUM_LAYERS,
    window_size: int   = TRANS_WINDOW_SIZE,
    dropout: float     = TRANS_DROPOUT,
) -> Model:
    """Build and return the improved CNN + Transformer model (not yet compiled)."""
    inputs = layers.Input(shape=input_shape)
 
    # Stage 1: Multi-scale CNN — extract spike, hourly, and trend features in parallel
    c3  = layers.Conv1D(32, kernel_size=3,  padding="same", activation="relu")(inputs)
    c15 = layers.Conv1D(32, kernel_size=15, padding="same", activation="relu")(inputs)
    c61 = layers.Conv1D(32, kernel_size=61, padding="same", activation="relu")(inputs)
    x   = layers.Concatenate()([c3, c15, c61])     # (B, 1440, 96)
    x   = layers.LayerNormalization()(x)
 
    # Stage 2: Project to d_model + positional encoding
    x = layers.Dense(d_model)(x)                   # (B, 1440, d_model)
    x = x + positional_encoding(input_shape[0], d_model)
 
    # Stage 3: Stacked local windowed attention layers
    for _ in range(num_layers):
        x = LocalAttentionEncoder(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            window_size=window_size,
            dropout=dropout,
        )(x)
 
    # Stage 4: One global attention pass — fixes evening decay & broad transitions
    x = GlobalAttentionEncoder(
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        pool_factor=10,        # 1440 → 144 steps for attention
        dropout=dropout,
    )(x)
 
    # Stage 5: Output projection
    outputs = layers.Dense(1)(x)
 
    return Model(inputs, outputs, name="CNN_Transformer_TEC_v2")
