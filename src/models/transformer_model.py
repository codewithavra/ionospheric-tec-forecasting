"""
Lightweight Pure Transformer Encoder for day-ahead TEC forecasting (PyTorch).

Architecture
────────────
Input      : (B, 1440, 1)  — today's normalised TEC, one scalar per minute
Embedding  : Linear(1 → d_model) + sinusoidal positional encoding
Encoder    : N × TransformerEncoderLayer  (self-attention + FFN, Pre-LN)
Output     : Linear(d_model → 1)  →  (B, 1440, 1)  — tomorrow's TEC

Why encoder-only?
─────────────────
The task is full-day → full-day regression, NOT token generation.
An encoder reads the entire source sequence and maps each position to
an output — all 1440 predictions are produced in a single forward pass.
This is both faster to train and faster at inference than an
encoder-decoder with autoregressive decoding.

Key changes vs old design
─────────────────────────
• Removed decoder stack  → no autoregressive loop, no teacher forcing
• Halved d_model (64), num_heads (4), ff_dim (128), num_layers (2)
  → attention matrix is still 1440×1440 but with far fewer parameters
• Removed TRANS_WINDOW_SIZE — no longer needed
• Pre-LayerNorm (norm_first=True) — more stable with small datasets
"""

import math
import torch
import torch.nn as nn

from src.configs.config import (
    TRANS_D_MODEL, TRANS_NUM_HEADS, TRANS_FF_DIM,
    TRANS_NUM_LAYERS, TRANS_DROPOUT, MINUTES_PER_DAY,
)


# ── Sinusoidal positional encoding ────────────────────────────────────────────

def positional_encoding(length: int, d_model: int) -> torch.Tensor:
    """
    Returns a (1, length, d_model) sinusoidal PE tensor.
    Registered as a buffer — moves to GPU automatically with the model.
    """
    pe       = torch.zeros(length, d_model)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)   # (L, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
    )                                                                      # (d_model/2,)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)                                                 # (1, L, d_model)


# ── Main model ────────────────────────────────────────────────────────────────

class TECTransformerEncoder(nn.Module):
    """
    Encoder-only Transformer for minute-wise TEC sequence regression.

    Parameters
    ----------
    seq_len    : number of time steps (1440 = minutes per day)
    d_model    : embedding dimension  (keep small — 64 is plenty)
    num_heads  : attention heads      (must divide d_model evenly)
    ff_dim     : FFN hidden dimension (2–4× d_model recommended)
    num_layers : number of stacked encoder layers
    dropout    : dropout probability
    """

    def __init__(
        self,
        seq_len   : int   = MINUTES_PER_DAY,
        d_model   : int   = TRANS_D_MODEL,
        num_heads : int   = TRANS_NUM_HEADS,
        ff_dim    : int   = TRANS_FF_DIM,
        num_layers: int   = TRANS_NUM_LAYERS,
        dropout   : float = TRANS_DROPOUT,
    ):
        super().__init__()
        self.d_model = d_model

        # Input projection: scalar TEC → d_model
        self.input_proj = nn.Linear(1, d_model)

        # Fixed positional encoding (buffer → auto GPU transfer)
        self.register_buffer("pos_enc", positional_encoding(seq_len, d_model))

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = num_heads,
            dim_feedforward = ff_dim,
            dropout        = dropout,
            activation     = "gelu",
            batch_first    = True,    # (B, L, d_model) — not (L, B, d_model)
            norm_first     = True,    # Pre-LN: more stable with small datasets
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers,
            norm       = nn.LayerNorm(d_model),
        )

        # Output projection: d_model → scalar TEC
        self.output_proj = nn.Linear(d_model, 1)

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for all Linear layers, zero bias."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1440, 1)  — today's normalised TEC sequence

        Returns
        -------
        (B, 1440, 1)  — predicted TEC for every minute of tomorrow
        """
        # Embed + positional encoding + dropout
        x = self.dropout(
            self.input_proj(x) + self.pos_enc[:, :x.size(1), :]
        )                                   # (B, 1440, d_model)

        x = self.encoder(x)                 # (B, 1440, d_model)
        return self.output_proj(x)          # (B, 1440, 1)


# ── Factory function ──────────────────────────────────────────────────────────

def build_cnn_transformer(
    seq_len   : int   = MINUTES_PER_DAY,
    d_model   : int   = TRANS_D_MODEL,
    num_heads : int   = TRANS_NUM_HEADS,
    ff_dim    : int   = TRANS_FF_DIM,
    num_layers: int   = TRANS_NUM_LAYERS,
    dropout   : float = TRANS_DROPOUT,
) -> TECTransformerEncoder:
    """
    Build and return the lightweight encoder-only Transformer.
    Function name kept as build_cnn_transformer so nothing else
    in the project needs to change.
    """
    return TECTransformerEncoder(
        seq_len    = seq_len,
        d_model    = d_model,
        num_heads  = num_heads,
        ff_dim     = ff_dim,
        num_layers = num_layers,
        dropout    = dropout,
    )