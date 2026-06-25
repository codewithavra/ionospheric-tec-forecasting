"""
Trains the lightweight Encoder-only Transformer for day-ahead TEC forecasting.

Training strategy
─────────────────
• Single forward pass  — encoder sees today's full 1440-step sequence and
  predicts all 1440 steps of tomorrow in one shot. No teacher forcing, no
  autoregressive loop, no decoder. Identical call signature to the LSTM
  training function.

• Warmup + Cosine LR schedule (per-batch stepping via PyTorch LambdaLR).

• Gradient clipping (max_norm=1.0) for transformer stability.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

from src.configs.config import (
    TRANS_EPOCHS, TRANS_BATCH,
    TRANS_LR_INIT, TRANS_LR_ALPHA,
    TRANS_WARMUP_EPOCHS, TRANS_SEED,
)
from src.models.transformer_model import build_cnn_transformer


# ── Warmup + Cosine Decay LR schedule ────────────────────────────────────────

def get_warmup_cosine_schedule(
    optimizer    : torch.optim.Optimizer,
    warmup_steps : int,
    total_steps  : int,
    peak_lr      : float,
    alpha        : float,
) -> LambdaLR:
    """
    Linear warmup 0 → peak_lr over warmup_steps,
    then cosine decay peak_lr → alpha over remaining steps.
    LambdaLR multiplies the base LR by the returned scalar.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor  = alpha / peak_lr
        return floor + (1.0 - floor) * cosine

    return LambdaLR(optimizer, lr_lambda)


# ── Training function ─────────────────────────────────────────────────────────

def train_transformer(
    X_train    : np.ndarray,
    y_train    : np.ndarray,
    X_val      : np.ndarray,
    y_val      : np.ndarray,
    epochs     : int = TRANS_EPOCHS,
    batch_size : int = TRANS_BATCH,
) -> tuple:
    """
    Train the encoder-only Transformer.

    Parameters
    ----------
    X_train, y_train : normalised arrays, shape (N, 1440, 1)
    X_val,   y_val   : normalised arrays, shape (M, 1440, 1)

    Returns
    -------
    model   : trained TECTransformerEncoder (nn.Module)
    history : dict with keys "loss" and "val_loss" (lists of floats)
    """
    # Reproducibility
    torch.manual_seed(TRANS_SEED)
    np.random.seed(TRANS_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Transformer] Training on: {device}")

    # numpy → tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_v  = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_v  = torch.tensor(y_val,   dtype=torch.float32).to(device)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size = batch_size,
        shuffle    = True,
    )

    # Model, optimiser, loss
    model     = build_cnn_transformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRANS_LR_INIT)
    criterion = nn.MSELoss()
    print(model)

    # LR schedule (stepped every batch)
    steps_per_epoch = max(1, len(X_train) // batch_size)
    total_steps     = epochs * steps_per_epoch
    warmup_steps    = TRANS_WARMUP_EPOCHS * steps_per_epoch

    scheduler = get_warmup_cosine_schedule(
        optimizer,
        warmup_steps = warmup_steps,
        total_steps  = total_steps,
        peak_lr      = TRANS_LR_INIT,
        alpha        = TRANS_LR_ALPHA,
    )

    # Training loop
    history = {"loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(device)   # (B, 1440, 1)
            yb = yb.to(device)   # (B, 1440, 1)

            optimizer.zero_grad()
            pred = model(xb)                # single forward pass — no decoder
            loss = criterion(pred, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)           # same single forward pass
            val_loss = criterion(val_pred, y_v).item()

        avg_train = epoch_loss / len(loader)
        history["loss"].append(avg_train)
        history["val_loss"].append(val_loss)

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"loss: {avg_train:.6f} | "
            f"val_loss: {val_loss:.6f} | "
            f"lr: {current_lr:.2e}"
        )

    return model, history