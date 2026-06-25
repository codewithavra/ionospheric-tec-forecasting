"""
Compiles and trains the LSTM model; returns the trained model and history.
"""
 
# import numpy as np
# import tensorflow as tf
 
# from src.configs.config import (
#     LSTM_EPOCHS, LSTM_BATCH,
#     LSTM_OPTIMIZER, LSTM_LOSS, LSTM_SEED,
# )
# from src.models.lstm_model import build_lstm
 
 
# def train_lstm(
#     X_train: np.ndarray,
#     y_train: np.ndarray,
#     X_val: np.ndarray,
#     y_val: np.ndarray,
#     epochs: int = LSTM_EPOCHS,
#     batch_size: int = LSTM_BATCH,
# ) -> tuple:

    # """
    # Train the LSTM model.
 
    # Parameters
    # ----------
    # X_train, y_train : normalised training arrays, shape (N, 1440, 1)
    # X_val,   y_val   : normalised validation arrays, shape (M, 1440, 1)
 
    # Returns
    # -------
    # model   : trained Keras model
    # history : Keras History object
    # """
    # tf.random.set_seed(LSTM_SEED)
    # np.random.seed(LSTM_SEED)
 
    # model = build_lstm()
    # model.summary()
 
    # history = model.fit(
    #     X_train, y_train,
    #     validation_data=(X_val, y_val),
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     verbose=1,
    # )
    # return model, history
    
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from src.configs.config import LSTM_EPOCHS, LSTM_BATCH, LSTM_SEED
from src.models.lstm_model import build_lstm

def train_lstm(X_train, y_train, X_val, y_val, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH):
    torch.manual_seed(LSTM_SEED)
    np.random.seed(LSTM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # numpy → tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_v  = torch.tensor(X_val,   dtype=torch.float32)
    y_v  = torch.tensor(y_val,   dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    model = build_lstm().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    history = {"loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v.to(device)), y_v.to(device)).item()

        history["loss"].append(epoch_loss / len(loader))
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1}/{epochs} — loss: {history['loss'][-1]:.6f} — val_loss: {val_loss:.6f}")

    return model, history
