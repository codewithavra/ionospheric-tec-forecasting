"""
Sequence-to-sequence LSTM for day-ahead TEC forecasting.
"""
 
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed
 
# from src.configs.config import (
#     LSTM_UNITS_1, LSTM_UNITS_2, MINUTES_PER_DAY,
# )
 
 
# def build_lstm(
#     input_length: int = MINUTES_PER_DAY,
#     units_1: int = LSTM_UNITS_1,
#     units_2: int = LSTM_UNITS_2,
# ) -> Sequential:
#     """
#     Build and return a compiled LSTM model.
 
#     Architecture:
#         Input (1440, 1)
#         → LSTM(units_1, return_sequences=True)
#         → LSTM(units_2, return_sequences=True)
#         → TimeDistributed Dense(1)
#     """
#     model = Sequential([
#         Input(shape=(input_length, 1)),
#         LSTM(units_1, return_sequences=True),
#         LSTM(units_2, return_sequences=True),
#         TimeDistributed(Dense(1)),
#     ], name="LSTM_TEC_Model")
 
#     model.compile(optimizer="adam", loss="mse")
#     return model
import torch
import torch.nn as nn
from src.configs.config import LSTM_UNITS_1, LSTM_UNITS_2, MINUTES_PER_DAY

class LSTMTECModel(nn.Module):
    def __init__(self, input_length=MINUTES_PER_DAY, units_1=LSTM_UNITS_1, units_2=LSTM_UNITS_2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=units_1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=units_1, hidden_size=units_2, batch_first=True)
        self.fc = nn.Linear(units_2, 1)

    def forward(self, x):           # x: (B, 1440, 1)
        x, _ = self.lstm1(x)        # (B, 1440, units_1)
        x, _ = self.lstm2(x)        # (B, 1440, units_2)
        return self.fc(x)           # (B, 1440, 1)

def build_lstm():
    return LSTMTECModel()