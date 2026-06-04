# Ionospheric TEC Forecasting
```text
Day-ahead forecasting of Total Electron Content (TEC) over the IISC Bangalore station using deep learning. Two models are implemented and compared — a sequence-to-sequence LSTM and a hybrid CNN + Windowed Transformer — with downstream conversion of TEC predictions to ionospheric range delay at GPS frequencies L1, L2, and L5.
```
---
# Problem Statement
```text
The ionosphere introduces a frequency-dependent delay in GPS signals that must be corrected for accurate positioning. TEC is the quantity that determines this delay. Forecasting TEC one day ahead allows receivers and correction systems to anticipate and compensate for ionospheric error before it occurs.
This project trains models on 40 days of TEC observations from the IISC Bangalore station (GPS DOY 1290–1690) and predicts the TEC profile for day 41 (iisc1690) at 1-minute resolution.
```
---
# Dataset
- Source: RINEX observation files processed into per-satellite CSV files for 32 GPS satellites (G01–G32)
- Station: IISC Bangalore, India
- Resolution: 1 minute (1440 samples per day)
- Total days: 41
- TEC value per minute: maximum across all visible satellites
- Train split: Days 2–30 (29 input→target pairs)
- Validation split: Days 31–41 (11 pairs)
---
# Models

## LSTM
```text
A two-layer sequence-to-sequence LSTM that maps a full 1440-minute input day to the next day's TEC profile.
```
```
Input (1440, 1)
→ LSTM(64, return_sequences=True)
→ LSTM(32, return_sequences=True)
→ TimeDistributed Dense(1)
```
## CNN + Windowed Transformer
```text
A hybrid model that uses multi-scale convolutions to extract spike and trend features, followed by stacked windowed local attention layers and a final global attention pass for full-day context.
```
```
Input (1440, 1)
→ Parallel Conv1D (kernel 3 / 15 / 61) → Concatenate → LayerNorm
→ Dense projection to d_model + Sinusoidal Positional Encoding
→ 4× LocalAttentionEncoder (60-minute windows)
→ 1× GlobalAttentionEncoder (pooled full-sequence attention)
→ Dense(1)
```
---
# Ionospheric Delay Conversion
```text
Predicted TEC is converted to ionospheric range delay using the thin-shell model:
```
>delay (m) = (40.308 × 10¹⁶ / f²) × TEC_TECU
```text
where f is the signal frequency. For slant paths, a mapping function M(E) based on satellite elevation angle is applied. Delay is computed at three GPS frequencies:
```
| Signal | Frequency |
|--------|-----------|
| L1 | 1575.42 MHz |
| L2 | 1227.60 MHz |
| L5 | 1176.45 MHz |
---
## Requirements
 
```
tensorflow >= 2.12
numpy
matplotlib
```

---
## Directory Structure
```
src/
├── __init__.py
├── configs/
│   └── config.py               ← ALL paths + hyperparameters in one place
├── preprocessing/
│   ├── dataset.py              ← daily matrix, train/val splits, normalisation
│   └── sort_tec_data.py        ← raw CSV discovery, minute-max aggregation
├── models/
│   ├── lstm_model.py           ← build_lstm()
│   └── transformer_model.py   ← build_cnn_transformer(), LocalAttentionEncoder
├── training/
│   ├── lstm_training.py           ← train_lstm()
│   └── transformer_training.py   ← train_transformer() + cosine LR schedule
├── plots/
│   ├── loss_plots.py           ← plot_lstm_loss(), plot_transformer_loss()
│   ├── prediction_plots.py    ← plot_lstm_prediction(), plot_transformer_prediction()
│   └── delay_plots.py         ← per-frequency delay plots + diurnal profile
└── utils/
    ├── iono_delay.py           ← mapping_function(), tec_to_iono_delay(), delay_metrics()
    └── save_results.py         ← save_delay_csv()
```
---
## Usage
 
Set the data path in `src/configs/config.py`:
 
```python
BASE_DIR = Path.cwd() / "TEC_data"
```
 
Then run `code.ipynb` top to bottom. The sorted dataset only needs to be built once (Cell 2). All subsequent runs can skip that cell.
 
The final output CSV (`iisc1690_ionospheric_delay.csv`) is written to `TEC_data/sortedDataSet/` and contains minute-wise actual and predicted TEC values alongside L1 ionospheric delay for both models.

---
## Attribution

If you use this project or significant portions of its code, please retain the original copyright notice and provide appropriate attribution to the authors.

---
## Contributors

- Avranil Dhar (@codewithavra)
- Amlan Roy
- Anirban Roy (@Anirban2718)
- Srijita Roy
- Kinnori Das
- Somenath
- Priyanshu Chakraborty