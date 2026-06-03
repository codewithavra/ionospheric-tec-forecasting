# Ionospheric TEC Forecasting

Forecasting Ionospheric Total Electron Content (TEC) using machine learning and deep learning techniques.

## Contributors

- Avranil Dhar (@codewithavra)
- Amlan Roy
- Anirban Roy (@Anirban2718)
- Srijita Roy
- Kinnori Das
- Somenath
- Priyanshu Chakraborty

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
## Attribution

If you use this project or significant portions of its code, please retain the original copyright notice and provide appropriate attribution to the authors.