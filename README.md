# Ionospheric TEC Forecasting

This repository explores day-ahead forecasting of Total Electron Content (TEC) over the IISC Bangalore station using deep learning. The workflow compares a PyTorch-based LSTM against an encoder-only Transformer and evaluates the resulting forecasts through TEC plots, ionospheric delay analysis, and CSV export.

## Project goal

The ionosphere introduces a frequency-dependent range delay in GPS signals. Forecasting TEC one day ahead helps receivers and correction systems anticipate and compensate for this error before it occurs. This project trains models on daily TEC sequences and predicts the next-day TEC profile at 1-minute resolution.

## Forecast targets

The notebook workflow in [code.ipynb](code.ipynb) includes two experimental datasets:

- Dataset 1: Day 41 forecast for 18 June 2023
- Dataset 2: Day 40 forecast for 9 May 2024 and Day 41 forecast for 10 May 2024

## Data

- Source: TEC series derived from per-satellite GPS observation files and aggregated into daily CSV files
- Station: IISC Bangalore, India
- Resolution: 1 minute, giving 1440 samples per day
- Data layout:
  - [DataSet/DataSet1](DataSet/DataSet1)
  - [DataSet/DataSet2](DataSet/DataSet2)
- Preprocessing:
  - daily files are sorted into per-day CSVs in the sortedDataSet folders
  - each day is loaded as a 1440-point vector
  - input/target pairs are built as day $t$ → day $t+1$
  - training and validation splits are created from the full 41-day sequence

## Models

### LSTM

A two-layer PyTorch LSTM maps a full 1440-minute input day to the next day’s TEC profile in a single forward pass.

### Transformer

A lightweight encoder-only Transformer uses sinusoidal positional encoding and a warmup + cosine learning-rate schedule. It predicts the full next-day sequence without autoregressive decoding.

## Ionospheric delay conversion

Predicted TEC values are converted into ionospheric range delay using the thin-shell model:

$delay(m) = \frac{K_{iono} \cdot TECU}{f^2} \cdot TEC$

where $f$ is the GPS signal frequency. Delay is evaluated for L1, L2, and L5 in the plotting and analysis workflow.

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

Core packages include NumPy, Matplotlib, PyTorch, and TensorFlow.

## Repository structure

```text
src/
├── configs/config.py
├── models/
│   ├── lstm_model.py
│   └── transformer_model.py
├── preprocessing/
│   ├── dataset.py
│   └── sort_tec_data.py
├── plots/
│   ├── delay_plots.py
│   ├── loss_plots.py
│   └── prediction_plots.py
├── training/
│   ├── lstm_training.py
│   └── transformer_training.py
└── utils/
    ├── iono_delay.py
    └── save_results.py
```

## Usage

1. Place the raw TEC folders under [DataSet/DataSet1](DataSet/DataSet1) and [DataSet/DataSet2](DataSet/DataSet2).
2. Open [code.ipynb](code.ipynb) and run the cells from top to bottom.
3. The preprocessing cells generate sorted daily CSV files once; subsequent cells train the models, generate plots, and export delay results.
4. Output CSV tables are written into the sortedDataSet folders, and the notebook also produces prediction and delay plots for inspection.

## Outputs

The workflow produces:

- training and validation loss curves
- TEC forecast plots for the LSTM and Transformer
- ionospheric delay plots for L1, L2, and L5
- a summary CSV with actual and predicted TEC plus ionospheric delay values

## Attribution

If you reuse this project or substantial parts of it, please retain attribution to the repository authors.

## Contributors

- Avranil Dhar — [@codewithavra](https://github.com/codewithavra)
- Amlan Roy
- Anirban Roy — [@Anirban2718](https://github.com/Anirban2718)
- Srijita Roy — [@srijita004](https://github.com/srijita004)
- Kinnori Das — [@infjbytes](https://github.com/infjbytes)
- Somnath Banerjee
- Priyanshu Chakraborty
