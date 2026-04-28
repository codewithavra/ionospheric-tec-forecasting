# Ionospheric TEC Forecasting using Deep Learning

Predicts day-ahead Total Electron Content (TEC) from GNSS ground station 
data using LSTM and Transformer models.

## Project Structure
code.ipynb        # Full pipeline: preprocessing → training → evaluation

## Data
Raw GNSS data from IISC Bangalore receiver (not included in repo).
Place folders iisc1290_TECU to iisc1690_TECU inside:
  D:/ProjectMentor/TEC_data/

## How to Run
1. pip install -r requirements.txt
2. Update BASE_DIR in the notebook to your local data path
3. Run all cells top to bottom

## Models
- LSTM (2-layer, seq2seq)
- Transformer (3-layer encoder, sinusoidal positional encoding)

## Results
| Model       | RMSE (TECU) | MAE (TECU) |
|-------------|-------------|------------|
| LSTM        | TBD         | TBD        |
| Transformer | TBD         | TBD        |