# Time Series Analysis

Overview
- Methods for modeling data indexed by time to forecast, detect anomalies, and understand temporal patterns.

Important subtopics
- Stationarity, seasonality, trend decomposition
- ARIMA, SARIMA, exponential smoothing
- State-space models and Kalman filters
- Forecasting with RNNs / Transformers for long sequences

Key notes
- Always visualize series and check for missing values and seasonality.
- Use backtesting and rolling-window validation for realistic evaluation.

Quick example (forecasting)
- Fit an ARIMA model to monthly sales data and forecast the next 12 months.

Mermaid pipeline
```mermaid
flowchart LR
  A[Load time series] --> B[Visualize & decompose]
  B --> C[Stationarize / feature engineer]
  C --> D[Train model (ARIMA/NN)]
  D --> E[Forecast & evaluate]
```

Notes on image
- Add a time series plot and ACF/PACF plots: `images/ts_plots.png`.
