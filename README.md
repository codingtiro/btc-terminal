# BTC Historical Prediction Terminal

Research-only Streamlit app estimating 15-minute BTC Up/Down probabilities from completed Coinbase one-minute candles.

## Install and run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Set the actual strike/reference price and select the remaining-time bucket. Download 3, 7, 14, or 30 days of history. The app builds past 15-minute-window snapshots, estimates analog probabilities using volatility-normalized distance and momentum, and reports a chronological 70/30 holdout Brier score and directional accuracy against a base-rate benchmark. If there are too few comparable observations it withholds the estimate.

**Limitations:** This model uses Coinbase minute-close proxy strikes and outcomes, **not** Kalshi's CF Benchmarks official reference and settlement. The app cannot guarantee accuracy or profitability. No live orders, accounts, or trading signals. Data download requires internet access. Initial 14-day download makes many requests and may hit Coinbase rate limits; try 3 days first or rerun later. Snapshot horizons are 15, 12, 9, 6, and 3 minutes, not arbitrary seconds. Calibration and performance depend on the sample and market regime. The validation split is chronological, but the live estimate is fitted on all complete historical windows. Future improvements: official settlement data, streaming quotes, realistic backtests including fees and slippage, rolling walk-forward evaluation and calibration charts.

## Polymarket Chainlink addition
Click **Fetch live Chainlink BTC/USD quote** to obtain a best-effort snapshot from Polymarket RTDS. This requires network access and may fail when the endpoint changes or is unavailable. The official market rules must be checked for each contract. The historical model still uses Coinbase candles as proxy data and **is not validated against Chainlink settlement outcomes**. Enter the exact Polymarket Price to Beat manually. No live orders are supported.

## Authentic settlement comparison
The dashboard now includes an upload-driven comparison. Obtain actual Chainlink BTC/USD **60-second TWAP** reports (not instantaneous BTC/USD ticks and not reconstructed Coinbase averages) and official Polymarket resolved BTC 15-minute market records. Export two CSVs:
- `markets.csv`: `start_time,end_time,winner,strike` (UTC timestamps; winner Up or Down). Optional `twapLookbackSeconds` and `resolution_source` fields filter nonmatching contracts.
- `twap.csv`: `timestamp,price` (UTC timestamps and actual 60-second TWAP feed values).
Upload both at the bottom of the app. It aligns reports within 2 seconds before each boundary, excludes missing records, compares Coinbase proxy direction and TWAP direction against official outcomes, and evaluates the Coinbase-trained probability model chronologically on official labels if 100+ aligned windows are available. A TWAP-trained forecasting model requires continuous TWAP history and is not claimed here. Never assume an uploaded dataset is authentic without checking its provenance. The 2-second boundary tolerance is an analysis approximation, not an assertion about Polymarket's exact oracle adapter semantics.

Historical sources to investigate: https://pmdata.dev/chainlink-historical-feed-data and https://github.com/Ligengxin96/polymarket-data-samples . Provider access and data completeness vary; these sources are not bundled with this project.
