# Realtime-Crypto-predict

## Overview

This repository contains scripts to fetch real-time or historical candle data of stocks, predict their prices for the next hour using a trained LSTM model, and visualize the predictions alongside actual prices. The main components of the project are:

- `update_candle_data()`: Fetches or updates candle data from a data source.
- `predict_prices_for_60_minutes()`: Uses a trained LSTM model to predict stock prices for the next 60 minutes.
- `plot_predictions()`: Plots actual vs predicted prices using Matplotlib.

## Requirements

- Python 3.x
- Dependencies listed in `requirements.txt`
- Trained LSTM model (`mod.h5`) for stock price prediction.
