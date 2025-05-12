# Realtime-Crypto-predict

## Overview

This repository contains a Python-based system to fetch **real-time** or **historical** candle data using the **yFinance API**, predict prices for the next 60 minutes using a trained **LSTM model**, and visualize the predicted vs actual prices.

The main use case is for tracking and forecasting **Solana (SOL-USD)** prices in near real-time, but the model can be adapted to other cryptocurrencies or stocks as well.

### Main Components

- `update_candle_data()`: Fetches and updates real-time candle data from Yahoo Finance using `yfinance`.
- `predict_prices_for_60_minutes()`: Loads a pre-trained LSTM model (`mod.h5`) to generate predictions for the next hour.
- `plot_predictions()`: Plots a graph comparing actual vs predicted prices using Matplotlib.

## Features

- Real-time data fetched via **yfinance** (`SOL-USD`)
- Price forecasting for the next **60 minutes** using **LSTM**
- Hourly auto-update and prediction cycle
- Clean visualization to compare predicted trends with actual market movement

## Requirements

Ensure you have the following libraries installed:

- Python 3.x
- `numpy`
- `pandas`
- `matplotlib`
- `keras`
- `scikit-learn`
- `yfinance`

Install with:

```bash
pip install -r requirements.txt
