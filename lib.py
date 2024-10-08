# lstm_predictor.py (updated)

import numpy as np
import pandas as pd
import logging
import time
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Initialize logging
logging.basicConfig(level=logging.INFO)

candle_data = []
predicted_close_prices = []
scaler = MinMaxScaler(feature_range=(0, 1))
symbol = 'SOL-USD'
model = None


def load_lstm_model(model_path):
    global model
    try:
        model = load_model(model_path)
        logging.info(f"Model loaded from {model_path}")
        logging.info(f"Model input shape: {model.input_shape}")
        logging.info(f"Model output shape: {model.output_shape}")
        
        if not model.compiled_loss:
            model.compile(optimizer='adam', loss='mean_squared_error')
            logging.info("Model compiled with optimizer and loss.")
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")

def update_candle_data():
    global candle_data
    try:
        data = yf.download(tickers=symbol, period='1d', interval='1m')
        if not data.empty:
            candle_data = data[['Open', 'High', 'Low', 'Close']].values.tolist()
            logging.info(f"Updated candle_data: {candle_data[-5:]}")
            logging.info(f"Candle data length: {len(candle_data)}")
        else:
            logging.warning("No data received from yfinance.")
    except Exception as e:
        logging.error(f"Error fetching candle data: {e}")

def scale_candle_data(candle_data):
    global scaler
    scaled_data = scaler.fit_transform(candle_data)
    return scaled_data

def inverse_scale_data(predicted_data):
    global scaler
    reshaped_data = np.zeros((len(predicted_data), 4))
    reshaped_data[:, 3] = predicted_data.flatten()  # Filling 'Close' column
    inverse_scaled_data = scaler.inverse_transform(reshaped_data)
    return inverse_scaled_data[:, [0, 3]]  # Return Open and Close prices

def predict_prices_for_60_minutes():
    global candle_data, predicted_close_prices
    if len(candle_data) >= 50:
        try:
            scaled_candles = scale_candle_data(candle_data)
            predictions = []

            for i in range(60):
                data_input = np.array(scaled_candles[-50:]).reshape(1, 50, 4)
                predicted_price = model.predict(data_input)
                predictions.append(predicted_price[0][0])  # Close
                
                new_candle = np.zeros(4)
                new_candle[0] = predicted_price[0][0]  # Open
                new_candle[3] = predicted_price[0][0]  # Close
                new_candle[1] = max(new_candle[0], new_candle[3])  # High
                new_candle[2] = min(new_candle[0], new_candle[3])  # Low
                scaled_candles = np.vstack([scaled_candles, new_candle])
            
            predictions = np.array(predictions)
            inverse_predicted_prices = inverse_scale_data(predictions)
            predicted_close_prices = inverse_predicted_prices[:, 1].tolist()  # Close prices
            return predicted_close_prices
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return None
    else:
        logging.warning(f"Not enough data to predict. Current data length: {len(candle_data)}")
        return None

def plot_predictions():
    global candle_data, predicted_close_prices
    if len(predicted_close_prices) > 0:
        actual_close_prices = [candle[3] for candle in candle_data[-60:]]
        
        end_time = pd.Timestamp.now().floor('min')
        date_range = pd.date_range(end=end_time, periods=len(actual_close_prices) + len(predicted_close_prices), freq='min')
        
        plt.figure(figsize=(12, 6))
        plt.plot(date_range[:len(actual_close_prices)], actual_close_prices, label="Actual Close Prices", color='blue')
        plt.plot(date_range[len(actual_close_prices):], predicted_close_prices, label="Predicted Close Prices", color='red', linestyle='--')
        
        plt.title('Actual vs Predicted Close Prices for the Next 60 Minutes')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        logging.warning("Not enough predictions to plot.")

def plot_training_history(history):
    """Function to plot the training loss and validation loss over epochs."""
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.tight_layout()
    plt.show()
