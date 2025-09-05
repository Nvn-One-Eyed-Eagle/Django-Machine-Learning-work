import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import io, base64
import os
from django.conf import settings
from keras.models import load_model

MODEL_PATH = os.path.join(settings.BASE_DIR, "StockPredictionPortal", "ml", "stock_prediction_model.keras")
model = load_model(MODEL_PATH)

def get_stock_plot():
    now = datetime.now()
    start = datetime(now.year-10, now.month, now.day)
    end = now

    ticker = "AAPL"
    df = yf.download(ticker, start, end).reset_index()
    df["MA_100"] = df["Close"].rolling(100).mean()

    data_training = pd.DataFrame(df["Close"][0:int(len(df)*0.7)])
    data_testing = pd.DataFrame(df["Close"][int(len(df)*0.7):])

    scaler = MinMaxScaler(feature_range=(0,1))
    data_training_array = scaler.fit_transform(data_training)

    x_train, y_train = [], []
    for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i-100:i])
        y_train.append(data_training_array[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    # load model
    model = load_model("StockPredictionPortal/ml/stock_prediction_model.keras")

    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.transform(final_df)

    x_test, y_test = [], []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)

    y_predicted = model.predict(x_test)
    y_predicted = scaler.inverse_transform(y_predicted.reshape(-1,1)).flatten()
    y_test = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    # Plot into memory
    plt.figure(figsize=(10,5))
    plt.plot(y_test, "b", label="Original Price")
    plt.plot(y_predicted, "r", label="Predicted Price")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # Encode to base64 for HTML
    graphic = base64.b64encode(image_png).decode('utf-8')
    return graphic
