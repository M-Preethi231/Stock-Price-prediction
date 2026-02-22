import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Download stock data
ticker = "AAPL"
data = yf.download(ticker, start="2015-01-01", end="2024-01-01")

# Use closing price
data = data[['Close']]
data['Prediction'] = data['Close'].shift(-30)

# Prepare features & labels
X = np.array(data.drop(['Prediction'], axis=1))[:-30]
y = np.array(data['Prediction'])[:-30]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict next 30 days
future_prices = model.predict(X[-30:])

# Plot
plt.figure(figsize=(10,5))
plt.plot(data['Close'], label="Actual Price")
plt.plot(range(len(data)-30, len(data)), future_prices, label="Predicted Price", color="red")
plt.legend()
plt.title(f"{ticker} Stock Price Prediction")
plt.show()
