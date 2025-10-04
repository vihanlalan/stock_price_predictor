import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

while True:
    stock_symbol = input("Enter stock symbol (e.g., AAPL, MSFT, TSLA): ").upper()
    try:``
        stock = yf.Ticker(stock_symbol)
        info = stock.info
        break
    except:
        print("Invalid stock symbol. Please try again.")

df = stock.history(period="2y")

if len(df) == 0:
    print(f"No data available for {stock_symbol}")
    exit()

def backtest(data, window_size=60):
    predictions = []
    actuals = []
    
    for i in range(window_size, len(data)):
        train_data = data.iloc[i-window_size:i]
        
        X_train = np.array(train_data['Close']).reshape(-1, 1)
        y_train = np.array(train_data['Close'].shift(-1).iloc[:-1]).reshape(-1, 1)
        X_train = X_train[:-1]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        next_day = np.array([[data['Close'].iloc[i]]])
        prediction = model.predict(next_day)[0][0]
        actual = data['Close'].iloc[i]
        
        predictions.append(prediction)
        actuals.append(actual)
    
    return predictions, actuals

predictions, actuals = backtest(df)

rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals))**2))
r2 = 1 - (np.sum((np.array(actuals) - np.array(predictions))**2) / 
          np.sum((np.array(actuals) - np.mean(actuals))**2))

plt.figure(figsize=(12,6))
plt.text(0.02, 0.98, f'R² = {r2:.4f}\nRMSE = ${rmse:.2f}', 
         transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8))
plt.plot(df.index[-len(predictions):], actuals, label='Actual Price')
plt.plot(df.index[-len(predictions):], predictions, label='Predicted Price')
plt.title(f'{stock_symbol} Stock Price Prediction - Backtest Results')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.show()

last_window = df['Close'].iloc[-60:].values.reshape(-1, 1)
last_target = df['Close'].shift(-1).iloc[-61:-1].values.reshape(-1, 1)
model = LinearRegression()
model.fit(last_window[:-1], last_target)
next_day_prediction = model.predict([[df['Close'].iloc[-1]]])

print(f"\nBacktest Results for {stock_symbol}:")
print(f"R² Score: {r2:.4f}")
print(f"Root Mean Square Error: ${rmse:.2f}")
print(f"Last Close Price: ${df['Close'].iloc[-1]:.2f}")
print(f"Next Day Prediction: ${next_day_prediction[0][0]:.2f}")