Stock Price Prediction with Linear Regression

This repository contains a stock price prediction and backtesting tool that uses historical stock market data from Yahoo Finance and applies a linear regression model with a rolling window approach to forecast next-day prices.

The project demonstrates machine learning for finance concepts including time-series modeling, backtesting, and error evaluation.

Features
Fetches 2 years of historical stock data using yfinance.
Implements sliding-window linear regression for next-day predictions.
Backtests predictions and calculates:
R² Score (goodness of fit)
Root Mean Square Error (RMSE)
Visualizes Actual vs Predicted prices with performance metrics.
Provides a next-day stock price prediction based on the latest data.
Works with any valid Yahoo Finance stock ticker (e.g., AAPL, MSFT, TSLA).

Installation
Clone this repository and install dependencies:
git clone https://github.com/vihanlalan/stock_price_prediction.git
cd stock-price-prediction
pip install -r requirements.txt


Or manually install:

pip install yfinance pandas numpy scikit-learn matplotlib

Usage
Run the script:
python stock_predictor.py
Enter a stock symbol when prompted:
Enter stock symbol (e.g., AAPL, MSFT, TSLA): AAPL


The script will:
Fetch 2 years of historical stock data.
Perform rolling backtests using linear regression.
Plot actual vs predicted stock prices.
Display performance metrics and a next-day prediction.

Example Output
Backtest Results for AAPL:
R² Score: 0.9234
Root Mean Square Error: $5.67
Last Close Price: $172.50
Next Day Prediction: $173.12



Project Structure
├── stock_predictor.py   # Main script
├── requirements.txt     # Dependencies
├── README.md            # Project documentation

How It Works
Uses a 60-day rolling window of closing prices.
Trains a Linear Regression model on each window to predict the next day’s price.
Compares predictions against actuals → calculates RMSE and R².
Visualizes performance and outputs the final next-day prediction.

Future Improvements
Add support for advanced models (LSTMs, Transformers).
Integrate real-time data streaming.
Deploy as a Flask/Django web app with dashboards.
Add automated hyperparameter tuning for better accuracy.


Contributions are welcome!
Fork the repo
Create a feature branch (git checkout -b feature-new)
Commit changes (git commit -m 'Added new feature')
Push and create a PR

Disclaimer
This project is for educational and research purposes only.
It is not financial advice and should not be used for actual trading decisions.

License
This project is licensed under the MIT License.
