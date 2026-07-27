import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP("Stock Analysis & ML Server")

@mcp.tool()
def get_stock_data(ticker: str, period: str = "6mo") -> str:
    """Fetches historical stock market price data and key indicators for a given ticker."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.empty:
        return f"No data found for ticker {ticker}"
    
    # Calculate Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    latest = df.iloc[-1]
    info = {
        "Ticker": ticker.upper(),
        "Latest Close Price": round(latest['Close'], 2),
        "Volume": int(latest['Volume']),
        "SMA 20": round(latest['SMA_20'], 2) if not np.isnan(latest['SMA_20']) else "N/A",
        "SMA 50": round(latest['SMA_50'], 2) if not np.isnan(latest['SMA_50']) else "N/A",
    }
    return str(info)

@mcp.tool()
def predict_stock_trend_ml(ticker: str, days_ahead: int = 5) -> str:
    """Uses a linear regression Machine Learning model to forecast stock prices for N days ahead."""
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    
    if len(df) < 30:
        return f"Insufficient historical data to train ML model for {ticker}"

    # Feature Engineering
    df['Day'] = np.arange(len(df))
    X = df[['Day']].values
    y = df['Close'].values

    # Train local Scikit-Learn Model
    model = LinearRegression()
    model.fit(X, y)

    # Predict future prices
    last_day = df['Day'].iloc[-1]
    future_days = np.array([[last_day + i] for i in range(1, days_ahead + 1)])
    predictions = model.predict(future_days)

    current_price = y[-1]
    predicted_future_price = predictions[-1]
    predicted_change_pct = ((predicted_future_price - current_price) / current_price) * 100

    results = {
        "Ticker": ticker.upper(),
        "Current Price": round(current_price, 2),
        f"Predicted Price ({days_ahead} days)": round(predicted_future_price, 2),
        "Forecasted Trend": "Bullish" if predicted_change_pct > 0 else "Bearish",
        "Estimated Change (%)": f"{round(predicted_change_pct, 2)}%"
    }
    return str(results)

if __name__ == "__main__":
    mcp.run(transport="stdio")