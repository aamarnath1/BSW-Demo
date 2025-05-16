import os
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

def test_alpaca_connection():
    # Use the provided API keys
    API_KEY = "PKJEXWMPGWZTGLVO8GJB"
    SECRET_KEY = "xcFH83Y9QfocpkBLEfU1nKW4bobGhkFvUvHYEYqi"
    
    try:
        # Initialize trading client
        trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        
        # Get account information
        account = trading_client.get_account()
        print("\n=== Account Information ===")
        print(f"Account Status: {account.status}")
        print(f"Cash Balance: ${float(account.cash):,.2f}")
        print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"Buying Power: ${float(account.buying_power):,.2f}")
        
        # Test market data access
        data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
        
        # Get recent AAPL data
        end = datetime.now()
        start = end - timedelta(minutes=5)
        
        request_params = StockBarsRequest(
            symbol_or_symbols="AAPL",
            timeframe=TimeFrame.Minute,
            start=start,
            end=end
        )
        
        bars = data_client.get_stock_bars(request_params)
        print("\n=== Market Data Test ===")
        print(f"Successfully retrieved {len(bars.df)} bars of AAPL data")
        
        print("\nConnection test successful! You can now run the paper trading script.")
        
    except Exception as e:
        print(f"Error testing Alpaca connection: {e}")

if __name__ == "__main__":
    test_alpaca_connection() 