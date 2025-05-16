import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from vwap_bot.indicators.vwap import VWAP

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paper_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VWAPPaperTrader:
    def __init__(self, api_key, secret_key, paper=True):
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        self.positions = {}
        self.orders = {}
        
        # Load optimized parameters
        self.params = {
            'vwap_dev_threshold': 0.0034,  # From optimization results
            'stop_loss': 0.006,            # From optimization results
            'take_profit': 0.012,          # From optimization results
            'max_hold_minutes': 43,        # From optimization results
            'min_volume': 2500,            # From optimization results
            'trailing_stop': 0.0035,       # From optimization results
            'max_trades_per_day': 3        # Conservative limit
        }
        
    def get_account(self):
        """Get account information"""
        return self.trading_client.get_account()
    
    def get_positions(self):
        """Get current positions"""
        return self.trading_client.get_all_positions()
    
    def place_order(self, symbol, qty, side):
        """Place a market order"""
        try:
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            order = self.trading_client.submit_order(order_data)
            logger.info(f"Order placed: {order}")
            return order
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def get_historical_data(self, symbol, timeframe='1Min', lookback=100):
        """Get historical data for VWAP calculation"""
        end = datetime.now()
        start = end - timedelta(minutes=lookback)
        
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end
        )
        
        bars = self.data_client.get_stock_bars(request_params)
        return bars.df
    
    def calculate_vwap(self, data):
        # data: pandas DataFrame with columns ['close', 'volume']
        data = data.copy()
        data['cum_vol'] = data['volume'].cumsum()
        data['cum_vol_price'] = (data['close'] * data['volume']).cumsum()
        data['vwap'] = data['cum_vol_price'] / data['cum_vol']
        # Calculate bands (e.g., 0.34% above/below VWAP)
        band_width = self.params['vwap_dev_threshold']
        data['upper_band'] = data['vwap'] * (1 + band_width)
        data['lower_band'] = data['vwap'] * (1 - band_width)
        return data['vwap'].iloc[-1], data['upper_band'].iloc[-1], data['lower_band'].iloc[-1]
    
    def check_exit_conditions(self, position, current_price):
        """Check if we should exit a position"""
        entry_price = float(position.avg_entry_price)
        side = position.side
        qty = float(position.qty)
        
        # Calculate P&L
        if side == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Check stop loss
        if pnl_pct <= -self.params['stop_loss']:
            logger.info(f"Stop loss triggered: {pnl_pct:.2%}")
            return True
            
        # Check take profit
        if pnl_pct >= self.params['take_profit']:
            logger.info(f"Take profit triggered: {pnl_pct:.2%}")
            return True
            
        return False
    
    def run(self, symbols=['AAPL', 'NVDA', 'QQQ', 'SPY']):
        """Main trading loop"""
        logger.info("Starting paper trading...")
        
        while True:
            try:
                # Get current positions
                positions = self.get_positions()
                
                # Check exit conditions for existing positions
                for position in positions:
                    symbol = position.symbol
                    current_price = float(position.current_price)
                    
                    if self.check_exit_conditions(position, current_price):
                        side = OrderSide.SELL if position.side == 'long' else OrderSide.BUY
                        self.place_order(symbol, position.qty, side)
                
                # Check for new entries
                for symbol in symbols:
                    # Skip if we already have a position
                    if any(p.symbol == symbol for p in positions):
                        continue
                    
                    # Get historical data
                    data = self.get_historical_data(symbol)
                    if data.empty:
                        continue
                    
                    # Calculate VWAP
                    vwap, upper_band, lower_band = self.calculate_vwap(data)
                    current_price = data['close'].iloc[-1]
                    
                    # Check entry conditions
                    if current_price < lower_band:
                        # Long entry
                        account = self.get_account()
                        cash = float(account.cash)
                        qty = int(cash * 0.1 / current_price)  # Use 10% of cash
                        if qty > 0:
                            self.place_order(symbol, qty, OrderSide.BUY)
                    elif current_price > upper_band:
                        # Short entry
                        account = self.get_account()
                        cash = float(account.cash)
                        qty = int(cash * 0.1 / current_price)  # Use 10% of cash
                        if qty > 0:
                            self.place_order(symbol, qty, OrderSide.SELL)
                
                # Sleep for 1 minute
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    # Hardcoded Alpaca API keys for local testing and demo purposes
    API_KEY = "PKJEXWMPGWZTGLVO8GJB"
    SECRET_KEY = "xcFH83Y9QfocpkBLEfU1nKW4bobGhkFvUvHYEYqi"
    
    trader = VWAPPaperTrader(API_KEY, SECRET_KEY, paper=True)
    trader.run() 