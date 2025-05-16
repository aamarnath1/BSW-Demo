import backtrader as bt
import logging
from datetime import datetime, timedelta
from vwap_bot.indicators.vwap import VWAP

class VWAPReversionStrategy(bt.Strategy):
    params = (
        ('vwap_dev_threshold', 0.002),  # VWAP deviation threshold for entry
        ('stop_loss', 0.01),            # Stop loss percentage
        ('take_profit', 0.02),          # Take profit percentage
        ('max_hold_minutes', 30),       # Maximum holding time in minutes
        ('min_volume', 1000),           # Minimum volume for trade
        ('max_trades_per_day', 3),      # Maximum number of trades per day
        ('trailing_stop', 0.005),       # Trailing stop percentage
    )

    def __init__(self):
        self.vwap = VWAP(self.data, min_volume=self.p.min_volume)
        self.order = None
        self.trade_start_time = None
        self.trades_today = 0
        self.last_trade_date = None
        self.highest_price = 0
        self.lowest_price = float('inf')
        self.logger = logging.getLogger(__name__)

    def notify_trade(self, trade):
        if trade.isclosed:
            self.logger.info(f'Trade closed - Profit: {trade.pnl:.2f}, Net Profit: {trade.pnlcomm:.2f}')
            self.trade_start_time = None
            self.highest_price = 0
            self.lowest_price = float('inf')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.logger.info(f'BUY EXECUTED - Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.trade_start_time = self.datas[0].datetime.datetime(0)
                self.highest_price = order.executed.price
                self.lowest_price = order.executed.price
            else:
                self.logger.info(f'SELL EXECUTED - Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.logger.warning(f'Order Canceled/Margin/Rejected - Status: {order.status}')

        self.order = None

    def next(self):
        # Reset daily trade counter
        current_date = self.datas[0].datetime.date(0)
        if self.last_trade_date != current_date:
            self.trades_today = 0
            self.last_trade_date = current_date

        # Skip if we have a pending order
        if self.order:
            return

        # Skip if we've reached max trades for the day
        if self.trades_today >= self.p.max_trades_per_day:
            return

        # Get current price and VWAP
        price = self.data.close[0]
        vwap = self.vwap.lines.vwap[0]
        
        # Skip if price or VWAP is invalid
        if price <= 0 or vwap <= 0:
            return

        # Calculate deviation from VWAP
        deviation = (price - vwap) / vwap

        # Check for exit conditions if we have a position
        if self.position:
            # Update highest/lowest prices for trailing stop
            if price > self.highest_price:
                self.highest_price = price
            if price < self.lowest_price:
                self.lowest_price = price

            # Calculate profit/loss
            if self.position.size > 0:  # Long position
                pnl_pct = (price - self.position.price) / self.position.price
                trailing_stop_triggered = (self.highest_price - price) / self.highest_price >= self.p.trailing_stop
            else:  # Short position
                pnl_pct = (self.position.price - price) / self.position.price
                trailing_stop_triggered = (price - self.lowest_price) / self.lowest_price >= self.p.trailing_stop

            # Check exit conditions
            if (pnl_pct <= -self.p.stop_loss or  # Stop loss
                pnl_pct >= self.p.take_profit or  # Take profit
                trailing_stop_triggered or  # Trailing stop
                (self.trade_start_time and  # Max hold time
                 self.datas[0].datetime.datetime(0) - self.trade_start_time >= timedelta(minutes=self.p.max_hold_minutes))):
                
                self.logger.info(f'Closing position - PnL: {pnl_pct:.2%}')
                self.close()
                self.trades_today += 1
                return

        # Check for entry conditions if we don't have a position
        elif abs(deviation) >= self.p.vwap_dev_threshold:
            # Long entry if price is below VWAP
            if deviation < -self.p.vwap_dev_threshold:
                self.logger.info(f'Long entry - Price: {price:.2f}, VWAP: {vwap:.2f}, Deviation: {deviation:.2%}')
                self.buy()
                self.trades_today += 1
            # Short entry if price is above VWAP
            elif deviation > self.p.vwap_dev_threshold:
                self.logger.info(f'Short entry - Price: {price:.2f}, VWAP: {vwap:.2f}, Deviation: {deviation:.2%}')
                self.sell()
                self.trades_today += 1