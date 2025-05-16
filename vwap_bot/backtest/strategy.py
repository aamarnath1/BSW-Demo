import backtrader as bt
import numpy as np

class VWAPReversionStrategy(bt.Strategy):
    params = (
        ('vwap_dev_threshold', 0.002),
        ('stop_loss', 0.01),
        ('take_profit', 0.02),
        ('max_hold_minutes', 15),
        ('min_volume', 1000),
        ('trailing_stop', 0.005),
    )

    def __init__(self):
        self.vwap = bt.indicators.VWAP(self.data, period=20, std_dev=2, min_volume=self.p.min_volume)
        self.order = None
        self.entry_price = None
        self.entry_time = None
        self.trailing_stop_price = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
                self.entry_time = len(self)
                self.trailing_stop_price = self.entry_price * (1 - self.p.trailing_stop)
            else:
                self.entry_price = None
                self.entry_time = None
                self.trailing_stop_price = None

        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            # Check for entry conditions
            if self.data.volume[0] < self.p.min_volume:
                return

            price = self.data.close[0]
            vwap = self.vwap.vwap[0]
            upper_band = self.vwap.upper_band[0]
            lower_band = self.vwap.lower_band[0]

            # Long entry
            if price < lower_band:
                self.order = self.buy()
            # Short entry
            elif price > upper_band:
                self.order = self.sell()

        else:
            # Check exit conditions
            price = self.data.close[0]
            current_time = len(self)
            hold_time = (current_time - self.entry_time) * self.data._timeframe

            # Update trailing stop
            if self.position.size > 0:  # Long position
                self.trailing_stop_price = max(self.trailing_stop_price, price * (1 - self.p.trailing_stop))
                if price <= self.trailing_stop_price:
                    self.order = self.close()
            else:  # Short position
                self.trailing_stop_price = min(self.trailing_stop_price, price * (1 + self.p.trailing_stop))
                if price >= self.trailing_stop_price:
                    self.order = self.close()

            # Check stop loss and take profit
            if self.position.size > 0:  # Long position
                if price <= self.entry_price * (1 - self.p.stop_loss) or \
                   price >= self.entry_price * (1 + self.p.take_profit):
                    self.order = self.close()
            else:  # Short position
                if price >= self.entry_price * (1 + self.p.stop_loss) or \
                   price <= self.entry_price * (1 - self.p.take_profit):
                    self.order = self.close()

            # Check max hold time
            if hold_time >= self.p.max_hold_minutes:
                self.order = self.close() 