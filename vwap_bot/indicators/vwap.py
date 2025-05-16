import backtrader as bt
import numpy as np

class VWAP(bt.Indicator):
    lines = ('vwap', 'upper_band', 'lower_band')
    params = (
        ('period', None),
        ('std_dev', 2.0),  # Standard deviations for bands
        ('min_volume', 1000),  # Minimum volume threshold
    )

    def __init__(self):
        self.addminperiod(1)
        self.cum_vol = 0
        self.cum_vol_price = 0
        self.last_date = None
        self.price_window = []
        self.vol_window = []
        self.max_window_size = 20  # Maximum size for volatility calculation window

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        
        # Reset on new day
        if self.last_date is not None and current_date != self.last_date:
            self.cum_vol = 0
            self.cum_vol_price = 0
            self.price_window = []
            self.vol_window = []
        
        self.last_date = current_date

        price = self.datas[0].close[0]
        vol = self.datas[0].volume[0]

        # Skip if price or volume is invalid
        if price <= 0 or vol <= 0:
            self.lines.vwap[0] = price
            self.lines.upper_band[0] = price * 1.003  # 0.3% default band width
            self.lines.lower_band[0] = price * 0.997
            return

        # Only update if volume meets minimum threshold
        if vol >= self.p.min_volume:
            self.cum_vol += vol
            self.cum_vol_price += price * vol
            
            # Update rolling windows for volatility calculation
            self.price_window.append(price)
            self.vol_window.append(vol)
            
            # Keep window size limited
            if len(self.price_window) > self.max_window_size:
                self.price_window.pop(0)
                self.vol_window.pop(0)

        # Calculate VWAP
        if self.cum_vol > 0:
            self.lines.vwap[0] = self.cum_vol_price / self.cum_vol
        else:
            self.lines.vwap[0] = price

        # Calculate volatility-based bands
        if len(self.price_window) >= 2:  # Need at least 2 points for std dev
            try:
                # Calculate weighted standard deviation
                weights = np.array(self.vol_window) / sum(self.vol_window)
                weighted_mean = np.average(self.price_window, weights=weights)
                weighted_var = np.average((np.array(self.price_window) - weighted_mean) ** 2, weights=weights)
                std = np.sqrt(weighted_var)
                
                # Ensure minimum volatility
                min_std = price * 0.0005  # Minimum 0.05% volatility
                std = max(std, min_std)
                
                self.lines.upper_band[0] = self.lines.vwap[0] + (std * self.p.std_dev)
                self.lines.lower_band[0] = self.lines.vwap[0] - (std * self.p.std_dev)
            except (ValueError, ZeroDivisionError):
                # Fallback to default bands if calculation fails
                default_band_width = price * 0.003  # 0.3% default band width
                self.lines.upper_band[0] = self.lines.vwap[0] + default_band_width
                self.lines.lower_band[0] = self.lines.vwap[0] - default_band_width
        else:
            # Default bands if not enough data
            default_band_width = price * 0.003  # 0.3% default band width
            self.lines.upper_band[0] = self.lines.vwap[0] + default_band_width
            self.lines.lower_band[0] = self.lines.vwap[0] - default_band_width
