import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_intraday_data(symbols, interval, period, out_path='data'):
    """
    Download intraday data for a list of symbols at a given interval and period.
    Args:
        symbols (list): List of stock symbols
        interval (str): Interval string ('1m', '5m', '15m')
        period (str): Period string ('7d' for 1m, '60d' for 5m/15m)
        out_path (str): Directory to save data
    """
    os.makedirs(out_path, exist_ok=True)
    end_date = datetime.now()
    if period.endswith('d'):
        days = int(period[:-1])
        start_date = end_date - timedelta(days=days)
    else:
        start_date = end_date - timedelta(days=7)  # fallback
    logger.info(f"Downloading {interval} data from {start_date.date()} to {end_date.date()} for: {symbols}")
    for symbol in symbols:
        try:
            logger.info(f"Downloading {symbol} {interval} data...")
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False
            )
            if data.empty:
                logger.error(f"❌ No data found for {symbol} {interval}")
                continue
            filename = f"{symbol}_{interval}.csv"
            filepath = os.path.join(out_path, filename)
            data.to_csv(filepath)
            logger.info(f"✅ Saved {len(data)} rows to {filepath}")
        except Exception as e:
            logger.error(f"❌ Error downloading {symbol} {interval}: {str(e)}")

if __name__ == "__main__":
    # All tickers to use
    all_symbols = ['AAPL', 'NVDA', 'QQQ', 'SPY']
    # Download 1-minute data (7 days)
    download_intraday_data(
        symbols=all_symbols,
        interval='1m',
        period='7d',
        out_path='data'
    )
    # Download 5-minute data (60 days)
    download_intraday_data(
        symbols=all_symbols,
        interval='5m',
        period='60d',
        out_path='data'
    )
    # Download 15-minute data (60 days)
    download_intraday_data(
        symbols=all_symbols,
        interval='15m',
        period='60d',
        out_path='data'
    )
