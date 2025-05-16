import backtrader as bt
import pandas as pd
import os
import logging
import datetime

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_bt_data(filepath):
    """Load data from Yahoo Finance CSV file into Backtrader format."""
    logger.debug(f"Loading data from {filepath}")
    
    # Read the CSV file, skipping the first three rows, and use correct column names
    df = pd.read_csv(filepath, skiprows=3, names=['datetime', 'close', 'high', 'low', 'open', 'volume'])
    logger.debug(f"Original columns: {df.columns.tolist()}")
    
    # Convert datetime to proper format
    df['datetime'] = pd.to_datetime(df['datetime'])
    logger.debug(f"First datetime: {df['datetime'].iloc[0]}")
    
    # Convert numeric columns to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        logger.debug(f"{col} range: {df[col].min()} to {df[col].max()}")
    
    # Remove any empty rows and reset index
    df = df.dropna().reset_index(drop=True)
    logger.debug(f"Number of rows after cleaning: {len(df)}")
    
    # Save cleaned data to a temporary file
    temp_file = filepath.replace('.csv', '_cleaned.csv')
    df.to_csv(temp_file, index=False, date_format='%Y-%m-%d %H:%M:%S', float_format='%.6f')
    logger.debug(f"Saved cleaned data to {temp_file}")
    
    # Load the cleaned data into Backtrader using PandasData
    data = bt.feeds.PandasData(
        dataname=df,
        datetime='datetime',
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        timeframe=bt.TimeFrame.Minutes,
        compression=1
    )
    
    # Store the temp file path in the data object for cleanup later
    data.temp_file = temp_file
    
    return data
