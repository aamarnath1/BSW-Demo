VWAP_DEV_THRESHOLDS = [0.001, 0.002, 0.003]
STOP_LOSSES = [0.005, 0.01, 0.015]
TAKE_PROFITS = [0.01, 0.02, 0.03]
CAPITAL = 100000.0  # Increased capital for better testing
COMMISSION = 0.001

# Data configuration
TICKERS = ['AAPL', 'NVDA', 'QQQ', 'SPY']
TIMEFRAMES = ['1m', '5m', '15m']
DATA_DIR = "data"
RESULTS_DIR = "results"

# Optimization parameters
N_TRIALS = 100  # Number of optimization trials
OPTIMIZATION_START_DATE = "2023-01-01"
OPTIMIZATION_END_DATE = "2023-12-31"
