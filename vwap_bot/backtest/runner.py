import backtrader as bt
import pandas as pd
import itertools
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *
from strategies.vwap_reversion import VWAPReversionStrategy
from utils.data_loader import load_bt_data

results = []

data = load_bt_data(DATA_PATH)

for vdev, sl, tp in itertools.product(VWAP_DEV_THRESHOLDS, STOP_LOSSES, TAKE_PROFITS):
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(VWAPReversionStrategy,
                        vwap_dev_threshold=vdev,
                        stop_loss=sl,
                        take_profit=tp)
    cerebro.broker.setcash(CAPITAL)
    cerebro.broker.setcommission(commission=COMMISSION)

    result = cerebro.run()
    strat = result[0]
    final_value = cerebro.broker.getvalue()

    results.append({
        'vwap_dev': vdev,
        'stop_loss': sl,
        'take_profit': tp,
        'final_value': final_value,
        'pnl': final_value - CAPITAL
    })

pd.DataFrame(results).to_csv("results/vwap_backtest_results.csv", index=False)
print("Backtest complete. Results saved to CSV.")
