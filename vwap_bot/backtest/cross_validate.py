import pandas as pd
import backtrader as bt
import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *
from vwap_bot.strategies.vwap_reversion import VWAPReversionStrategy
from vwap_bot.utils.data_loader import load_bt_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_backtest(ticker, config):
    """Run backtest for a single ticker and configuration."""
    data = load_bt_data(f"data/{ticker}.csv")
    
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(VWAPReversionStrategy,
                        vwap_dev_threshold=config['vwap_dev'],
                        stop_loss=config['stop_loss'],
                        take_profit=config['take_profit'])
    cerebro.broker.setcash(CAPITAL)
    cerebro.broker.setcommission(commission=COMMISSION)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
    
    result = cerebro.run()
    strat = result[0]
    
    # Get analyzer results
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    time_return = strat.analyzers.time_return.get_analysis()
    
    # Calculate equity curve
    equity_curve = pd.Series(time_return)
    
    return {
        'ticker': ticker,
        'vwap_dev': config['vwap_dev'],
        'stop_loss': config['stop_loss'],
        'take_profit': config['take_profit'],
        'final_value': cerebro.broker.getvalue(),
        'pnl': cerebro.broker.getvalue() - CAPITAL,
        'num_trades': trades.get('total', {}).get('total', 0),
        'win_rate': trades.get('won', {}).get('total', 0) / trades.get('total', {}).get('total', 1) if trades.get('total', {}).get('total', 0) > 0 else 0,
        'sharpe_ratio': sharpe.get('sharperatio', 0),
        'max_drawdown': drawdown.get('max', {}).get('drawdown', 0) / 100,
        'equity_curve': equity_curve
    }

def plot_equity_curves(results, ticker):
    """Plot equity curves for all configurations of a ticker."""
    plt.figure(figsize=(12, 6))
    
    for result in results:
        if result['ticker'] == ticker:
            label = f"Dev: {result['vwap_dev']:.3f}, SL: {result['stop_loss']:.3f}, TP: {result['take_profit']:.3f}"
            plt.plot(result['equity_curve'].index, result['equity_curve'].values, label=label)
    
    plt.title(f'{ticker} Equity Curves')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/vwap_{ticker.lower()}_equity.png')
    plt.close()

def run_cross_validation(tickers, intervals, data_dir='data', results_dir='results'):
    """
    Run cross-validation for the VWAP reversion strategy on all tickers and intervals using the best parameters.
    Args:
        tickers (list): List of ticker symbols.
        intervals (list): List of intervals ('1m', '5m', '15m').
        data_dir (str): Directory containing the CSV data files.
        results_dir (str): Directory to save cross-validation results.
    """
    # Load best parameters from the optimization results
    best_params_path = os.path.join(results_dir, 'best_parameters.csv')
    if not os.path.exists(best_params_path):
        logger.error(f"Best parameters file not found at {best_params_path}. Please run optimization first.")
        return
    best_params = pd.read_csv(best_params_path).iloc[0].to_dict()
    # Remove non-strategy keys
    for k in list(best_params.keys()):
        if k not in ['vwap_dev_threshold', 'stop_loss', 'take_profit', 'max_hold_minutes', 'min_volume', 'trailing_stop']:
            del best_params[k]
    logger.info(f"Loaded best parameters: {best_params}")

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Collect all results
    all_results = []

    # Run backtest for each ticker and interval
    for ticker in tickers:
        for interval in intervals:
            data_file = os.path.join(data_dir, f"{ticker}_{interval}.csv")
            if not os.path.exists(data_file):
                logger.warning(f"Data file not found: {data_file}. Skipping.")
                continue
            logger.info(f"Running cross-validation for {ticker} {interval}...")
            cerebro = bt.Cerebro()
            data = load_bt_data(data_file)
            cerebro.adddata(data)
            cerebro.addstrategy(VWAPReversionStrategy, **best_params)
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            results = cerebro.run()
            strat = results[0]
            sharpe = strat.analyzers.sharpe.get_analysis()
            drawdown = strat.analyzers.drawdown.get_analysis()
            trade = strat.analyzers.trade.get_analysis()
            returns = strat.analyzers.returns.get_analysis()
            logger.info(f"Results for {ticker} {interval}:")
            logger.info(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
            logger.info(f"Max Drawdown: {drawdown.get('max', {}).get('drawdown', 'N/A')}")
            logger.info(f"Total Trades: {trade.get('total', {}).get('total', 'N/A')}")
            logger.info(f"Compound Return: {returns.get('rtot', 'N/A')}")
            # Save result
            result = {
                'ticker': ticker,
                'interval': interval,
                'sharpe_ratio': sharpe.get('sharperatio', None),
                'max_drawdown': drawdown.get('max', {}).get('drawdown', None),
                'total_trades': trade.get('total', {}).get('total', None),
                'compound_return': returns.get('rtot', None),
            }
            result.update(best_params)
            all_results.append(result)
    # Save all results to CSV
    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(results_dir, 'vwap_cross_validation.csv')
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"All cross-validation results saved to {results_csv_path}")

def main():
    # Load top configurations
    summary_path = Path("results/vwap_summary.csv")
    if not summary_path.exists():
        print("Error: Summary file not found! Run evaluate_results.py first.")
        return
    
    top_configs = pd.read_csv(summary_path).head(3)
    
    # Test tickers
    tickers = ['NVDA', 'QQQ', 'SPY']
    results = []
    
    # Run backtests for each ticker and configuration
    for ticker in tickers:
        print(f"\nTesting {ticker}...")
        ticker_results = []
        for _, config in top_configs.iterrows():
            result = run_backtest(ticker, config)
            results.append(result)
            ticker_results.append(result)
            
            # Save individual results
            result_copy = result.copy()
            result_copy['equity_curve'] = result_copy['equity_curve'].to_dict()  # Convert to dict for CSV
            pd.DataFrame([result_copy]).to_csv(f"results/vwap_{ticker.lower()}.csv", index=False)
        
        # Plot equity curves for this ticker
        plot_equity_curves(ticker_results, ticker)
    
    # Create comparison table
    comparison = pd.DataFrame(results)
    comparison['return_pct'] = (comparison['final_value'] / CAPITAL - 1) * 100
    
    # Print comparison
    print("\nCross-Validation Results:")
    print("-" * 70)
    for ticker in tickers:
        ticker_results = comparison[comparison['ticker'] == ticker]
        print(f"\n{ticker} Results:")
        for _, row in ticker_results.iterrows():
            config_str = f"Dev: {row['vwap_dev']:.3f}, SL: {row['stop_loss']:.3f}, TP: {row['take_profit']:.3f}"
            perf_str = f"PnL: ${row['pnl']:.2f} | Return: {row['return_pct']:.1f}% | Trades: {row['num_trades']}"
            perf_str += f" | Sharpe: {row['sharpe_ratio']:.2f} | Max DD: {row['max_drawdown']*100:.1f}%"
            print(f"{config_str} | {perf_str}")
    
    # Save comparison
    comparison.to_csv("results/vwap_cross_validation.csv", index=False)
    print("\nCross-validation results saved to: results/vwap_cross_validation.csv")
    print("Equity curve plots saved to results/ directory")

if __name__ == "__main__":
    tickers = ['AAPL', 'NVDA', 'QQQ', 'SPY']
    intervals = ['1m', '5m', '15m']
    run_cross_validation(tickers, intervals) 