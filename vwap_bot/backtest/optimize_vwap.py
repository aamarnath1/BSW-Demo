import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import os
import sys
import logging
from datetime import datetime
from vwap_bot.strategies.vwap_reversion import VWAPReversionStrategy
from vwap_bot.utils.data_loader import load_bt_data
from vwap_bot.indicators.vwap import VWAP
from vwap_bot.config import TICKERS, TIMEFRAMES, DATA_DIR, RESULTS_DIR, N_TRIALS, OPTIMIZATION_START_DATE, OPTIMIZATION_END_DATE

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VWAPOptimizer:
    def __init__(self, data, start_date, end_date):
        self.data = data
        self.start_date = start_date
        self.end_date = end_date
        self.cerebro = None
        self.best_params = None
        self.best_value = float('-inf')
        self.trials_results = []

    def run_backtest(self, params):
        cerebro = bt.Cerebro()
        cerebro.adddata(self.data)
        cerebro.addstrategy(VWAPReversionStrategy, **params)
        cerebro.broker.setcash(100000.0)
        cerebro.broker.setcommission(commission=0.001)  # 0.1% commission
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # Run backtest
        results = cerebro.run()
        return results[0]

    def objective(self, trial):
        # Define parameter ranges
        params = {
            'vwap_dev_threshold': trial.suggest_float('vwap_dev_threshold', 0.001, 0.005),
            'stop_loss': trial.suggest_float('stop_loss', 0.005, 0.02),
            'take_profit': trial.suggest_float('take_profit', 0.01, 0.04),
            'max_hold_minutes': trial.suggest_int('max_hold_minutes', 15, 60),
            'min_volume': trial.suggest_int('min_volume', 500, 5000),
            'trailing_stop': trial.suggest_float('trailing_stop', 0.003, 0.01),
            'max_trades_per_day': trial.suggest_int('max_trades_per_day', 2, 5)
        }

        logger.info(f"Trial {trial.number} - Testing parameters: {params}")
        
        try:
            results = self.run_backtest(params)
            
            # Get analyzers
            drawdown = results.analyzers.drawdown.get_analysis()
            trades = results.analyzers.trades.get_analysis()
            returns = results.analyzers.returns.get_analysis()
            
            # Calculate metrics
            total_trades = trades.get('total', {}).get('total', 0)
            won_trades = trades.get('won', {}).get('total', 0)
            lost_trades = trades.get('lost', {}).get('total', 0)
            
            if total_trades < 5:  # Minimum required trades
                logger.warning(f"Trial {trial.number} - Insufficient trades: {total_trades}")
                return float('-inf')
            
            # Calculate win rate
            win_rate = won_trades / total_trades if total_trades > 0 else 0
            
            # Get max drawdown
            max_drawdown = drawdown.get('max', {}).get('drawdown', 0) / 100
            
            # Calculate compound return
            compound_return = returns.get('rtot', 0)
            
            # Composite score: prioritize return, penalize high drawdown, reward win rate
            score = compound_return * (win_rate / 0.5) * (1 - max_drawdown)
            
            # Store trial results
            trial_result = {
                'trial_number': trial.number,
                'params': params,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'compound_return': compound_return,
                'score': score
            }
            self.trials_results.append(trial_result)
            
            logger.info(f"Trial {trial.number} - Score: {score:.4f}, Return: {compound_return:.4f}, "
                       f"Drawdown: {max_drawdown:.2%}, Win Rate: {win_rate:.2%}, "
                       f"Trades: {total_trades}")
            
            return score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {str(e)}")
            return float('-inf')

    def plot_optuna(self, study):
        import optuna.visualization.matplotlib as optuna_plot
        import matplotlib.pyplot as plt
        # Optimization history
        ax1 = optuna_plot.plot_optimization_history(study)
        fig1 = ax1.figure
        plt.savefig('optuna_optimization_history.png')
        plt.close(fig1)
        # Parameter importances
        ax2 = optuna_plot.plot_param_importances(study)
        fig2 = ax2.figure
        plt.savefig('optuna_param_importances.png')
        plt.close(fig2)

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        self.best_params = study.best_params
        self.best_value = study.best_value
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.trials_results)
        results_df.to_csv('optimization_results.csv', index=False)
        # Plot optuna history and importances
        self.plot_optuna(study)
        return self.best_params, self.best_value

    def plot_results(self, results_df=None, ticker=None, timeframe=None):
        if results_df is None:
            if not self.trials_results:
                logger.warning("No results to plot")
                return
            results_df = pd.DataFrame(self.trials_results)
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(15, 10))
        # Plot 1: Parameter distributions
        plt.subplot(2, 2, 1)
        for param in ['vwap_dev_threshold', 'stop_loss', 'take_profit', 'trailing_stop']:
            if param in results_df.columns:
                sns.kdeplot(data=results_df, x=param, label=param)
        plt.title('Parameter Distributions')
        plt.legend()
        # Plot 2: Score vs Parameters
        plt.subplot(2, 2, 2)
        if 'compound_return' in results_df.columns and 'score' in results_df.columns:
            plt.scatter(results_df['compound_return'], results_df['score'])
        plt.xlabel('Compound Return')
        plt.ylabel('Score')
        plt.title('Score vs Compound Return')
        # Plot 3: Win Rate vs Drawdown
        plt.subplot(2, 2, 3)
        if 'win_rate' in results_df.columns and 'max_drawdown' in results_df.columns:
            plt.scatter(results_df['win_rate'], results_df['max_drawdown'])
        plt.xlabel('Win Rate')
        plt.ylabel('Max Drawdown')
        plt.title('Win Rate vs Drawdown')
        # Plot 4: Trade Distribution
        plt.subplot(2, 2, 4)
        if 'total_trades' in results_df.columns:
            sns.histplot(data=results_df, x='total_trades', bins=20)
        plt.xlabel('Total Trades')
        plt.ylabel('Count')
        plt.title('Trade Distribution')
        plt.tight_layout()
        if ticker and timeframe:
            plt.savefig(os.path.join(RESULTS_DIR, f'optimization_{ticker}_{timeframe}.png'))
        else:
            plt.savefig('optimization_plots.png')
        plt.close()

# --- Cross-validation utility ---
def cross_validate_on_tickers_and_timeframes(tickers, timeframes, base_data_dir, start_date, end_date, n_trials=50):
    results = []
    for ticker in tickers:
        for tf in timeframes:
            data_file = os.path.join(base_data_dir, f"{ticker}_{tf}.csv")
            if not os.path.exists(data_file):
                logger.warning(f"Data file not found: {data_file}")
                continue
            data = load_bt_data(data_file)
            optimizer = VWAPOptimizer(data, start_date, end_date)
            best_params, best_value = optimizer.optimize(n_trials=n_trials)
            results.append({
                'ticker': ticker,
                'timeframe': tf,
                'best_params': best_params,
                'best_value': best_value
            })
            print(f"[CV] {ticker} {tf}: Best value={best_value}, Params={best_params}")
    pd.DataFrame(results).to_csv('cross_validation_results.csv', index=False)
    return results

def main():
    logger.info("Starting optimization...")
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    all_results = []
    
    # Run optimization for each ticker and timeframe
    for ticker in TICKERS:
        for timeframe in TIMEFRAMES:
            logger.info(f"Optimizing {ticker} {timeframe}...")
            
            # Load data
            data_file = os.path.join(DATA_DIR, f"{ticker}_{timeframe}_cleaned.csv")
            if not os.path.exists(data_file):
                logger.warning(f"Data file not found: {data_file}")
                continue
                
            data = load_bt_data(data_file)
            
            # Create optimizer
            optimizer = VWAPOptimizer(
                data=data,
                start_date=datetime.strptime(OPTIMIZATION_START_DATE, "%Y-%m-%d"),
                end_date=datetime.strptime(OPTIMIZATION_END_DATE, "%Y-%m-%d")
            )
            
            # Run optimization
            best_params, best_value = optimizer.optimize(n_trials=N_TRIALS)
            
            # Save full trial results for this run
            results_df = pd.DataFrame(optimizer.trials_results)
            results_df.to_csv(os.path.join(RESULTS_DIR, f'optimization_{ticker}_{timeframe}.csv'), index=False)
            # Save plots for this run
            optimizer.plot_results(results_df, ticker, timeframe)
            
            # Store results
            result = {
                'ticker': ticker,
                'timeframe': timeframe,
                'best_params': best_params,
                'best_value': best_value
            }
            all_results.append(result)
            
            logger.info(f"Best parameters for {ticker} {timeframe}: {best_params}")
            logger.info(f"Best value: {best_value}")
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'optimization_results.csv'), index=False)
    logger.info("Optimization complete. Results saved to CSV.")

if __name__ == '__main__':
    main() 