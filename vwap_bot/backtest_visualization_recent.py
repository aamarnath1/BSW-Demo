import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import seaborn as sns
from tabulate import tabulate

class VWAPBacktestVisualizer:
    def __init__(self, api_key, secret_key):
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        self.params = {
            'vwap_dev_threshold': 0.0034,  # Same as paper trading
            'stop_loss': 0.006,
            'take_profit': 0.012,
            'max_hold_minutes': 43,
            'min_volume': 2500,
            'trailing_stop': 0.0035,
            'max_trades_per_day': 3
        }
        
    def get_historical_data(self, symbol, start_date, end_date):
        """Get historical data for backtesting"""
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start_date,
                end=end_date
            )
            
            bars = self.data_client.get_stock_bars(request_params)
            df = bars.df
            
            # Reset index to make timestamp a column
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                df = df.reset_index()
                df['timestamp'] = pd.to_datetime(df.index)
            
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_vwap(self, data):
        """Calculate VWAP and bands"""
        data = data.copy()
        data['cum_vol'] = data['volume'].cumsum()
        data['cum_vol_price'] = (data['close'] * data['volume']).cumsum()
        data['vwap'] = data['cum_vol_price'] / data['cum_vol']
        data['upper_band'] = data['vwap'] * (1 + self.params['vwap_dev_threshold'])
        data['lower_band'] = data['vwap'] * (1 - self.params['vwap_dev_threshold'])
        return data
    
    def run_backtest(self, symbol, start_date, end_date, initial_capital=100000):
        """Run backtest and return results"""
        # Get historical data
        data = self.get_historical_data(symbol, start_date, end_date)
        if data.empty:
            print(f"No data available for {symbol}")
            return None
        
        # Calculate VWAP
        data = self.calculate_vwap(data)
        
        # Initialize backtest variables
        position = 0
        entry_price = 0
        entry_time = None
        trades = []
        equity = [initial_capital]
        current_capital = initial_capital
        trades_today = 0
        last_trade_date = None
        
        # Run backtest
        for i in range(1, len(data)):
            current_price = data['close'].iloc[i]
            current_time = data['timestamp'].iloc[i]
            current_date = current_time.date()
            
            # Reset daily trade counter
            if last_trade_date != current_date:
                trades_today = 0
                last_trade_date = current_date
            
            # Check exit conditions if we have a position
            if position != 0:
                pnl_pct = (current_price - entry_price) / entry_price if position > 0 else (entry_price - current_price) / entry_price
                
                # Check stop loss
                if pnl_pct <= -self.params['stop_loss']:
                    trade_pnl = position * (current_price - entry_price)
                    current_capital += trade_pnl
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'pnl': trade_pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'stop_loss'
                    })
                    position = 0
                    trades_today += 1
                
                # Check take profit
                elif pnl_pct >= self.params['take_profit']:
                    trade_pnl = position * (current_price - entry_price)
                    current_capital += trade_pnl
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'pnl': trade_pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'take_profit'
                    })
                    position = 0
                    trades_today += 1
                
                # Check max hold time
                elif entry_time and (current_time - entry_time).total_seconds() / 60 >= self.params['max_hold_minutes']:
                    trade_pnl = position * (current_price - entry_price)
                    current_capital += trade_pnl
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'pnl': trade_pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'max_hold_time'
                    })
                    position = 0
                    trades_today += 1
            
            # Check entry conditions if we don't have a position and haven't exceeded daily trade limit
            if position == 0 and trades_today < self.params['max_trades_per_day']:
                if current_price < data['lower_band'].iloc[i]:
                    # Long entry
                    position = int(current_capital * 0.1 / current_price)
                    entry_price = current_price
                    entry_time = current_time
                elif current_price > data['upper_band'].iloc[i]:
                    # Short entry
                    position = -int(current_capital * 0.1 / current_price)
                    entry_price = current_price
                    entry_time = current_time
            
            # Update equity curve
            if position != 0:
                current_pnl = position * (current_price - entry_price)
                equity.append(current_capital + current_pnl)
            else:
                equity.append(current_capital)
        
        return {
            'trades': trades,
            'equity': equity,
            'data': data
        }
    
    def plot_results(self, results, symbol):
        """Plot backtest results"""
        if not results:
            print(f"No data available for {symbol}")
            return
        
        trades = results['trades']
        equity = results['equity']
        data = results['data']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
        
        # Plot price and VWAP
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(data['timestamp'], data['close'], label='Price', alpha=0.7)
        ax1.plot(data['timestamp'], data['vwap'], label='VWAP', alpha=0.7)
        ax1.plot(data['timestamp'], data['upper_band'], 'r--', label='Upper Band', alpha=0.5)
        ax1.plot(data['timestamp'], data['lower_band'], 'g--', label='Lower Band', alpha=0.5)
        
        # Plot trades
        for trade in trades:
            if trade['position'] > 0:  # Long
                ax1.scatter(trade['entry_time'], trade['entry_price'], color='g', marker='^')
                ax1.scatter(trade['exit_time'], trade['exit_price'], color='r', marker='v')
            else:  # Short
                ax1.scatter(trade['entry_time'], trade['entry_price'], color='r', marker='v')
                ax1.scatter(trade['exit_time'], trade['exit_price'], color='g', marker='^')
        
        ax1.set_title(f'{symbol} Price and VWAP (Last 30 Days)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot equity curve
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(data['timestamp'][:len(equity)], equity, label='Equity')
        ax2.set_title('Equity Curve')
        ax2.grid(True)
        
        # Plot trade P&L
        ax3 = fig.add_subplot(gs[2])
        trade_pnls = [t['pnl'] for t in trades]
        ax3.bar(range(len(trade_pnls)), trade_pnls, color=['g' if p > 0 else 'r' for p in trade_pnls])
        ax3.set_title('Trade P&L')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'backtest_results_recent_{symbol}.png')
        plt.close()
        
        # Calculate additional statistics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        total_pnl = sum(t['pnl'] for t in trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate drawdown
        equity_series = pd.Series(equity)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 0)
        returns = pd.Series(equity).pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
        
        stats = [
            ['Total Trades', total_trades],
            ['Winning Trades', winning_trades],
            ['Win Rate', f'{win_rate:.2%}'],
            ['Total P&L', f'${total_pnl:,.2f}'],
            ['Average P&L per Trade', f'${total_pnl/total_trades:,.2f}' if total_trades > 0 else '$0.00'],
            ['Max Drawdown', f'{max_drawdown:.2%}'],
            ['Sharpe Ratio', f'{sharpe_ratio:.2f}'],
            ['Final Equity', f'${equity[-1]:,.2f}']
        ]
        
        print(f"\nBacktest Results for {symbol} (Last 30 Days):")
        print(tabulate(stats, tablefmt='grid'))
        
        return stats

if __name__ == "__main__":
    # Your Alpaca API keys
    API_KEY = "PKJEXWMPGWZTGLVO8GJB"
    SECRET_KEY = "xcFH83Y9QfocpkBLEfU1nKW4bobGhkFvUvHYEYqi"
    
    # Initialize visualizer
    visualizer = VWAPBacktestVisualizer(API_KEY, SECRET_KEY)
    
    # Set date range (last month)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Run backtest for each symbol
    symbols = [
        'AAPL',   # Apple
        'NVDA',   # NVIDIA
        'MSFT',   # Microsoft
        'GOOGL',  # Google
        'AMZN',   # Amazon
        'META',   # Meta
        'TSLA',   # Tesla
        'SPY',    # S&P 500 ETF
        'QQQ'     # NASDAQ ETF
    ]
    all_stats = []
    all_trades = []
    
    for symbol in symbols:
        print(f"\nRunning backtest for {symbol}...")
        results = visualizer.run_backtest(symbol, start_date, end_date)
        if results:
            stats = visualizer.plot_results(results, symbol)
            all_stats.append([symbol] + [stat[1] for stat in stats])
            
            # Save individual trade data
            trades_df = pd.DataFrame(results['trades'])
            if not trades_df.empty:
                trades_df['symbol'] = symbol
                all_trades.append(trades_df)
    
    # Print comparison table
    if all_stats:
        headers = ['Symbol', 'Total Trades', 'Winning Trades', 'Win Rate', 
                  'Total P&L', 'Avg P&L/Trade', 'Max Drawdown', 'Sharpe Ratio', 'Final Equity']
        print("\nStrategy Performance Comparison (Last 30 Days):")
        print(tabulate(all_stats, headers=headers, tablefmt='grid'))
        
        # Save summary statistics
        summary_df = pd.DataFrame(all_stats, columns=headers)
        summary_df.to_csv('backtest_summary_recent.csv', index=False)
        summary_df.to_excel('backtest_summary_recent.xlsx', index=False)
        
        # Save detailed trade data
        if all_trades:
            trades_df = pd.concat(all_trades, ignore_index=True)
            # Convert timezone-aware timestamps to timezone-naive
            trades_df['entry_time'] = trades_df['entry_time'].dt.tz_localize(None)
            trades_df['exit_time'] = trades_df['exit_time'].dt.tz_localize(None)
            trades_df.to_csv('backtest_trades_recent.csv', index=False)
            trades_df.to_excel('backtest_trades_recent.xlsx', index=False)
            
            # Create additional analysis
            analysis = {
                'Total Trades': len(trades_df),
                'Total Winning Trades': len(trades_df[trades_df['pnl'] > 0]),
                'Total Losing Trades': len(trades_df[trades_df['pnl'] <= 0]),
                'Average Win': trades_df[trades_df['pnl'] > 0]['pnl'].mean(),
                'Average Loss': trades_df[trades_df['pnl'] <= 0]['pnl'].mean(),
                'Largest Win': trades_df['pnl'].max(),
                'Largest Loss': trades_df['pnl'].min(),
                'Average Trade Duration': (trades_df['exit_time'] - trades_df['entry_time']).mean(),
                'Most Profitable Symbol': trades_df.groupby('symbol')['pnl'].sum().idxmax(),
                'Most Active Symbol': trades_df['symbol'].value_counts().idxmax()
            }
            
            # Save analysis
            analysis_df = pd.DataFrame(list(analysis.items()), columns=['Metric', 'Value'])
            analysis_df.to_csv('backtest_analysis_recent.csv', index=False)
            analysis_df.to_excel('backtest_analysis_recent.xlsx', index=False)
            
            print("\nDetailed analysis saved to:")
            print("- backtest_summary_recent.csv/xlsx (Overall performance by symbol)")
            print("- backtest_trades_recent.csv/xlsx (Individual trade data)")
            print("- backtest_analysis_recent.csv/xlsx (Detailed strategy analysis)") 