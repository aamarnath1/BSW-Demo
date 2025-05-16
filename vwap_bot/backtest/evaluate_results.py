import pandas as pd
import numpy as np
from pathlib import Path

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate annualized Sharpe ratio."""
    excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
    if len(excess_returns) < 2:
        return 0
    return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown."""
    rolling_max = equity_curve.expanding().max()
    drawdowns = equity_curve / rolling_max - 1
    return drawdowns.min()

def calculate_metrics(df):
    """Calculate performance metrics for each configuration."""
    # Calculate basic metrics
    df['total_pnl'] = df['final_value'] - CAPITAL
    df['return_pct'] = (df['final_value'] / CAPITAL - 1) * 100
    
    # Calculate risk-adjusted metrics if we have daily returns
    if 'daily_returns' in df.columns:
        df['sharpe_ratio'] = df['daily_returns'].apply(calculate_sharpe_ratio)
        df['max_drawdown'] = df['equity_curve'].apply(calculate_max_drawdown)
    
    # Calculate trade metrics
    if 'num_trades' in df.columns:
        df['avg_trade_pnl'] = df['total_pnl'] / df['num_trades']
        df['profit_factor'] = df['gross_profit'] / df['gross_loss'] if 'gross_profit' in df.columns else None
    
    # Filter out configurations with insufficient trades or poor performance
    df = df[df['num_trades'] >= 5]  # Minimum 5 trades
    df = df[df['total_pnl'] > 0]    # Positive PnL only
    
    # Calculate win rate if available
    if 'win_rate' in df.columns:
        df['win_rate'] = df['win_rate'] * 100
    
    # Add configuration tags
    df['config_type'] = df.apply(lambda row: tag_configuration(row), axis=1)
    
    # Sort by Sharpe ratio if available, otherwise by total PnL
    sort_by = 'sharpe_ratio' if 'sharpe_ratio' in df.columns else 'total_pnl'
    df = df.sort_values(sort_by, ascending=False)
    
    return df

def tag_configuration(row):
    """Tag configurations based on their risk/return profile."""
    if 'sharpe_ratio' in row and 'max_drawdown' in row:
        if row['sharpe_ratio'] > 2 and abs(row['max_drawdown']) < 0.1:
            return 'aggressive'
        elif row['sharpe_ratio'] > 1.5 and abs(row['max_drawdown']) < 0.15:
            return 'balanced'
        else:
            return 'conservative'
    return 'unknown'

def print_top_configs(df, n=3):
    """Print the top n configurations in a readable format."""
    print("\nTop VWAP Configurations:")
    print("-" * 70)
    
    for _, row in df.head(n).iterrows():
        config_str = f"Dev: {row['vwap_dev']:.3f}, SL: {row['stop_loss']:.3f}, TP: {row['take_profit']:.3f}"
        perf_str = f"PnL: ${row['total_pnl']:.2f} | Trades: {row['num_trades']}"
        
        if 'win_rate' in row:
            perf_str += f" | Win Rate: {row['win_rate']:.1f}%"
        if 'sharpe_ratio' in row:
            perf_str += f" | Sharpe: {row['sharpe_ratio']:.2f}"
        if 'max_drawdown' in row:
            perf_str += f" | Max DD: {row['max_drawdown']*100:.1f}%"
        if 'config_type' in row:
            perf_str += f" | Type: {row['config_type']}"
            
        print(f"{config_str} | {perf_str}")

def main():
    # Load results
    results_path = Path("results/vwap_backtest_results.csv")
    if not results_path.exists():
        print("Error: Backtest results file not found!")
        return
    
    df = pd.read_csv(results_path)
    
    # Calculate metrics
    df = calculate_metrics(df)
    
    # Save summary
    summary_path = Path("results/vwap_summary.csv")
    df.to_csv(summary_path, index=False)
    
    # Print top configurations
    print_top_configs(df)
    
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main() 