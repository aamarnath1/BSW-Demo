# 🧠 Auto-Investor Bot – Modular Trading Framework

This project aims to build a **modular, adaptive trading bot** that dynamically selects and runs the best-performing strategies in live markets. Think of it as a strategy engine powered by logic and data — similar to a retrieval-augmented generator (RAG), but for trading.

---

## 🔷 Project Phases

### ✅ Phase 1: Intraday Strategy Engine

You are currently focused on backtesting and optimizing 3 intraday strategies:

---

### 📊 Strategy 1: VWAP Reversion

- **Timeframe**: 1-minute intraday bars
- **Entry**:  
  - Long when price drops more than X% below VWAP and crosses back above VWAP  
  - Short when price rises X% above VWAP and crosses back below
- **Exit**:  
  - Price returns to VWAP  
  - OR Y minutes passed  
  - OR Stop-loss / Take-profit triggered  
- **Parameters to Backtest**:
  - VWAP deviation (%)
  - Stop-loss (%)
  - Take-profit (%)

---

### 📊 Strategy 2: Open Range Breakout (ORB)

- **Timeframe**: 1-minute or 5-minute bars
- **Opening Range**: High/Low of first 30 minutes after open  
- **Entry**:  
  - Long if price breaks above OR high  
  - Short if price breaks below OR low  
- **Exit**:  
  - When price re-enters range  
  - OR after fixed duration  
  - OR SL/TP hit (e.g., 10% SL / 20% TP)  
- **Parameters to Test**:
  - Risk/reward ratio
  - Exit time
  - Range buffer (optional)

---

### 📊 Strategy 3: RSI Oversold Reversal

- **Timeframe**: 1-minute or 5-minute bars
- **Entry**:  
  - RSI < 30  
  - (Optional) bullish candle or price dip confirmation  
- **Exit**:  
  - RSI > 50  
  - OR Stop-loss / Take-profit hit  
- **Parameters to Backtest**:
  - RSI period (e.g., 7, 14, 21)
  - Stop-loss (%)
  - Take-profit (%)

---

## 🛠 Dev Steps – For Each Strategy

1. **Define Rules Clearly**  
   Write entry/exit/risk logic in plain English first.
2. **Build Strategy Class**  
   Use Backtrader or a modular Python function.
3. **Backtest Parameter Grid**  
   Try 3 values for each: RSI period, SL%, TP%, etc.
4. **Save Results**  
   Output to `results/` as CSV.
5. **Evaluate Best Configs**  
   Look at:
   - PnL
   - Drawdown
   - Trade count
   - Edge stability
6. **Cross-Validation**  
   Test top configs on:
   - Other tickers (SPY, TSLA, MSFT)
   - Different days/timeframes
   - Choppy vs trending conditions
7. **Behavioral Analysis**  
   Understand *why* it works and when it fails.
8. **Repeat For All 3 Strategies**

---

## 🔁 Next Phase: Meta-Strategy Selector

Eventually build a smart selector to:
- Choose strategy based on market regime
- React to volatility, volume spikes, time-of-day, or news flow
- Combine strategies via:
  - Voting ensemble
  - Meta-model (ML)
  - Dynamic switching

---

## 🗂 Folder Structure
auto_trader/
├── config.py
├── data/
├── results/
├── strategies/
│ ├── vwap_reversion.py
│ ├── open_range_breakout.py
│ └── rsi_oversold.py
├── backtest/
│ └── runner.py
├── utils/
│ ├── get_data.py
│ └── data_loader.py
├── meta/
│ └── strategy_selector.py # (for later)

---

## ✅ Tools

- Python 3.11+
- [Backtrader](https://www.backtrader.com/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- Alpaca API (for real-time paper trading, optional)
- Pandas, itertools, matplotlib (standard data cleaning)

---