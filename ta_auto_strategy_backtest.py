#!/usr/bin/env python3
"""
ta_auto_strategy_backtest.py

- Technical indicators: RSI (14), MACD (12,26,9), Bollinger Bands (20,2)
- Auto-generate BUY/SELL signals by combining:
    BUY  when: RSI < 30 (oversold) AND MACD histogram crosses above zero AND Close < Bollinger lower (optional)
    SELL when: RSI > 70 (overbought) AND MACD histogram crosses below zero AND Close > Bollinger upper (optional)
- Backtests with backtrader and prints performance metrics + saves signals CSV.

Requirements:
pip install yfinance backtrader pandas numpy matplotlib
"""

import argparse
import datetime as dt
import math
import os
import sys
from typing import List, Dict

import backtrader as bt
import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# Strategy
# -----------------------------
class TAConfirmStrategy(bt.SignalStrategy):
    """
    Strategy that uses RSI + MACD histogram crossover + Bollinger Bands to generate signals.
    - Entry (BUY) when:
        * RSI < rsi_low (default 30)
        * MACD histogram crosses above 0 (i.e. from negative to positive)
        * (optional) price < BB.bot  (we include it as confirmation)
    - Exit (SELL) when:
        * RSI > rsi_high (default 70)
        * MACD histogram crosses below 0
        * (optional) price > BB.top
    The strategy will log signals and trades for analysis.
    """

    params = dict(
        rsi_period=14,
        rsi_low=30,
        rsi_high=70,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        bb_period=20,
        bb_devfactor=2,
        stake=0,  # 0 -> use full-allocation sizing inside next()
        printlog=True,
    )

    def log(self, txt, dt=None):
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()} {txt}")

    def __init__(self):
        d = self.datas[0]
        # Indicators
        self.rsi = bt.indicators.RSI(d.close, period=self.p.rsi_period)
        self.macd = bt.indicators.MACD(d.close,
                                      period_me1=self.p.macd_fast,
                                      period_me2=self.p.macd_slow,
                                      period_signal=self.p.macd_signal)
        # MACD histogram (macd - macd.signal) available as self.macd.histo in backtrader
        self.macdh = bt.indicators.MACDHisto(d.close,
                                            period_me1=self.p.macd_fast,
                                            period_me2=self.p.macd_slow,
                                            period_signal=self.p.macd_signal)
        self.bb = bt.indicators.BollingerBands(d.close,
                                               period=self.p.bb_period,
                                               devfactor=self.p.bb_devfactor)

        # Crossovers
        self.macd_hist_cross = bt.indicators.CrossOver(self.macdh, 0)  # +1 when crosses up, -1 crosses down

        # To store signals for output
        self.signals = []  # list of dicts: {'date':..., 'signal': 'BUY'/'SELL', 'price':...}

        # Track order/trade
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}")
            elif order.issell():
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f"TRADE PnL, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}")

    def next(self):
        dt = self.datas[0].datetime.date(0)
        price = float(self.datas[0].close[0])

        # Conditions
        rsi_val = float(self.rsi[0]) if not np.isnan(self.rsi[0]) else None
        macd_cross = int(self.macd_hist_cross[0])  # 1 = cross up, -1 = cross down, 0 = nothing
        bb_low = float(self.bb.lines.bot[0])
        bb_high = float(self.bb.lines.top[0])

        # Logging not necessary every bar, but helpful when debugging
        # self.log(f"Price {price:.2f} RSI {rsi_val} MACDcross {macd_cross} BBlow {bb_low:.2f} BBhigh {bb_high:.2f}")

        # If we are not in the market
        if not self.position:
            buy_cond = False
            # Require RSI oversold AND MACD histogram crossing up
            if rsi_val is not None and rsi_val < self.p.rsi_low and macd_cross == 1:
                buy_cond = True
            # Extra confirmation: price below lower BB strengthens the buy
            if buy_cond and price < bb_low:
                pass  # stronger confirmation (we already set buy_cond True)
            if buy_cond:
                # Determine size: either stake param or use full cash
                if self.p.stake > 0:
                    size = self.p.stake
                else:
                    # buy with half the portfolio (risk control)
                    cash = self.broker.getcash()
                    alloc = cash * 0.5
                    size = int(alloc / price)
                    if size <= 0:
                        return
                self.log(f"SIGNAL BUY at {price:.2f} (RSI {rsi_val:.2f}, macd_cross {macd_cross})")
                self.signals.append({"date": dt.isoformat(), "signal": "BUY", "price": price, "rsi": rsi_val, "macd_cross": macd_cross, "bb_low": bb_low, "bb_high": bb_high})
                self.order = self.buy(size=size)
        else:
            # In the market: check sell conditions
            sell_cond = False
            if rsi_val is not None and rsi_val > self.p.rsi_high and macd_cross == -1:
                sell_cond = True
            # Extra confirmation: price above upper BB strengthens the sell
            if sell_cond and price > bb_high:
                pass
            if sell_cond:
                self.log(f"SIGNAL SELL at {price:.2f} (RSI {rsi_val:.2f}, macd_cross {macd_cross})")
                self.signals.append({"date": dt.isoformat(), "signal": "SELL", "price": price, "rsi": rsi_val, "macd_cross": macd_cross, "bb_low": bb_low, "bb_high": bb_high})
                self.order = self.sell(size=self.position.size)


# -----------------------------
# Helper: run backtest
# -----------------------------
def run_backtest(ticker: str, start: str, end: str, cash: float = 100000.0, commission: float = 0.001):
    # Fetch data via yfinance and convert to backtrader feed (Pandas)
    print(f"Downloading {ticker} from {start} to {end} ...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise SystemExit("No data downloaded. Check ticker and date range.")

    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df.index = pd.to_datetime(df.index)

    # Create Cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # Feed
    datafeed = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(datafeed, name=ticker)

    # Strategy
    strat = cerebro.addstrategy(TAConfirmStrategy)

    # Run
    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
    results = cerebro.run()
    strat_res = results[0]

    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())

    # Extract analyzers
    sharpe = strat_res.analyzers.sharpe.get_analysis()
    dd = strat_res.analyzers.drawdown.get_analysis()
    trades = strat_res.analyzers.trades.get_analysis()

    # Print key metrics
    print("\n=== Performance Summary ===")
    try:
        print("Sharpe Ratio:", round(sharpe.get("sharperatio", float("nan")), 4))
    except Exception:
        print("Sharpe Ratio: n/a")
    print(f"Max Drawdown: {dd.get('max', {}).get('drawdown', dd.get('maxdrawdown', 'n/a'))}%")
    print("Total trades:", trades.total.closed if hasattr(trades.total, 'closed') else trades.get('total', {}).get('closed', 'n/a'))

    # Save signals to CSV
    signals = strat_res.signals if hasattr(strat_res, "signals") else []
    if signals:
        outdf = pd.DataFrame(signals)
        fn = f"signals_{ticker}.csv"
        outdf.to_csv(fn, index=False)
        print(f"Signals saved to {fn}")
    else:
        print("No signals recorded by strategy.")

    # Plot (optional)
    try:
        cerebro.plot(iplot=False)  # set iplot=False to avoid interactive in some environments
    except Exception as e:
        print("Plotting failed (headless environment?).", e)

    return {"final_value": cerebro.broker.getvalue(), "signals": signals, "sharpe": sharpe, "drawdown": dd, "trades": trades}


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Technical Indicators Auto-Strategy Backtest")
    p.add_argument("--ticker", type=str, required=True, help="Ticker symbol (e.g. AAPL)")
    p.add_argument("--start", type=str, default="2018-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=dt.datetime.today().strftime("%Y-%m-%d"), help="End date YYYY-MM-DD")
    p.add_argument("--cash", type=float, default=100000.0, help="Starting cash")
    p.add_argument("--commission", type=float, default=0.001, help="Commission (fraction), e.g. 0.001 = 0.1%")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_backtest(args.ticker, args.start, args.end, cash=args.cash, commission=args.commission)
    print("\nDone.")
          
