import backtrader as bt
import yfinance as yf

# Moving Average Crossover Strategy
class SmaCross(bt.Strategy):
    params = (("fast", 10), ("slow", 30),)

    def __init__(self):
        sma_fast = bt.indicators.SMA(period=self.p.fast)
        sma_slow = bt.indicators.SMA(period=self.p.slow)
        self.crossover = bt.indicators.CrossOver(sma_fast, sma_slow)

    def next(self):
        if not self.position:  # not in market
            if self.crossover > 0:  # Golden Cross
                self.buy()
                print(f"BUY at {self.data.close[0]:.2f}")
        else:
            if self.crossover < 0:  # Death Cross
                self.sell()
                print(f"SELL at {self.data.close[0]:.2f}")

# Download data
ticker = "AAPL"
data = yf.download(ticker, "2020-01-01", "2024-01-01")
datafeed = bt.feeds.PandasData(dataname=data)

# Backtest
cerebro = bt.Cerebro()
cerebro.addstrategy(SmaCross)
cerebro.adddata(datafeed)
cerebro.broker.setcash(10000)
cerebro.run()
cerebro.plot()
