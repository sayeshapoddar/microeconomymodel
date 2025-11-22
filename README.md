A micro economy model

EDIT 13/8/25: PLEASE USE VS CODE OR A SIMILAR CODE EDITOR WHILE RUNNING THE FINAL FILE FOR IT TO WORK. OTHERWISE IT MAY NOT WORK. THE STOCK PYTHON EDITOR DOES NOT WORK AS OF NOW, I AM WORKING ON MAKING IT WORK. THANK YOU


this project contains a simple fakestock market inspired by the Indian stock market. It models a small set of assets and several different agent types that trade over time using rule based strategies. The goal is to study basic price formation, volatility and agent profit/loss patterns inside a controlled micro economy. I've seen a lot of these kinds of projects for US and international markets, so I thought i would try my hand at making one for Indian markets.

Features:

1. Assets
* Three to five synthetic stocks (it varies)
* Initial prices, volatility parameters and fundamental values
* Linear price impact with noise
* Scheduled external shocks. About as random as i could make it

2. Agents
* Value investors
* Momentum traders
* Noise traders
* Panic sellers or contrarians
* Each agent tracks cash, holdings and risk aversion
* Simple buy, sell or hold decisions each step

3. Market rules
* Aggregate net demand determines price change
* Trades execute at the updated price
* Partial fills allowed based on cash and holdings
* Prices protected from becoming negative

4. Outputs
* Price time series for each asset
* Return histograms
* Wealth summary per agent and per agent type
* Volatility over time
* Correlation matrix of asset returns
* Net demand plot for a chosen asset

How to run
1. Install Python with numpy, pandas and matplotlib.
2. Download the main script from this repository.
3. Run:
```
python micro_economy.py
```
4. The script will simulate the market for the configured number of steps and produce plots and printed summaries.

Files

* `micro_economy.py`
  full simulation logic including asset setup, agent strategies, market engine, shock events and analysis. Its just one file

## Reproducibility

The script uses a fixed random seed so results remain consistent between runs. Hopefully. 

I hope you have as much fun messing around with this as i did building it :) if you find any bugs or want to talk, i am availible at sayeshapoddar@gmail. com
