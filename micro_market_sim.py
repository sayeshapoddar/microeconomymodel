
#Micro economy stock market simulation with simple agents and analysis.
#Requirements: Python 3, numpy, pandas, matplotlib.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fixed seed to reproduce results and prove hypothesis.
np.random.seed(42)


# config 
N_ASSETS = 4
ASSET_NAMES = ["STOCK_A", "STOCK_B", "STOCK_C", "STOCK_D"]
INITIAL_PRICES = np.array([100.0, 80.0, 120.0, 60.0])
BASE_VOLATILITY = np.array([0.01, 0.012, 0.009, 0.015])
FUNDAMENTALS = np.array([105.0, 85.0, 115.0, 65.0])

TOTAL_AGENTS = 120
ALPHA = 0.02
TIME_STEPS = 300
MIN_PRICE = 1.0
MOMENTUM_WINDOW = 4

# scheduledd external shocks
SHOCKS = [
    (80, 0, 0.08),    # Positive budget news for STOCK_A at step 80
    (170, 2, -0.1),   # Earnings miss for stock c at  170
    (240, 1, 0.05),   # Sector increase or boost ig for stock b at step 240
]


#creating agent
def create_agents():
    # its basically creating a list of agent dictionaries with type, cash, holdinggs, and also risk
    agents = []

    type_counts = {
        "value": 35,
        "momentum": 30,
        "noise": 30,
        "contrarian": 25,
    }

    for agent_type, count in type_counts.items():
        for _ in range(count):
            agent = {
                "type": agent_type,
                "cash": 10000.0,
                "holdings": np.zeros(N_ASSETS, dtype=int),
                "risk": np.random.uniform(0.5, 1.5),
            }
            agents.append(agent)

    return agents


#Strategy helpers
def value_investor_strategy(prices, fundamentals, agent):
    #this one basically buy undervalued assets, sell overvalued ones
    threshold = 0.05
    orders = np.zeros(N_ASSETS, dtype=int)
    mispricing = (fundamentals - prices) / fundamentals

    for i in range(N_ASSETS):
        if mispricing[i] > threshold:
            orders[i] = int(5 * agent["risk"])  # buy
        elif mispricing[i] < -threshold:
            orders[i] = -int(5 * agent["risk"])  # sell
    return orders


def momentum_trader_strategy(price_history, agent):
    # this one follows recent trends using last MOMENTUM_WINDOW returns 
    orders = np.zeros(N_ASSETS, dtype=int)
    if price_history.shape[0] <= MOMENTUM_WINDOW:
        return orders

    recent_returns = price_history[-MOMENTUM_WINDOW:] / price_history[-MOMENTUM_WINDOW - 1:-1] - 1
    avg_returns = recent_returns.mean(axis=0)

    for i in range(N_ASSETS):
        if avg_returns[i] > 0:
            orders[i] = int(3 * agent["risk"])
        elif avg_returns[i] < 0:
            orders[i] = -int(3 * agent["risk"])
    return orders


def noise_trader_strategy(agent):
    # this one is kind of random to stimulate noise, like small buy/sell/hold 
    orders = np.zeros(N_ASSETS, dtype=int)
    choices = [-1, 0, 1]
    probs = [0.25, 0.5, 0.25]
    for i in range(N_ASSETS):
        action = np.random.choice(choices, p=probs)
        orders[i] = int(action * agent["risk"])
    return orders


def contrarian_strategy(price_history, agent):
    #Buy dips gently, sell sharply after big drops. its basically panicking
    orders = np.zeros(N_ASSETS, dtype=int)
    if price_history.shape[0] < 2:
        return orders

    last_returns = price_history[-1] / price_history[-2] - 1
    for i in range(N_ASSETS):
        if last_returns[i] < -0.04:
            orders[i] = -int(6 * agent["risk"])  # panic sell
        elif last_returns[i] < -0.015:
            orders[i] = int(4 * agent["risk"])   # buy the dip
    return orders


# Simulation
def generate_orders(agents, price_history):
    prices = price_history[-1]
    all_orders = []

    for agent in agents:
        if agent["type"] == "value":
            orders = value_investor_strategy(prices, FUNDAMENTALS, agent)
        elif agent["type"] == "momentum":
            orders = momentum_trader_strategy(price_history, agent)
        elif agent["type"] == "noise":
            orders = noise_trader_strategy(agent)
        else:
            orders = contrarian_strategy(price_history, agent)
        all_orders.append(orders)

    return np.array(all_orders)


def apply_shocks(step, prices, shock_log):
    # this will Apply scheduled price shocks.
    price_shift = np.zeros_like(prices)
    for shock in SHOCKS:
        shock_step, asset_idx, pct = shock
        if step == shock_step:
            price_shift[asset_idx] += pct
            shock_log.append(
                {"step": step, "asset": ASSET_NAMES[asset_idx], "impact_pct": pct}
            )
    return price_shift


def update_prices(prices, net_demand, shocks):
    # this will Update prices using  impact, noise, and shocks basically all that happens
    new_prices = prices.copy()
    for i in range(N_ASSETS):
        noise = np.random.normal(0, BASE_VOLATILITY[i])
        impact = ALPHA * net_demand[i] / TOTAL_AGENTS
        shift = shocks[i]
        raw_price = prices[i] * (1 + impact + noise + shift)
        new_prices[i] = max(raw_price, MIN_PRICE)
    return new_prices


def execute_trades(agents, prices, desired_orders, transaction_log, step):
    # this will Execute trades at current prices with cash/holding constraints that are defined
    for idx, agent in enumerate(agents):
        for asset_idx in range(N_ASSETS):
            order = desired_orders[idx, asset_idx]
            price = prices[asset_idx]

            if order > 0:
                affordable = int(agent["cash"] // price)
                trade_size = min(order, affordable)
                if trade_size > 0:
                    cost = trade_size * price
                    agent["cash"] -= cost
                    agent["holdings"][asset_idx] += trade_size
                    transaction_log.append(
                        {
                            "step": step,
                            "agent_type": agent["type"],
                            "asset": ASSET_NAMES[asset_idx],
                            "action": "buy",
                            "shares": trade_size,
                            "price": price,
                        }
                    )
            elif order < 0:
                available = agent["holdings"][asset_idx]
                trade_size = min(abs(order), available)
                if trade_size > 0:
                    proceeds = trade_size * price
                    agent["cash"] += proceeds
                    agent["holdings"][asset_idx] -= trade_size
                    transaction_log.append(
                        {
                            "step": step,
                            "agent_type": agent["type"],
                            "asset": ASSET_NAMES[asset_idx],
                            "action": "sell",
                            "shares": trade_size,
                            "price": price,
                        }
                    )


def run_simulation():
    # this is the main Main simulation loop
    agents = create_agents()
    price_history = [INITIAL_PRICES]
    net_demand_history = []
    transaction_log = []
    shock_log = []

    for step in range(1, TIME_STEPS + 1):
        prices = price_history[-1]
        desired_orders = generate_orders(agents, np.array(price_history))
        net_demand = desired_orders.sum(axis=0)
        net_demand_history.append(net_demand)

        shocks = apply_shocks(step, prices, shock_log)
        new_prices = update_prices(prices, net_demand, shocks)
        execute_trades(agents, new_prices, desired_orders, transaction_log, step)

        price_history.append(new_prices)

    return (
        np.array(price_history),
        np.array(net_demand_history),
        agents,
        pd.DataFrame(transaction_log),
        pd.DataFrame(shock_log),
    )


# Analysis of the simulation
def compute_agent_wealth(agents, final_prices):
    wealth = []
    for agent in agents:
        holdings_value = np.sum(agent["holdings"] * final_prices)
        total = agent["cash"] + holdings_value
        wealth.append(
            {
                "type": agent["type"],
                "cash": agent["cash"],
                "holdings_value": holdings_value,
                "final_wealth": total,
                "risk": agent["risk"],
            }
        )
    return pd.DataFrame(wealth)


def plot_prices(price_history):
    plt.figure(figsize=(10, 5))
    for i, name in enumerate(ASSET_NAMES):
        plt.plot(price_history[:, i], label=name)
    plt.title("Asset Prices Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_return_hist(price_history, asset_index=0):
    returns = price_history[1:, asset_index] / price_history[:-1, asset_index] - 1
    plt.figure(figsize=(7, 4))
    plt.hist(returns, bins=30, alpha=0.7, color="tab:blue")
    plt.title("Return Distribution for {}".format(ASSET_NAMES[asset_index]))
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_volatility(price_history, window=10, asset_index=0):
    returns = price_history[1:, asset_index] / price_history[:-1, asset_index] - 1
    rolling_std = pd.Series(returns).rolling(window).std()
    plt.figure(figsize=(8, 4))
    plt.plot(rolling_std, color="tab:red")
    plt.title("Rolling Volatility (std) for {}".format(ASSET_NAMES[asset_index]))
    plt.xlabel("Time Step")
    plt.ylabel("Volatility")
    plt.tight_layout()
    plt.show()


def plot_demand_vs_price(price_history, net_demand_history, asset_index=0):
    plt.figure(figsize=(9, 5))
    plt.plot(price_history[1:, asset_index], label="Price", color="tab:blue")
    plt.plot(net_demand_history[:, asset_index], label="Net Demand", color="tab:orange")
    plt.title("Price and Net Demand for {}".format(ASSET_NAMES[asset_index]))
    plt.xlabel("Time Step")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(price_history):
    returns = price_history[1:] / price_history[:-1] - 1
    df_returns = pd.DataFrame(returns, columns=ASSET_NAMES)
    corr = df_returns.corr()

    print("Asset return correlation matrix:")
    print(corr.round(3))

    plt.figure(figsize=(6, 5))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(N_ASSETS), ASSET_NAMES)
    plt.yticks(range(N_ASSETS), ASSET_NAMES)
    plt.title("Return Correlation Heatmap")
    plt.tight_layout()
    plt.show()


# Main (runs everything)
def main():
    price_history, net_demand_history, agents, trades_df, shocks_df = run_simulation()
    final_prices = price_history[-1]

    wealth_df = compute_agent_wealth(agents, final_prices)
    summary = wealth_df.groupby("type")["final_wealth"].agg(["mean", "median", "std"])

    print("Final prices:")
    print(pd.Series(final_prices, index=ASSET_NAMES).round(2))
    print("\nAgent wealth summary by type:")
    print(summary.round(2))
    print("\nNumber of trades executed:", len(trades_df))
    print("Shocks applied:")
    print(shocks_df)

    plot_prices(price_history)
    plot_return_hist(price_history, asset_index=0)
    plot_volatility(price_history, window=10, asset_index=0)
    plot_correlation_heatmap(price_history)
    plot_demand_vs_price(price_history, net_demand_history, asset_index=0)


if __name__ == "__main__":
    main()
