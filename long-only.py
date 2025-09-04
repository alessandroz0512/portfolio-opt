# -----------------------------
# LONG-ONLY CONDITIONAL BETA PORTFOLIO WORKFLOW
# -----------------------------
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go

# -----------------------------
# PARAMETERS
# -----------------------------
tickers = [
    "AAPL","MSFT","NVDA","GOOGL","META",
    "JPM","BAC","WFC","C","GS",
    "JNJ","PFE","MRK","ABBV","LLY",
    "CAT","BA","GE","HON","UNP",
    "AMZN","TSLA","HD","MCD","NKE",
    "XOM","CVX","COP","SLB","EOG",
    "PG","KO","PEP","WMT","COST"
]
rf_rate = 0.02
down_weight = 1.5  # conditional beta adjustment
max_weight = 0.4  # max 40% per stock to increase diversification

# -----------------------------
# STEP 1: LOAD DATA
# -----------------------------
prices = pd.read_pickle("sp500_prices.pkl")
prices = prices[[t for t in tickers if t in prices.columns]]
returns = prices.pct_change().dropna()
expected_returns = returns.mean().values * 252
cov_matrix = returns.cov().values * 252  # annualized covariance

# Conditional beta dataset
beta_df = pd.read_csv("conditional_beta_dataset.csv")
beta_df = beta_df.set_index("Ticker").loc[prices.columns]
beta_plus = beta_df["Beta+"].values
beta_minus = beta_df["Beta-"].values

# -----------------------------
# STEP 2: ADJUST COVARIANCE MATRIX
# -----------------------------
adj = beta_plus**2 + down_weight * beta_minus**2
adjusted_cov = cov_matrix * np.outer(1 + adj, 1 + adj)

# -----------------------------
# STEP 3: PORTFOLIO FUNCTIONS
# -----------------------------
def portfolio_performance(weights, mu, cov, rf=rf_rate):
    ret = np.dot(weights, mu)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    sharpe = (ret - rf)/vol if vol != 0 else 0
    return ret, vol, sharpe

def neg_sharpe(weights, mu, cov, rf=rf_rate):
    # Standard negative Sharpe objective
    return -portfolio_performance(weights, mu, cov, rf)[2]

def optimize_portfolio(mu, cov, max_weight=max_weight):
    n = len(mu)
    init_guess = np.ones(n)/n
    bounds_list = [(0, max_weight)]*n  # long-only + max weight constraint
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # fully invested
    result = minimize(neg_sharpe, init_guess, args=(mu, cov),
                      method='SLSQP', bounds=bounds_list, constraints=constraints)
    return result.x

# -----------------------------
# STEP 4: RUN OPTIMIZATION
# -----------------------------
weights = optimize_portfolio(expected_returns, adjusted_cov)
ret, vol, sharpe = portfolio_performance(weights, expected_returns, adjusted_cov)

# -----------------------------
# STEP 5: OUTPUT TABLE FUNCTION
# -----------------------------
def make_weights_table(weights, name="Portfolio"):
    df = pd.DataFrame({
        "Ticker": prices.columns,
        f"{name} Weight": weights.round(4)
    })
    df["Expected Contribution"] = (weights * expected_returns).round(4)
    total_row = pd.DataFrame({
        "Ticker": ["Total"],
        f"{name} Weight": [df[f"{name} Weight"].sum().round(4)],
        "Expected Contribution": [df["Expected Contribution"].sum().round(4)]
    })
    df = pd.concat([df, total_row], ignore_index=True)
    return df

table = make_weights_table(weights, "Conditional Beta")
print("Conditional Beta Long-Only Portfolio:")
print(table)
print(f"Return: {ret:.3f}, Vol: {vol:.3f}, Sharpe: {sharpe:.3f}\n")

# -----------------------------
# STEP 6: COMPARE TO S&P500
# -----------------------------
try:
    market_prices = pd.read_pickle("sp500_prices.pkl")["^GSPC"]
    market_returns = market_prices.pct_change().dropna()
    market_mean = market_returns.mean() * 252
    market_vol = market_returns.std() * np.sqrt(252)
    market_sharpe = (market_mean - rf_rate) / market_vol
    print("S&P500 Performance:")
    print(f"Return: {market_mean:.3f}, Vol: {market_vol:.3f}, Sharpe: {market_sharpe:.3f}")
except KeyError:
    print("S&P500 prices not available in dataset.")

# -----------------------------
# STEP 7: PLOT PORTFOLIO ALLOCATION
# -----------------------------
def plot_portfolio_pie(weights, tickers, title="Portfolio Allocation"):
    fig = go.Figure(data=[go.Pie(
        labels=tickers,
        values=weights,
        hoverinfo='label+percent',
        textinfo='label+percent',
        textfont_size=18,
        hole=0.3
    )])
    fig.update_layout(
        title_text=title,
        title_font_size=24,
        height=800,
        width=800,
    )
    fig.show()

plot_portfolio_pie(weights, prices.columns, "Conditional Beta Long-Only Allocation")

# -----------------------------
# STEP 8: PLOT STOCK PRICE HISTORY
# -----------------------------
def plot_stock_prices(prices, tickers, title="Stock Prices Over Time"):
    fig = go.Figure()
    for t in tickers:
        fig.add_trace(go.Scatter(
            x=prices.index,
            y=prices[t],
            mode='lines',
            name=t
        ))
    fig.update_layout(
        title=title,
        title_font_size=24,
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        height=800,
        width=1200
    )
    fig.show()

plot_stock_prices(prices, prices.columns, "Stock Prices Over Time")
