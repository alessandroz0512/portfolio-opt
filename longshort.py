# -----------------------------
# LONG/SHORT DOLLAR-NEUTRAL CONDITIONAL BETA PORTFOLIO
# -----------------------------
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px

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
weight_bound = 0.3
allow_short = True
dollar_neutral = True

# -----------------------------
# STEP 1: LOAD DATA
# -----------------------------
prices = pd.read_pickle("sp500_prices.pkl")
prices = prices[[t for t in tickers if t in prices.columns]]
returns = prices.pct_change().dropna()
expected_returns = returns.mean().values * 252
cov_matrix = returns.cov().values * 252

beta_df = pd.read_csv("conditional_beta_dataset.csv")
beta_df = beta_df.set_index("Ticker").loc[prices.columns]
beta_plus = beta_df["Beta+"].values
beta_minus = beta_df["Beta-"].values

# -----------------------------
# STEP 2: PORTFOLIO FUNCTIONS
# -----------------------------
def portfolio_performance(weights, mu, cov, rf=rf_rate):
    ret = np.dot(weights, mu)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    sharpe = (ret - rf)/vol if vol != 0 else 0
    return ret, vol, sharpe

def neg_sharpe(weights, mu, cov, rf=rf_rate):
    return -portfolio_performance(weights, mu, cov, rf)[2]

def optimize_portfolio(mu, cov, bounds, allow_short=True, dollar_neutral=True, init_guess=None):
    n = len(mu)
    if init_guess is None:
        init_guess = np.zeros(n)
        half = n // 2
        init_guess[:half] = bounds/2
        init_guess[half:] = -bounds/2
    
    bounds_list = [(-bounds,bounds)]*n if allow_short else [(0,bounds)]*n
    constraints = []
    if dollar_neutral:
        constraints = [{'type':'eq', 'fun': lambda w: np.sum(w)}]

    result = minimize(neg_sharpe, init_guess, args=(mu,cov),
                      method='SLSQP', bounds=bounds_list, constraints=constraints)
    w_opt = result.x
    long_sum = w_opt[w_opt>0].sum()
    short_sum = -w_opt[w_opt<0].sum()
    if max(long_sum, short_sum) > 1:
        w_opt /= max(long_sum, short_sum)
    return w_opt

# -----------------------------
# STEP 3: GRID SEARCH OVER DOWN-BETA WEIGHT
# -----------------------------
w_values = np.linspace(0, 1.5, 15)  
sharpe_list = []
best_sharpe = -np.inf
best_w = None
best_weights = None
best_cov = None

for w in w_values:
    adjusted_cov = cov_matrix * (1 + beta_plus**2 + w*beta_minus**2)[:, None]
    adjusted_cov = (adjusted_cov + adjusted_cov.T)/2
    weights = optimize_portfolio(expected_returns, adjusted_cov, weight_bound,
                                 allow_short=True, dollar_neutral=True)
    _, _, sharpe = portfolio_performance(weights, expected_returns, adjusted_cov)
    sharpe_list.append(sharpe)
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_w = w
        best_weights = weights
        best_cov = adjusted_cov

print(f"Optimal down-beta weight: {best_w:.2f}")
print(f"Conditional Beta Portfolio Sharpe: {best_sharpe:.3f}")

# -----------------------------
# STEP 4: STANDARD BETA PORTFOLIO
# -----------------------------
weights_beta = optimize_portfolio(expected_returns, cov_matrix, weight_bound,
                                  allow_short=True, dollar_neutral=True)
ret_b, vol_b, sharpe_b = portfolio_performance(weights_beta, expected_returns, cov_matrix)

# -----------------------------
# STEP 5: TABLE FUNCTION WITH LONG/SHORT SUMS
# -----------------------------
def make_weights_table(weights, name):
    df = pd.DataFrame({
        "Ticker": prices.columns,
        f"{name} Weight": weights.round(4)
    })
    df["Expected Contribution"] = (weights * expected_returns).round(4)
    sum_long = df[f"{name} Weight"][df[f"{name} Weight"]>0].sum().round(4)
    sum_short = df[f"{name} Weight"][df[f"{name} Weight"]<0].sum().round(4)
    
    total_row = pd.DataFrame({
        "Ticker":["Total"],
        f"{name} Weight":[df[f"{name} Weight"].sum().round(4)],
        "Expected Contribution":[df["Expected Contribution"].sum().round(4)]
    })
    summary_row = pd.DataFrame({
        "Ticker":["Long/Short Sum"],
        f"{name} Weight":[f"Long: {sum_long}, Short: {sum_short}"],
        "Expected Contribution":[""]
    })
    df = pd.concat([df, total_row, summary_row], ignore_index=True)
    return df

table_conditional = make_weights_table(best_weights, "Conditional Beta")
table_beta = make_weights_table(weights_beta, "Standard Beta")

print("\nConditional Beta Portfolio:")
print(table_conditional)
ret_c, vol_c, sharpe_c = portfolio_performance(best_weights, expected_returns, best_cov)
print(f"Return: {ret_c:.3f}, Vol: {vol_c:.3f}, Sharpe: {sharpe_c:.3f}\n")

print("Standard Beta Portfolio:")
print(table_beta)
print(f"Return: {ret_b:.3f}, Vol: {vol_b:.3f}, Sharpe: {sharpe_b:.3f}\n")

# -----------------------------
# STEP 6: COMPARE TO S&P500
# -----------------------------
market_prices = pd.read_pickle("sp500_prices.pkl")["^GSPC"]
market_returns = market_prices.pct_change().dropna()
market_mean = market_returns.mean() * 252
market_vol = market_returns.std() * np.sqrt(252)
market_sharpe = (market_mean - rf_rate)/market_vol

print(f"S&P500 Return: {market_mean:.3f}, Vol: {market_vol:.3f}, Sharpe: {market_sharpe:.3f}")
print(f"Conditional Beta Sharpe vs S&P500: {sharpe_c:.3f} | {market_sharpe:.3f}")
print(f"Standard Beta Sharpe vs S&P500: {sharpe_b:.3f} | {market_sharpe:.3f}")



# -----------------------------
# STEP 7: LONG/SHORT PIE CHARTS (Plotly)
# -----------------------------
from plotly.subplots import make_subplots

long_weights = np.array([w if w>0 else 0 for w in best_weights])
short_weights = np.array([-w if w<0 else 0 for w in best_weights])

long_labels = np.array(prices.columns)[long_weights>0]
long_values = long_weights[long_weights>0]

short_labels = np.array(prices.columns)[short_weights>0]
short_values = short_weights[short_weights>0]

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                    subplot_titles=['Long Positions', 'Short Positions'])

fig.add_trace(go.Pie(labels=long_labels, values=long_values, name='Long Positions',
                     hole=0.3, marker=dict(colors=px.colors.sequential.Greens)), row=1, col=1)

fig.add_trace(go.Pie(labels=short_labels, values=short_values, name='Short Positions',
                     hole=0.3, marker=dict(colors=px.colors.sequential.Reds)), row=1, col=2)

fig.update_layout(title_text="Long/Short Portfolio Allocation")
fig.show()

# -----------------------------
# STEP 8: TIME SERIES OF STOCKS (Plotly)
# -----------------------------
# Color code: green for long, red for short
colors = ["green" if w>0 else "red" for w in best_weights]

fig = go.Figure()
for i, t in enumerate(prices.columns):
    fig.add_trace(go.Scatter(x=prices.index, y=prices[t], mode='lines', name=t, line=dict(color=colors[i])))

fig.update_layout(title="Stock Prices During Period",
                  xaxis_title="Date", yaxis_title="Price")
fig.show()
