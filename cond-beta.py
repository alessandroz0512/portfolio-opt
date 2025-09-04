# ------------------------
# CONDITIONAL BETA DATASET FROM EXISTING DATA EXTRACTION
# ------------------------

import pandas as pd
import numpy as np

# ------------------------
# Step 1: Assuming you already have `prices` DataFrame from your code
prices = pd.DataFrame(price_data).dropna(how="all")
# ------------------------

# Calculate returns
returns = prices.pct_change().dropna()

# Market returns: S&P500
market_ret = returns["^GSPC"]

# ------------------------
# Step 2: Conditional Beta Function
# ------------------------
def conditional_beta(stock_ret, market_ret):
    up = market_ret > 0
    down = market_ret < 0
    if stock_ret[up].empty or stock_ret[down].empty:
        return np.nan, np.nan
    beta_plus = np.cov(stock_ret[up], market_ret[up])[0,1] / np.var(market_ret[up])
    beta_minus = np.cov(stock_ret[down], market_ret[down])[0,1] / np.var(market_ret[down])
    return beta_plus, beta_minus

# ------------------------
# Step 3: Compute Conditional Betas for All Tickers
# ------------------------
beta_data = []
for t in prices.columns:
    if t == "^GSPC":
        continue
    b_plus, b_minus = conditional_beta(returns[t], market_ret)
    beta_data.append({
        "Ticker": t,
        "Beta+": b_plus,
        "Beta-": b_minus,
        "Beta_Ratio": b_plus / b_minus if b_minus != 0 else np.nan
    })

beta_df = pd.DataFrame(beta_data)

# ------------------------
# Step 4: Save Dataset for Portfolio Optimization
# ------------------------
beta_df.to_csv("conditional_beta_dataset.csv", index=False)
print("Conditional beta dataset saved as conditional_beta_dataset.csv")
