# ------------------------
# SNIPPET 1: DOWNLOAD AND SAVE PRICE DATA
# ------------------------

import yfinance as yf
import pandas as pd

# ------------------------
# Step 1: Tickers & Market Index
# ------------------------
tickers = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "JPM", "BAC", "WFC", "C", "GS",
    "JNJ", "PFE", "MRK", "ABBV", "LLY",
    "CAT", "BA", "GE", "HON", "UNP",
    "AMZN", "TSLA", "HD", "MCD", "NKE",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "PG", "KO", "PEP", "WMT", "COST"
]
market_index = "^GSPC"
all_tickers = tickers + [market_index]

# ------------------------
# Step 2: Download Prices
# ------------------------
price_data = {}
for t in all_tickers:
    try:
        data = yf.download(t, start="2020-01-01", end="2025-01-01", progress=False, auto_adjust=True)
        if "Adj Close" in data.columns:
            adj = data["Adj Close"]
        else:
            adj = data.iloc[:,0]
        price_data[t] = adj
    except Exception as e:
        print(f"Could not download {t}: {e}")

# Convert dictionary to DataFrame
prices = pd.DataFrame(price_data).dropna(how="all")

# ------------------------
# Step 3: Save for reuse
# ------------------------
prices.to_pickle("sp500_prices.pkl")
print("Download complete and saved to sp500_prices.pkl")
