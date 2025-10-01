# ========================
# PORTFOLIO TRACKER
# ========================
def simulate_portfolio(prices, weights, initial_budget=1_000_000, rebalance="none"):
    """
    Simulate portfolio evolution with optional rebalancing.
    
    rebalance: "none", "weekly", "monthly", "quarterly"
    """
    dates = prices.index
    
    # normalize weights (long/short sum balanced)
    long_sum = weights[weights > 0].sum()
    short_sum = -weights[weights < 0].sum()
    scale = max(long_sum, short_sum)
    weights = weights / scale
    
    # determine rebalance frequency
    if rebalance == "weekly":
        rebalance_dates = dates[::5]
    elif rebalance == "monthly":
        rebalance_dates = dates[::21]
    elif rebalance == "quarterly":
        rebalance_dates = dates[::63]
    else:
        rebalance_dates = [dates[0]]  # only once at start
    
    # tracking variables
    portfolio_values = []
    portfolio_cash = initial_budget
    shares = None
    
    for date, row in prices.iterrows():
        prices_today = row.values
        
        # first allocation
        if shares is None:
            allocation = portfolio_cash * weights
            shares = allocation / prices_today
        
        # current portfolio value
        position_values = shares * prices_today
        long_val = position_values[weights > 0].sum()
        short_val = position_values[weights < 0].sum()
        
        proceeds_short = (-weights[weights < 0] * portfolio_cash).sum()
        current_short = -position_values[weights < 0].sum()
        short_pnl = proceeds_short - current_short
        
        total_value = long_val + short_pnl
        portfolio_values.append(total_value)
        
        # rebalance if date matches
        if date in rebalance_dates:
            portfolio_cash = total_value
            allocation = portfolio_cash * weights
            shares = allocation / prices_today
    
    portfolio_series = pd.Series(portfolio_values, index=dates, name="Portfolio Value")
    return portfolio_series

# ========================
# PERFORMANCE SUMMARY
# ========================
def portfolio_stats(portfolio_series, rf_annual=0.0402):
    """
    Compute detailed portfolio metrics with US 3-month Treasury as risk-free.
    """
    returns = portfolio_series.pct_change().dropna()
    
    # Convert annual risk-free to daily
    rf_daily = (1 + rf_annual)**(1/252) - 1
    
    # Excess returns for Sharpe
    excess_returns = returns - rf_daily
    
    # Metrics
    total_return = portfolio_series.iloc[-1] / portfolio_series.iloc[0] - 1
    annualized_return = (1 + total_return) ** (252/len(returns)) - 1
    vol_daily = returns.std()
    vol_annual = vol_daily * np.sqrt(252)
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    # Drawdowns
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdowns = (cum - running_max) / running_max
    max_dd = drawdowns.min()
    
    # Weekly & monthly returns
    weekly_returns = returns.resample("W").sum()
    monthly_returns = returns.resample("M").sum()
    
    stats = {
        "Total Return": f"{total_return:.2%}",
        "Annualized Return (CAGR)": f"{annualized_return:.2%}",
        "Daily Volatility": f"{vol_daily:.2%}",
        "Annualized Volatility": f"{vol_annual:.2%}",
        "Sharpe Ratio (excess)": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Best Daily Return": f"{returns.max():.2%}",
        "Worst Daily Return": f"{returns.min():.2%}",
        "Mean Daily Return": f"{returns.mean():.2%}",
        "Median Daily Return": f"{returns.median():.2%}"
    }
    
    return stats, weekly_returns, monthly_returns

# ========================
# EXAMPLE USAGE
# ========================
portfolio_none = simulate_portfolio(prices[tickers], best_weights, rebalance="none")
portfolio_weekly = simulate_portfolio(prices[tickers], best_weights, rebalance="weekly")
portfolio_monthly = simulate_portfolio(prices[tickers], best_weights, rebalance="monthly")


# Compute stats
stats_none, weekly_none, monthly_none = portfolio_stats(portfolio_none)
stats_weekly, weekly_weekly, monthly_weekly = portfolio_stats(portfolio_weekly)
stats_monthly, weekly_monthly, monthly_monthly = portfolio_stats(portfolio_monthly)


# Print example
print("=== Buy & Hold Performance ===")
for k,v in stats_none.items():
    print(f"{k:25s} {v}")
print("\n=== Weekly Rebalanced Performance ===")
for k,v in stats_weekly.items():
    print(f"{k:25s} {v}")
print("\n=== Monthly Rebalanced Performance ===")
for k,v in stats_monthly.items():
    print(f"{k:25s} {v}")

# Benchmark
all_prices = pd.read_pickle("sp500_prices.pkl")
if "^GSPC" in all_prices.columns:
    sp500 = all_prices["^GSPC"] / all_prices["^GSPC"].iloc[0] * 1_000_000

# ========================
# PLOT
# ========================
plt.figure(figsize=(12,6))
plt.plot(portfolio_none, label="Buy & Hold (No Rebalancing)")
plt.plot(portfolio_weekly, label="Weekly Rebalancing")
plt.plot(portfolio_monthly, label="Monthly Rebalancing")

if "^GSPC" in all_prices.columns:
    plt.plot(sp500, label="S&P500 (scaled)", linestyle="--")
plt.title("Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Value ($)")
plt.legend()
plt.show()