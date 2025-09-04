# Portfolio Optimization with Conditional and Standard Betas

This project implements and analyzes different portfolio optimization strategies using both **conditional betas** and **standard historical betas**. Three portfolio types are considered: **long-only**, **long/short**, and **high-leverage long/short**. The analysis focuses on risk-adjusted performance, diversification, and market stability.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Data](#data)  
3. [Methodology](#methodology)  
   - [Portfolio Types](#portfolio-types)  
   - [Conditional vs Standard Beta](#conditional-vs-standard-beta)  
4. [Results](#results)  
   - [Long-Only Portfolio](#long-only-portfolio)  
   - [Long/Short Portfolio](#longshort-portfolio)  
   - [High-Leverage Long/Short Portfolio](#high-leverage-longshort-portfolio)  
5. [Analysis and Insights](#analysis-and-insights)  
6. [Conclusions](#conclusions)  
7. [Visualizations](#visualizations)  

---

## Project Overview

The goal is to explore portfolio optimization under different constraints and beta adjustments. Specifically, we investigate:

- How **conditional beta** (which considers downside risk) affects portfolio allocation and stability.  
- Differences between **long-only** and **long/short** portfolios.  
- The effect of **leverage** on Sharpe ratios and tail risk.  

---

## Data

- **Equity Universe:** 35 S&P 500 stocks across multiple sectors.  
- **Market Benchmark:** S&P 500 Index (^GSPC).  
- **Conditional Betas:** Derived from historical asset returns considering downside and upside market movements.  
- **Price Data:** Daily adjusted closing prices.  

---

## Methodology

### Portfolio Types

1. **Long-Only Portfolio**  
   - No short positions allowed (`weights ≥ 0`).  
   - Fully invested (`sum(weights) = 1`).  

2. **Long/Short Portfolio**  
   - Short positions allowed.  
   - Dollar-neutral (`sum(weights) = 0`).  
   - Can leverage exposures.  

3. **High-Leverage Long/Short Portfolio**  
   - Long/short positions with leverage applied.  
   - Risk-return amplified; conditional beta used to manage downside risk.  

### Conditional vs Standard Beta

- **Standard Beta:** Historical beta calculated against market returns.  
- **Conditional Beta:** Adjusts for **downside beta**, reducing exposure to assets that fall disproportionately during market stress.  

### Portfolio Optimization

- **Objective:** Maximize Sharpe ratio `(E[R] - Rf) / Volatility`.  
- **Covariance Adjustment:**  
  - Conditional Beta: `Cov_adjusted = Cov * (1 + Beta+^2 + w * Beta-^2)`  
  - Standard Beta: `Cov_adjusted = Cov * (1 + Beta_standard^2)`  
- **Constraints:**  
  - Long-only: `weights ≥ 0` and `sum(weights) = 1`  
  - Long/short: `weights ∈ [-max, max]` and `sum(weights) = 0`  

- **Optimization Method:** Sequential Least Squares Programming (SLSQP).  

---

## Results

### Long-Only Portfolio

- **Observation:** Concentrated holdings in **3–5 stocks** due to Sharpe maximization and long-only constraint.  
- **Conditional vs Standard Beta:**  
  - Conditional beta portfolios **favor low downside beta stocks**, reducing tail risk.  
  - Standard beta portfolios **may overweight high-beta stocks**, increasing expected return but also downside risk.  

### Long/Short Portfolio

- **Observation:** More positions, balanced across **long and short** due to dollar-neutrality.  
- **Conditional vs Standard Beta:**  
  - Conditional beta reduces exposure to high downside beta, improving stability.  
  - Standard beta increases volatility exposure but can capture alpha in bull markets.  

### High-Leverage Long/Short Portfolio

- **Observation:** Leverage amplifies both returns and volatility.  
- **Conditional beta helps mitigate** but does not eliminate tail risk.  
- **Sharpe ratio can increase**, but extreme drawdowns are more likely.  

---

## Analysis and Insights

| Portfolio Type              | Conditional Beta        | Standard Beta          | Notes |
|-----------------------------|-----------------------|-----------------------|-------|
| Long-only                   | Sharpe slightly lower, more robust | Sharpe slightly higher, more concentrated | Reduces tail risk in down markets |
| Long/short                  | More diversified, lower drawdowns | Higher exposure to volatile stocks | Conditional beta improves stability |
| High-leverage long/short    | Lower tail risk but still high | Highest potential Sharpe but risky | Leverage amplifies exposures |

- **Hypotheses:**  
  1. Long-only optimization naturally leads to concentration. Soft diversification constraints could broaden holdings.  
  2. Conditional beta adds stability, particularly in volatile markets.  
  3. Leverage amplifies Sharpe but increases tail risk.  
  4. Long/short strategies capture alpha from overvalued stocks.  

---

## Conclusions

- **Conditional beta portfolios** are more resilient to market downturns.  
- **Long-only portfolios** maximize Sharpe but at the cost of diversification.  
- **Long/short portfolios** are more flexible and diversified, capturing both long and short alpha.  
- **High leverage** can increase returns but also risk; conditional beta partially mitigates downside.  

---

## Visualizations

### Portfolio Allocation

- Conditional Beta Long-Only:

![Conditional Beta Pie Chart](path_to_conditional_pie_chart.png)

- Standard Beta Long-Only:

![Standard Beta Pie Chart](path_to_standard_pie_chart.png)

### Sharpe vs Down-Beta Weight

- Shows Sharpe ratios for different down-beta weight `w` values (conditional beta adjustment).

![Sharpe vs Down-Beta Weight](path_to_sharpe_plot.png)

### Stock Prices Over Time

- Provides historical price context for the assets analyzed.

![Stock Prices](path_to_stock_prices_plot.png)

---

## References

- Modern Portfolio Theory (Markowitz, 1952)  
- Conditional Beta and Downside Risk literature  
- Python libraries: `numpy`, `pandas`, `scipy.optimize`, `plotly`, `matplotlib`
