# Hierarchical Risk Parity (HRP) Implementation

## Paper Validation: "Building Diversified Portfolios that Outperform Out-of-Sample"

**Author:** Marcos López de Prado  
**Date:** October 2025

-----

## 📋 Executive Summary

This repository provides a Python implementation of López de Prado's [Hierarchical Risk Parity (HRP) algorithm](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678), designed to validate its out-of-sample performance against common benchmarks.

**Key Finding:** In our specific test on the **S\&P 500 Financial Sector** from 2021-2024, HRP **did not outperform** the naive 1/N (equal-weight) portfolio. The difference in performance was not statistically significant.

  * **HRP Out-of-Sample Sharpe Ratio: 0.828**
  * **1/N Out-of-Sample Sharpe Ratio: 0.836**

This result is a valuable finding that highlights a critical caveat of the algorithm: **HRP's advantages are minimized in highly correlated, single-sector universes.** This implementation serves as a practical demonstration of both how HRP works and the specific conditions under which its performance is limited.

-----

## 🎯 Core Methodology

### Three-Phase HRP Algorithm

1.  **Tree Clustering**: Hierarchical clustering using a correlation-based distance metric to identify asset clusters.
2.  **Quasi-Diagonalization**: Reordering the covariance matrix so that similar assets are placed together.
3.  **Recursive Bisection**: Top-down weight allocation that splits variance between clusters rather than individual assets.

### Key Innovation

Traditional mean-variance optimization (MVO) is sensitive to estimation errors, especially in the covariance matrix. HRP avoids matrix inversion, a common source of instability, by using a hierarchical structure to allocate weights, resulting in more robust portfolios.

-----

## 🔧 Dependencies

```bash
pip install pandas numpy scipy matplotlib seaborn yfinance scikit-learn
```

**Versions Used:**

  * Python 3.9+
  * pandas 2.0.3
  * numpy 1.24.3
  * scipy 1.11.1
  * yfinance 0.2.28

-----

## 📊 Data

**Source:** Yahoo Finance via `yfinance`  
**Universe:** **S\&P 500 Financial Sector** (97 stocks)  
**Period:** 2021-01-01 to 2024-07-31 (using data from 2020 for initial training)  
**Train/Test Split:** Rolling 252-day training windows with 63-day non-overlapping test periods.

### Data Quality Controls

  * Minimum 80% data availability per ticker.
  * Forward-fill gaps ≤ 5 consecutive days.
  * Remove tickers with \>20% zero-return days (indicating delisting or suspension).

-----

## 🧪 Experimental Design

### 1\. Walk-Forward Analysis (Out-of-Sample)

To rigorously test the paper's claims, we use a walk-forward methodology that prevents lookahead bias.

  * **Training Window:** 252 trading days (\~1 year)
  * **Test Window:** 63 trading days (\~1 quarter)
  * **Rolling:** Non-overlapping test periods for independent evaluation.
  * **Total Tests:** 15 independent periods.

**Why this matters:** In-sample optimization is often misleading. True performance is measured out-of-sample, which this framework is designed to do.

### 2\. Baseline Comparisons

HRP's performance is compared against several standard benchmarks:

  * **1/N (Equal-Weight)**: The primary naive diversification benchmark.
  * **Inverse Volatility**: Weights assets in inverse proportion to their volatility.
  * **Minimum Variance**: Classic Markowitz optimization without return forecasting.
  * **Risk Parity**: Allocates weights so that each asset contributes equally to total portfolio risk.

-----

## 📈 Key Results

### Out-of-Sample Performance (2021-2024)

The comprehensive validation report shows that HRP did not produce a statistically significant improvement over the 1/N benchmark in this test.

| Metric | HRP | 1/N | Inverse Vol | Min Variance |
| :--- | :---: | :---: | :---: | :---: |
| **OOS Sharpe Ratio** | 0.828 | **0.836** | 0.821 | 0.803 |
| **Average Turnover** | **31.1%** | 2.5% | 45.3% | 134.8%|

### Statistical Significance (HRP vs. 1/N)

A bootstrap test on the out-of-sample Sharpe ratios indicates the performance difference is statistically insignificant.

  * **P-value:** 0.3420
  * **Conclusion:** We cannot reject the null hypothesis that the Sharpe ratios are equal.
  * **95% Confidence Interval for Sharpe Difference:** [-0.1363, 0.2167]

### Transaction Cost Impact

Despite its gross performance in this test, HRP's low turnover makes it more resilient to transaction costs compared to more active strategies like Minimum Variance. The net Sharpe ratio for HRP barely declines after costs, whereas the high-turnover Min Variance strategy suffers a much larger penalty.

-----

## 🎓 Key Insights & Analysis

### ⚠️ Important Caveat: When HRP Works vs. Doesn't

The results of this analysis serve as a crucial lesson on the limitations of portfolio construction algorithms. HRP is not universally superior; its effectiveness is context-dependent.

#### HRP Outperforms When:

✅ Assets span multiple **uncorrelated sectors** or asset classes.
✅ The correlation matrix has a **clear hierarchical structure** (i.e., visible blocks).
✅ The correlation regime is relatively **stable**.
✅ The portfolio is high-dimensional, with **30+ diverse assets**.

#### HRP May Underperform When:

❌ The portfolio consists of a **single, highly-correlated sector** (like only financials), where all assets tend to move together.
❌ The market experiences **sudden regime shifts** (e.g., COVID, rate shocks) that destabilize historical correlations.
❌ There are **very few assets** (\<15), minimizing the benefit of clustering.
❌ The evaluation **time period is too short** (\<3 years).

### Our Results Explained

In testing on S\&P 500 **Financials only** (2021-2024):

  * **Result: HRP did NOT outperform 1/N.** ❌
  * **Why?** Financials are a highly correlated group (average correlation of 0.65+). HRP's hierarchical clustering identifies one large cluster, causing its weight allocation to collapse to a near-equal-weight scheme without providing its key diversification benefits.

-----

## 🚀 Running the Code

### Quick Start

```python
from hrp_implementation import main

# Run full pipeline
hrp_weights, results = main(
    csv_path='sample_data/financials.csv',
    start_date='2020-01-01',
    end_date='2024-12-31'
)
```

### Custom Analysis

```python
from hrp_implementation import (
    fetch_historical_data,
    calculate_returns,
    hrp_algorithm,
    walk_forward_analysis
)

# Load your own tickers
tickers = ['AAPL', 'GOOGL', 'MSFT', ...]
prices = fetch_historical_data(tickers)
returns = calculate_returns(prices)

# Out-of-sample backtest
oos_results = walk_forward_analysis(returns, train_window=252, test_window=63)
```

-----

## 📁 Repository Structure

```
.
.
├── sample_data/
│   └── financials.csv       # Sample S&P 500 Financials data
├── tests/
│   └── test_hrp.py          # Unit tests for the HRP functions
├── .gitignore               # Git ignore file
├── README.md                # This file
├── SETUP.md                 # Setup and installation instructions
├── config.yaml              # Configuration for backtest parameters
├── hrp.py                   # Core HRP algorithm logic
├── requirements.txt         # Project dependencies
└── run_analysis.py          # Script to run the full validation backtest
```

-----

## ⚖️ License

MIT License - Free to use for research and commercial applications.

-----

**Validation Conclusion:** ⚠️ **Paper's claim NOT confirmed in this specific test.** This analysis successfully implements the HRP algorithm but demonstrates that its outperformance is not guaranteed. The choice of a single, highly-correlated asset class (S\&P 500 Financials) is a key factor in its failure to beat a simple 1/N benchmark, providing a valuable insight into the practical limitations of the strategy.
