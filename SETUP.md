# Setup Guide - HRP Implementation

Complete installation and usage guide for the enhanced HRP implementation.

---

## 🚀 Quick Start (5 minutes)

```bash
# 1. Clone/download the repository
git clone <your-repo-url>
cd hrp-implementation

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the analysis
python hrp_implementation_enhanced.py
```

**That's it!** Results will be saved to `outputs/` folder.

---

## 📋 Prerequisites

- **Python 3.9+** (3.11 recommended)
- **pip** package manager
- **8GB RAM** minimum (16GB recommended for large portfolios)
- **Internet connection** (for downloading market data)

---

## 📦 Installation

### Option 1: Standard Installation

```bash
# Create and activate virtual environment
python -m venv venv

# Linux/Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, scipy, yfinance; print('✓ All dependencies installed')"
```

### Option 2: Conda Installation

```bash
# Create conda environment
conda create -n hrp python=3.11
conda activate hrp

# Install dependencies
pip install -r requirements.txt

# Or use conda:
conda install pandas numpy scipy matplotlib seaborn
pip install yfinance pytest
```

### Option 3: Development Installation (with testing)

```bash
pip install -r requirements.txt
pip install -e .  # Editable install

# Run tests
pytest tests/ -v --cov=hrp_implementation_enhanced
```

---

## 🗂️ Project Structure

```
hrp-implementation/
│
├── hrp_implementation_enhanced.py   # Main implementation
├── run_analysis.py                  # CLI runner with options
├── config.yaml                      # Configuration file
├── requirements.txt                 # Python dependencies
├── README.md                        # Project overview
├── SETUP.md                        # This file
│
├── sample_data/
│   └── financials.csv              # Sample ticker list (S&P 500 Financials)
│
├── tests/
│   └── test_hrp.py                 # Unit tests
│
├── outputs/                        # Generated results (created automatically)
│   ├── hrp_weights.csv
│   ├── in_sample_metrics.csv
│   ├── oos_summary.csv
│   ├── statistical_tests.csv
│   ├── turnover_analysis.csv
│   ├── validation_report.txt
│   ├── comprehensive_comparison.png
│   ├── oos_analysis.png
│   ├── statistical_tests.png
│   └── dendrogram.png
│
├── logs/                           # Execution logs (created automatically)
│   └── hrp_execution.log
│
└── notebooks/
    └── example_notebook.ipynb      # Interactive tutorial
```

---

## 🎮 Usage

### Method 1: Direct Python Execution

**Basic usage:**
```python
from hrp_implementation_enhanced import main

# Run with defaults (2020-2024, S&P 500 Financials)
hrp_weights, results = main()
```

**Custom parameters:**
```python
from hrp_implementation_enhanced import main

hrp_weights, results = main(
    csv_path='my_tickers.csv',
    start_date='2018-01-01',
    end_date='2023-12-31',
    output_dir='my_results'
)

# Access results
print(results['HRP']['metrics'])  # In-sample metrics
print(results['oos']['HRP']['oos_returns'])  # Out-of-sample returns
```

### Method 2: Command Line Interface

**Basic usage:**
```bash
python run_analysis.py
```

**With custom parameters:**
```bash
python run_analysis.py \
    --csv sample_data/financials.csv \
    --start-date 2019-01-01 \
    --end-date 2024-06-30 \
    --output-dir results_2019_2024
```

**Quick mode (skip intensive computations):**
```bash
python run_analysis.py --quick
```

**Available options:**
```bash
python run_analysis.py --help

Options:
  --config PATH          Configuration YAML file (default: config.yaml)
  --csv PATH            CSV file with tickers
  --start-date DATE     Start date (YYYY-MM-DD)
  --end-date DATE       End date (YYYY-MM-DD)
  --output-dir PATH     Output directory
  --no-walk-forward     Skip walk-forward analysis
  --no-costs            Skip transaction cost analysis
  --no-tests            Skip statistical tests
  --quick               Quick mode (skip all intensive computations)
  --verbose             Enable verbose output
```

### Method 3: Configuration File

**Edit `config.yaml`:**
```yaml
data:
  csv_path: 'sample_data/financials.csv'
  start_date: '2020-01-01'
  end_date: '2024-12-31'

walk_forward:
  train_window: 252
  test_window: 63

transaction_costs:
  cost_bps: 10.0
  rebalance_freq: 21
```

**Run with config:**
```bash
python run_analysis.py --config config.yaml
```

### Method 4: Jupyter Notebook

```bash
# Install Jupyter
pip install jupyter

# Launch notebook
jupyter notebook notebooks/example_notebook.ipynb
```

---

## 📊 Using Your Own Data

### Option 1: CSV File

Create a CSV with a `Symbol` column:

```csv
Symbol
AAPL
GOOGL
MSFT
TSLA
AMZN
```

Then run:
```python
from hrp_implementation_enhanced import main

hrp_weights, results = main(csv_path='my_tickers.csv')
```

### Option 2: Direct Ticker List

```python
from hrp_implementation_enhanced import (
    fetch_historical_data,
    calculate_returns,
    hrp_algorithm,
    run_complete_comparison
)

# Your tickers
tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META']

# Fetch data
prices = fetch_historical_data(tickers, start_date='2020-01-01', end_date='2024-12-31')
returns = calculate_returns(prices)

# Run HRP
hrp_weights, linkage_matrix, corr_matrix = hrp_algorithm(returns)

# Full comparison
results = run_complete_comparison(returns)
```

### Option 3: Pre-loaded Data

```python
import pandas as pd
from hrp_implementation_enhanced import calculate_returns, hrp_algorithm

# Load your own price data
prices = pd.read_csv('my_prices.csv', index_col=0, parse_dates=True)

# Must be a DataFrame with:
# - DatetimeIndex
# - Columns = asset names
# - Values = prices

returns = calculate_returns(prices)
hrp_weights, _, _ = hrp_algorithm(returns)
```

---

## 🧪 Running Tests

### Basic test run:
```bash
pytest tests/test_hrp.py -v
```

### With coverage:
```bash
pytest tests/ --cov=hrp_implementation_enhanced --cov-report=html
open htmlcov/index.html  # View coverage report
```

### Specific test classes:
```bash
pytest tests/test_hrp.py::TestHRPAlgorithm -v
pytest tests/test_hrp.py::TestStatisticalTests -v
```

### Run all tests with detailed output:
```bash
pytest tests/ -v --tb=short --capture=no
```

---

## 🔧 Troubleshooting

### Issue: yfinance download fails

**Symptom:**
```
Error fetching data: No data retrieved from yfinance
```

**Solutions:**
1. **Check internet connection**
2. **Try fewer tickers** (yfinance rate limits)
3. **Use shorter date range**
4. **Add retry logic:**

```python
import time

for attempt in range(3):
    try:
        prices = fetch_historical_data(tickers)
        break
    except Exception as e:
        if attempt < 2:
            print(f"Retry {attempt + 1}/3...")
            time.sleep(5)
        else:
            raise
```

### Issue: Singular covariance matrix

**Symptom:**
```
LinAlgError: Singular matrix
```

**Solutions:**
1. **Remove perfectly correlated assets**
2. **Use more historical data** (increase date range)
3. **Apply covariance shrinkage:**

```python
from sklearn.covariance import LedoitWolf

lw = LedoitWolf()
shrunk_cov = lw.fit(returns).covariance_ * 252
```

### Issue: Memory error with large portfolios

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Solutions:**
1. **Reduce number of assets** (< 200)
2. **Shorten time period**
3. **Disable walk-forward analysis:**

```bash
python run_analysis.py --no-walk-forward
```

4. **Use chunking for large computations**

### Issue: Tests failing

**Symptom:**
```
FAILED tests/test_hrp.py::TestHRPAlgorithm
```

**Solutions:**
1. **Update dependencies:**
```bash
pip install --upgrade -r requirements.txt
```

2. **Check Python version:**
```bash
python --version  # Should be 3.9+
```

3. **Clear pytest cache:**
```bash
pytest --cache-clear
```

### Issue: Plots not displaying

**Symptom:**
No plots shown when running in terminal

**Solutions:**
1. **Use non-interactive backend:**
```python
import matplotlib
matplotlib.use('Agg')  # Before other matplotlib imports
```

2. **Save plots only:**
```python
plt.savefig('output.png')
# Don't call plt.show()
```

3. **Use Jupyter notebook** for interactive plots

---

## ⚡ Performance Optimization

### For faster execution:

**1. Reduce bootstrap samples:**
```yaml
# config.yaml
statistical_tests:
  bootstrap_samples: 500  # Default: 1000
```

**2. Skip heavy computations:**
```bash
python run_analysis.py --quick
```

**3. Use fewer walk-forward periods:**
```yaml
walk_forward:
  test_window: 126  # 6 months instead of 3
```

**4. Parallel processing (future enhancement):**
```python
from joblib import Parallel, delayed

# Parallel walk-forward
results = Parallel(n_jobs=-1)(
    delayed(walk_forward_single_period)(i) 
    for i in range(n_periods)
)
```

### For better accuracy:

**1. More bootstrap samples:**
```yaml
statistical_tests:
  bootstrap_samples: 5000
```

**2. Smaller test windows:**
```yaml
walk_forward:
  test_window: 21  # 1 month
```

**3. Covariance shrinkage:**
```yaml
hrp:
  covariance_shrinkage: true
```

---

## 📈 Advanced Usage

### Custom Portfolio Constraints

```python
def hrp_with_constraints(returns, max_weight=0.1):
    """HRP with maximum weight constraint."""
    weights, _, _ = hrp_algorithm(returns)
    
    # Apply constraint
    while weights.max() > max_weight:
        excess_idx = weights.idxmax()
        excess = weights[excess_idx] - max_weight
        weights[excess_idx] = max_weight
        
        # Redistribute excess to other assets
        other_weights = weights.drop(excess_idx)
        weights[other_weights.index] += excess * other_weights / other_weights.sum()
    
    return weights / weights.sum()
```

### Multi-Asset Class Extension

```python
# Combine stocks, bonds, commodities
stock_returns = calculate_returns(stock_prices)
bond_returns = calculate_returns(bond_prices)
commodity_returns = calculate_returns(commodity_prices)

all_returns = pd.concat([stock_returns, bond_returns, commodity_returns], axis=1)
hrp_weights, _, _ = hrp_algorithm(all_returns)
```

### Live Trading Integration

```python
def get_latest_weights(lookback_days=252):
    """Get current HRP weights for live trading."""
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=lookback_days * 1.5)
    
    prices = fetch_historical_data(tickers, start_date, end_date)
    returns = calculate_returns(prices)
    
    weights, _, _ = hrp_algorithm(returns.iloc[-lookback_days:])
    return weights

# Use with trading API (e.g., Alpaca, Interactive Brokers)
current_weights = get_latest_weights()
# Execute trades...
```

---

## 🤝 Contributing

Found a bug? Have an improvement? 

1. **Fork the repository**
2. **Create feature branch:** `git checkout -b feature/improvement`
3. **Add tests:** `pytest tests/`
4. **Commit changes:** `git commit -am 'Add improvement'`
5. **Push to branch:** `git push origin feature/improvement`
6. **Submit pull request**

---

## 📚 Additional Resources

### Papers
- [López de Prado (2016) - Building Diversified Portfolios](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678)
- [Ledoit & Wolf (2004) - Covariance Shrinkage](https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-0000-00007f64e6c8/shrinkage.pdf)

### Books
- *Advances in Financial Machine Learning* - Marcos López de Prado
- *Machine Learning for Asset Managers* - Marcos López de Prado

### Online Resources
- [QuantStack HRP Implementation](https://quantdare.com/hierarchical-risk-parity/)
- [PyPortfolioOpt Documentation](https://pyportfolioopt.readthedocs.io/)

---

## 📧 Support

Questions? Issues?

- **GitHub Issues:** [Open an issue](https://github.com/your-repo/issues)
- **Email:** your.email@example.com
- **Documentation:** See README.md

---

## ⚖️ License

MIT License - Free for research and commercial use.

---

**Ready to start?**

```bash
python hrp_implementation_enhanced.py
```

Good luck with your Jane Street challenge! 🚀
