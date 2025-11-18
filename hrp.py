import pandas as pd
import numpy as np
import yfinance as yf
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
import time
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path



warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    start_date: str = '2020-01-01'
    end_date: str = '2024-12-31'
    min_data_pct: float = 0.8
    
    train_window: int = 252
    test_window: int = 63
    
    cost_bps: float = 10.0
    rebalance_freq: int = 21
    
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    
    output_dir: Path = Path('outputs')
    data_dir: Path = Path('data')

config = Config()

def load_tickers(csv_path: str = 'data/financials.csv') -> List[str]:
    try:
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        if 'Symbol' not in df.columns:
            raise ValueError("CSV must contain 'Symbol' column")
        
        tickers = df['Symbol'].dropna().unique().tolist()
        
        if len(tickers) == 0:
            raise ValueError("No tickers found in CSV")
        
        logger.info(f"✓ Loaded {len(tickers)} tickers from {csv_path}")
        return tickers
        
    except Exception as e:
        logger.error(f"Error loading tickers: {e}")
        raise

def fetch_historical_data(
    tickers: List[str],
    start_date: str = '2020-01-01',
    end_date: str = '2024-12-31',
    min_data_pct: float = 0.8,
    timeout: int = 60
) -> pd.DataFrame:
    logger.info(f"Fetching data for {len(tickers)} tickers...")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    try:
        prices = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=True,
            auto_adjust=True,
            timeout=timeout
        )['Close']
        
        if prices.empty:
            raise ValueError("No data retrieved from yfinance")
        
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
        
        initial_tickers = len(prices.columns)
        
        threshold = len(prices) * min_data_pct
        prices = prices.dropna(axis=1, thresh=threshold)
        
        prices = prices.fillna(method='ffill', limit=5)
        
        prices = prices.dropna(axis=1)
        
        returns = prices.pct_change()
        zero_pct = (returns == 0).sum() / len(returns)
        valid_tickers = zero_pct[zero_pct < 0.2].index
        prices = prices[valid_tickers]
        
        final_tickers = len(prices.columns)
        removed = initial_tickers - final_tickers
        
        if final_tickers == 0:
            raise ValueError("No valid tickers remaining after quality checks")
        
        logger.info(f"✓ Data quality checks complete")
        logger.info(f"  - Valid tickers: {final_tickers}")
        logger.info(f"  - Removed: {removed} ({removed/initial_tickers*100:.1f}%)")
        logger.info(f"  - Period: {prices.index[0].date()} to {prices.index[-1].date()}")
        logger.info(f"  - Trading days: {len(prices)}")
        
        return prices
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise

def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna()
    
    if returns.isnull().any().any():
        logger.warning("NaN values detected in returns")
        returns = returns.fillna(0)
    
    extreme = (returns.abs() > 1.0).any()
    if extreme.any():
        extreme_tickers = extreme[extreme].index.tolist()
        logger.warning(f"Extreme returns detected in: {extreme_tickers}")
    
    logger.info(f"✓ Calculated returns: {len(returns.columns)} assets, {len(returns)} days")
    
    return returns

def get_distance_matrix(corr_matrix: pd.DataFrame) -> np.ndarray:
    distance = np.sqrt(0.5 * (1 - corr_matrix))
    return distance

def tree_clustering(returns: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    corr_matrix = returns.corr(method='pearson')
    distance_matrix = get_distance_matrix(corr_matrix)
    dist_condensed = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(dist_condensed, method='single')
    
    return linkage_matrix, corr_matrix

def quasi_diagonalization(linkage_matrix: np.ndarray) -> List[int]:
    dendro = dendrogram(linkage_matrix, no_plot=True)
    sorted_indices = dendro['leaves']
    return sorted_indices

def get_inverse_variance_weights(cov_matrix: pd.DataFrame, indices: List[int]) -> np.ndarray:
    cov_slice = cov_matrix.iloc[indices, indices]
    inv_diag = 1 / np.diag(cov_slice)
    weights = inv_diag / inv_diag.sum()
    return weights

def get_cluster_variance(cov_matrix: pd.DataFrame, indices: List[int]) -> float:
    weights = get_inverse_variance_weights(cov_matrix, indices)
    cov_slice = cov_matrix.iloc[indices, indices]
    cluster_var = np.dot(weights, np.dot(cov_slice, weights))
    return cluster_var

def recursive_bisection(cov_matrix: pd.DataFrame, sorted_indices: List[int]) -> pd.Series:
    weights = pd.Series(1.0, index=sorted_indices)
    clusters = [sorted_indices]
    
    while len(clusters) > 0:
        new_clusters = []
        
        for cluster in clusters:
            if len(cluster) == 1:
                continue
            
            split_point = len(cluster) // 2
            left_cluster = cluster[:split_point]
            right_cluster = cluster[split_point:]
            
            left_var = get_cluster_variance(cov_matrix, left_cluster)
            right_var = get_cluster_variance(cov_matrix, right_cluster)
            
            alpha = 1 - left_var / (left_var + right_var)
            
            weights[left_cluster] *= alpha
            weights[right_cluster] *= (1 - alpha)
            
            new_clusters.extend([left_cluster, right_cluster])
        
        clusters = new_clusters
    
    weights_series = pd.Series(weights.values, index=cov_matrix.columns[weights.index])
    weights_series = weights_series / weights_series.sum()
    
    return weights_series

def hrp_algorithm(returns: pd.DataFrame, verbose: bool = False) -> Tuple[pd.Series, np.ndarray, pd.DataFrame]:
    if verbose:
        logger.info("Running HRP algorithm...")
    
    cov_matrix = returns.cov() * 252
    linkage_matrix, corr_matrix = tree_clustering(returns)
    sorted_indices = quasi_diagonalization(linkage_matrix)
    weights = recursive_bisection(cov_matrix, sorted_indices)
    
    if verbose:
        logger.info(f"✓ HRP complete: {len(weights)} weights, sum={weights.sum():.6f}")
    
    return weights, linkage_matrix, corr_matrix

def create_equal_weight_portfolio(returns: pd.DataFrame) -> pd.Series:
    n_assets = len(returns.columns)
    weights = pd.Series(1.0 / n_assets, index=returns.columns)
    return weights

def create_inverse_vol_portfolio(returns: pd.DataFrame) -> pd.Series:
    vols = returns.std() * np.sqrt(252)
    inv_vols = 1 / vols
    weights = inv_vols / inv_vols.sum()
    return weights

def create_min_variance_portfolio(returns: pd.DataFrame) -> pd.Series:
    cov_matrix = returns.cov() * 252
    
    try:
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(len(cov_matrix))
        
        weights = inv_cov @ ones
        weights = weights / weights.sum()
        
        weights = pd.Series(weights, index=returns.columns)
        weights = weights.clip(lower=0)
        weights = weights / weights.sum()
        
        return weights
        
    except np.linalg.LinAlgError:
        logger.warning("Covariance matrix singular, falling back to equal weights")
        return create_equal_weight_portfolio(returns)

def create_risk_parity_portfolio(returns: pd.DataFrame, max_iter: int = 100) -> pd.Series:
    cov_matrix = returns.cov() * 252
    n_assets = len(returns.columns)
    
    weights = np.ones(n_assets) / n_assets
    
    for _ in range(max_iter):
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        marginal_contrib = cov_matrix @ weights
        risk_contrib = weights * marginal_contrib / portfolio_vol
        
        target_risk = portfolio_vol / n_assets
        
        weights = weights * target_risk / risk_contrib
        weights = weights / weights.sum()
    
    return pd.Series(weights, index=returns.columns)

def walk_forward_analysis(
    returns: pd.DataFrame,
    train_window: int = 252,
    test_window: int = 63,
    strategy_func=hrp_algorithm,
    verbose: bool = True
) -> Dict:
    if verbose:
        logger.info("\n" + "="*70)
        logger.info("WALK-FORWARD OUT-OF-SAMPLE ANALYSIS")
        logger.info("="*70)
    
    results = []
    test_returns_list = []
    test_dates = []
    
    n_tests = (len(returns) - train_window) // test_window
    
    for i in range(n_tests):
        start_train = i * test_window
        end_train = start_train + train_window
        end_test = end_train + test_window
        
        if end_test > len(returns):
            break
        
        train_data = returns.iloc[start_train:end_train]
        test_data = returns.iloc[end_train:end_test]
        
        if strategy_func == hrp_algorithm:
            weights, _, _ = strategy_func(train_data, verbose=False)
        else:
            weights = strategy_func(train_data)
        
        test_port_returns = calculate_portfolio_returns(test_data, weights)
        
        test_sharpe = (test_port_returns.mean() / test_port_returns.std() * np.sqrt(252)) if test_port_returns.std() > 0 else 0
        test_return = test_port_returns.sum()
        
        results.append({
            'test_period': i + 1,
            'train_start': train_data.index[0],
            'train_end': train_data.index[-1],
            'test_start': test_data.index[0],
            'test_end': test_data.index[-1],
            'test_return': test_return,
            'test_sharpe': test_sharpe,
            'test_vol': test_port_returns.std() * np.sqrt(252),
            'weights': weights
        })
        
        test_returns_list.append(test_port_returns)
        test_dates.append(test_data.index)
        
        if verbose:
            logger.info(f"Period {i+1}/{n_tests}: Test Return={test_return:.2%}, Sharpe={test_sharpe:.3f}")
    
    full_oos_returns = pd.concat(test_returns_list)
    
    if verbose:
        logger.info(f"\n✓ Walk-forward complete: {len(results)} test periods")
        logger.info(f"✓ Total OOS days: {len(full_oos_returns)}")
    
    return {
        'period_results': results,
        'oos_returns': full_oos_returns,
        'test_dates': test_dates
    }

def calculate_portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    common_assets = list(set(weights.index) & set(returns.columns))
    weights_aligned = weights[common_assets]
    weights_aligned = weights_aligned / weights_aligned.sum()
    returns_aligned = returns[common_assets]
    portfolio_returns = (returns_aligned * weights_aligned).sum(axis=1)
    return portfolio_returns

def calculate_portfolio_returns_with_costs(
    returns: pd.DataFrame,
    initial_weights: pd.Series,
    strategy_func,
    cost_bps: float = 10.0,
    rebalance_freq: int = 21,
    verbose: bool = False
) -> Tuple[pd.Series, float, List[float]]:
    current_weights = initial_weights.copy()
    net_returns = []
    turnover_history = []
    total_cost = 0
    
    for i in range(len(returns)):
        daily_ret = calculate_portfolio_returns(returns.iloc[i:i+1], current_weights).iloc[0]
        
        if i > 0 and i % rebalance_freq == 0:
            if strategy_func == hrp_algorithm:
                new_weights, _, _ = strategy_func(returns.iloc[:i], verbose=False)
            else:
                new_weights = strategy_func(returns.iloc[:i])
            
            common_assets = list(set(current_weights.index) & set(new_weights.index))
            current_aligned = current_weights[common_assets] / current_weights[common_assets].sum()
            new_aligned = new_weights[common_assets] / new_weights[common_assets].sum()
            
            turnover = np.sum(np.abs(new_aligned - current_aligned))
            turnover_history.append(turnover)
            
            cost = turnover * cost_bps / 10000
            daily_ret -= cost
            total_cost += cost
            
            current_weights = new_weights
            
            if verbose and i % (rebalance_freq * 3) == 0:
                logger.info(f"Day {i}: Turnover={turnover:.2%}, Cost={cost*10000:.1f}bps")
        
        net_returns.append(daily_ret)
    
    net_returns = pd.Series(net_returns, index=returns.index)
    avg_turnover = np.mean(turnover_history) if turnover_history else 0
    
    if verbose:
        logger.info(f"Total transaction costs: {total_cost:.4%}")
        logger.info(f"Average turnover: {avg_turnover:.2%}")
    
    return net_returns, avg_turnover, turnover_history

def bootstrap_sharpe_test(
    returns1: pd.Series,
    returns2: pd.Series,
    n_samples: int = 1000,
    confidence: float = 0.95
) -> Dict:
    logger.info("\nBootstrap Sharpe Ratio Test...")
    
    differences = []
    
    for _ in range(n_samples):
        sample_idx = np.random.choice(len(returns1), len(returns1), replace=True)
        
        sample1 = returns1.iloc[sample_idx]
        sample2 = returns2.iloc[sample_idx]
        
        sharpe1 = sample1.mean() / sample1.std() * np.sqrt(252) if sample1.std() > 0 else 0
        sharpe2 = sample2.mean() / sample2.std() * np.sqrt(252) if sample2.std() > 0 else 0
        
        differences.append(sharpe1 - sharpe2)
    
    differences = np.array(differences)
    
    p_value = np.mean(differences <= 0)
    
    alpha = 1 - confidence
    ci_lower = np.percentile(differences, alpha / 2 * 100)
    ci_upper = np.percentile(differences, (1 - alpha / 2) * 100)
    
    actual_sharpe1 = returns1.mean() / returns1.std() * np.sqrt(252)
    actual_sharpe2 = returns2.mean() / returns2.std() * np.sqrt(252)
    actual_diff = actual_sharpe1 - actual_sharpe2
    
    result = {
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'actual_difference': actual_diff,
        'mean_bootstrap_diff': np.mean(differences),
        'std_bootstrap_diff': np.std(differences),
        'significant': p_value < (1 - confidence)
    }
    
    logger.info(f"✓ P-value: {p_value:.4f}")
    logger.info(f"✓ {int(confidence*100)}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    logger.info(f"✓ Significant: {'YES' if result['significant'] else 'NO'}")
    
    return result

def paired_t_test(returns1: pd.Series, returns2: pd.Series) -> Dict:
    diff = returns1 - returns2
    t_stat, p_value = stats.ttest_1samp(diff, 0)
    
    result = {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_diff': diff.mean(),
        'significant': p_value < 0.05
    }
    
    logger.info(f"\nPaired t-test:")
    logger.info(f"✓ t-statistic: {t_stat:.4f}")
    logger.info(f"✓ p-value: {p_value:.4f}")
    
    return result

def analyze_weight_stability(
    returns: pd.DataFrame,
    strategy_func,
    rebalance_months: int = 3,
    verbose: bool = True
) -> Dict:
    if verbose:
        logger.info("\n" + "="*70)
        logger.info("WEIGHT STABILITY ANALYSIS")
        logger.info("="*70)
    
    rebalance_days = rebalance_months * 21
    weight_history = []
    dates = []
    turnover_list = []
    
    for i in range(252, len(returns), rebalance_days):
        subset = returns.iloc[max(0, i-252):i]
        
        if len(subset) < 252:
            continue
        
        if strategy_func == hrp_algorithm:
            weights, _, _ = strategy_func(subset, verbose=False)
        else:
            weights = strategy_func(subset)
        
        weight_history.append(weights)
        dates.append(returns.index[i])
        
        if len(weight_history) > 1:
            common = list(set(weight_history[-1].index) & set(weight_history[-2].index))
            w1 = weight_history[-2][common] / weight_history[-2][common].sum()
            w2 = weight_history[-1][common] / weight_history[-1][common].sum()
            turnover = np.sum(np.abs(w2 - w1))
            turnover_list.append(turnover)
    
    avg_turnover = np.mean(turnover_list) if turnover_list else 0
    max_turnover = np.max(turnover_list) if turnover_list else 0
    min_turnover = np.min(turnover_list) if turnover_list else 0
    
    if verbose:
        logger.info(f"✓ Rebalancing periods: {len(weight_history)}")
        logger.info(f"✓ Average turnover: {avg_turnover:.2%}")
        logger.info(f"✓ Max turnover: {max_turnover:.2%}")
        logger.info(f"✓ Min turnover: {min_turnover:.2%}")
    
    return {
        'weight_history': weight_history,
        'dates': dates,
        'turnover_history': turnover_list,
        'avg_turnover': avg_turnover,
        'max_turnover': max_turnover,
        'min_turnover': min_turnover
    }

def evaluate_portfolio_comprehensive(
    returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_name: str = "Portfolio",
    benchmark_returns: Optional[pd.Series] = None
) -> Dict:
    portfolio_returns = calculate_portfolio_returns(returns, weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    total_return = cumulative_returns.iloc[-1] - 1
    n_years = len(portfolio_returns) / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
    
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino_ratio = annual_return / downside_vol if downside_vol > 0 else 0
    
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    skewness = portfolio_returns.skew()
    kurtosis = portfolio_returns.kurtosis()
    
    win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns)
    
    var_95 = portfolio_returns.quantile(0.05)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    
    metrics = {
        'Annual Return': annual_return,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar_ratio,
        'Total Return': total_return,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Win Rate': win_rate,
        'VaR (95%)': var_95,
        'CVaR (95%)': cvar_95
    }
    
    if benchmark_returns is not None:
        beta = np.cov(portfolio_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        alpha = annual_return - beta * (benchmark_returns.mean() * 252)
        
        tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
        info_ratio = (annual_return - benchmark_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        metrics['Beta'] = beta
        metrics['Alpha'] = alpha
        metrics['Tracking Error'] = tracking_error
        metrics['Information Ratio'] = info_ratio
    
    return {
        'metrics': metrics,
        'returns': portfolio_returns,
        'cumulative': cumulative_returns,
        'drawdown': drawdown
    }

def run_complete_comparison(
    returns: pd.DataFrame,
    include_costs: bool = True,
    walk_forward: bool = True,
    statistical_tests: bool = True,
    verbose: bool = True
) -> Dict:
    if verbose:
        logger.info("\n" + "="*70)
        logger.info("COMPREHENSIVE STRATEGY COMPARISON")
        logger.info("="*70)
    
    strategies = {
        'HRP': hrp_algorithm,
        '1/N': create_equal_weight_portfolio,
        'Inverse Vol': create_inverse_vol_portfolio,
        'Min Variance': create_min_variance_portfolio,
        'Risk Parity': create_risk_parity_portfolio
    }
    
    results = {}
    
    if verbose:
        logger.info("\n1. IN-SAMPLE EVALUATION")
    
    for name, func in strategies.items():
        if verbose:
            logger.info(f"\nEvaluating {name}...")
        
        if func == hrp_algorithm:
            weights, _, _ = func(returns, verbose=False)
        else:
            weights = func(returns)
        
        results[name] = evaluate_portfolio_comprehensive(returns, weights, name)
    
    if walk_forward:
        if verbose:
            logger.info("\n2. WALK-FORWARD OUT-OF-SAMPLE ANALYSIS")
        
        oos_results = {}
        for name, func in strategies.items():
            if verbose:
                logger.info(f"\n{name} walk-forward...")
            oos_results[name] = walk_forward_analysis(
                returns, 
                train_window=config.train_window,
                test_window=config.test_window,
                strategy_func=func,
                verbose=False
            )
        
        results['oos'] = oos_results
    
    if include_costs:
        if verbose:
            logger.info("\n3. TRANSACTION COST ANALYSIS")
        
        cost_results = {}
        for name, func in strategies.items():
            if name == '1/N':
                continue
            
            if verbose:
                logger.info(f"\n{name} with costs...")
            
            if func == hrp_algorithm:
                initial_weights, _, _ = func(returns.iloc[:252], verbose=False)
            else:
                initial_weights = func(returns.iloc[:252])
            
            net_returns, avg_turnover, turnover_hist = calculate_portfolio_returns_with_costs(
                returns,
                initial_weights,
                func,
                cost_bps=config.cost_bps,
                rebalance_freq=config.rebalance_freq,
                verbose=False
            )
            
            net_sharpe = (net_returns.mean() / net_returns.std() * np.sqrt(252)) if net_returns.std() > 0 else 0
            gross_sharpe = results[name]['metrics']['Sharpe Ratio']
            sharpe_impact = (net_sharpe - gross_sharpe) / gross_sharpe if gross_sharpe != 0 else 0
            
            cost_results[name] = {
                'net_returns': net_returns,
                'net_sharpe': net_sharpe,
                'gross_sharpe': gross_sharpe,
                'sharpe_impact': sharpe_impact,
                'avg_turnover': avg_turnover,
                'turnover_history': turnover_hist
            }
        
        results['costs'] = cost_results
    
    if statistical_tests:
        if verbose:
            logger.info("\n4. STATISTICAL SIGNIFICANCE TESTS")
        
        hrp_returns = results['HRP']['returns']
        
        stat_results = {}
        for name in ['1/N', 'Inverse Vol', 'Min Variance']:
            if verbose:
                logger.info(f"\nHRP vs {name}:")
            
            baseline_returns = results[name]['returns']
            
            bootstrap = bootstrap_sharpe_test(
                hrp_returns,
                baseline_returns,
                n_samples=config.bootstrap_samples,
                confidence=config.confidence_level
            )
            
            ttest = paired_t_test(hrp_returns, baseline_returns)
            
            stat_results[name] = {
                'bootstrap': bootstrap,
                'ttest': ttest
            }
        
        results['statistical_tests'] = stat_results
    
    if verbose:
        logger.info("\n5. WEIGHT STABILITY ANALYSIS")
    
    stability_results = {}
    for name in ['HRP', 'Min Variance', 'Risk Parity']:
        func = strategies[name]
        stability = analyze_weight_stability(
            returns,
            func,
            rebalance_months=3,
            verbose=False
        )
        stability_results[name] = stability
        
        if verbose:
            logger.info(f"{name}: Avg turnover = {stability['avg_turnover']:.2%}")
    
    results['stability'] = stability_results
    
    if verbose:
        logger.info("\n" + "="*70)
        logger.info("✓ COMPREHENSIVE COMPARISON COMPLETE")
        logger.info("="*70)
    
    return results

def plot_comprehensive_comparison(results: Dict, save_dir: str = 'outputs'):
    output_path = Path(save_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    fig1, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    for name in ['HRP', '1/N', 'Inverse Vol', 'Min Variance']:
        results[name]['cumulative'].plot(ax=axes[0, 0], label=name, linewidth=2)
    axes[0, 0].set_title('Cumulative Returns', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Cumulative Return')
    axes[0, 0].legend(loc='best')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    for name in ['HRP', '1/N', 'Min Variance']:
        (results[name]['drawdown'] * 100).plot(ax=axes[0, 1], label=name, linewidth=2)
    axes[0, 1].set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Drawdown (%)')
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(alpha=0.3)
    
    metrics = ['Annual Return', 'Sharpe Ratio', 'Sortino Ratio']
    strategies = ['HRP', '1/N', 'Inverse Vol', 'Min Variance']
    data = [[results[s]['metrics'][m] for m in metrics] for s in strategies]
    
    x = np.arange(len(metrics))
    width = 0.2
    colors = ['steelblue', 'coral', 'lightgreen', 'plum']
    
    for i, (strategy, color) in enumerate(zip(strategies, colors)):
        axes[1, 0].bar(x + i*width, data[i], width, label=strategy, color=color)
    
    axes[1, 0].set_title('Performance Metrics', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_xticks(x + width * 1.5)
    axes[1, 0].set_xticklabels(metrics)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    for name, color in zip(strategies, colors):
        ret = results[name]['metrics']['Annual Return']
        vol = results[name]['metrics']['Annual Volatility']
        axes[1, 1].scatter(vol, ret, s=200, label=name, color=color, 
                          edgecolors='black', linewidths=2)
    
    axes[1, 1].set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Annual Volatility')
    axes[1, 1].set_ylabel('Annual Return')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path / 'comprehensive_comparison.png'}")
    plt.close()
    
    if 'oos' in results:
        fig2, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        for name in ['HRP', '1/N', 'Min Variance']:
            oos_cum = (1 + results['oos'][name]['oos_returns']).cumprod()
            oos_cum.plot(ax=axes[0, 0], label=name, linewidth=2)
        
        axes[0, 0].set_title('Out-of-Sample Cumulative Returns', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        strategies_oos = ['HRP', '1/N', 'Min Variance']
        for name in strategies_oos:
            periods = [r['test_period'] for r in results['oos'][name]['period_results']]
            sharpes = [r['test_sharpe'] for r in results['oos'][name]['period_results']]
            axes[0, 1].plot(periods, sharpes, marker='o', label=name, linewidth=2)
        
        axes[0, 1].set_title('Out-of-Sample Sharpe by Period', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Test Period')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        if 'costs' in results:
            cost_strategies = list(results['costs'].keys())
            gross_sharpes = [results[s]['metrics']['Sharpe Ratio'] for s in cost_strategies]
            net_sharpes = [results['costs'][s]['net_sharpe'] for s in cost_strategies]
            
            x = np.arange(len(cost_strategies))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, gross_sharpes, width, label='Gross', color='steelblue')
            axes[1, 0].bar(x + width/2, net_sharpes, width, label='Net (after costs)', color='coral')
            
            axes[1, 0].set_title('Transaction Cost Impact', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylabel('Sharpe Ratio')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(cost_strategies, rotation=45)
            axes[1, 0].legend()
            axes[1, 0].grid(axis='y', alpha=0.3)
        
        if 'stability' in results:
            stab_strategies = list(results['stability'].keys())
            avg_turnovers = [results['stability'][s]['avg_turnover'] * 100 for s in stab_strategies]
            
            axes[1, 1].bar(stab_strategies, avg_turnovers, color='lightgreen', edgecolor='black')
            axes[1, 1].set_title('Average Portfolio Turnover', fontsize=14, fontweight='bold')
            axes[1, 1].set_ylabel('Turnover (%)')
            axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'oos_analysis.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: {output_path / 'oos_analysis.png'}")
        plt.close()
    
    if 'statistical_tests' in results:
        fig3, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        baselines = list(results['statistical_tests'].keys())
        p_values = [results['statistical_tests'][b]['bootstrap']['p_value'] for b in baselines]
        
        colors_sig = ['green' if p < 0.05 else 'red' for p in p_values]
        axes[0].bar(baselines, p_values, color=colors_sig, edgecolor='black')
        axes[0].axhline(y=0.05, color='black', linestyle='--', label='α = 0.05')
        axes[0].set_title('Statistical Significance (Bootstrap)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('P-value')
        axes[0].set_xlabel('HRP vs Baseline')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        for i, baseline in enumerate(baselines):
            ci_lower = results['statistical_tests'][baseline]['bootstrap']['ci_lower']
            ci_upper = results['statistical_tests'][baseline]['bootstrap']['ci_upper']
            actual = results['statistical_tests'][baseline]['bootstrap']['actual_difference']
            
            axes[1].errorbar(i, actual, 
                             yerr=[[actual - ci_lower], [ci_upper - actual]],
                             fmt='o', markersize=10, capsize=10, capthick=2,
                             label=baseline)
        
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_title('Sharpe Ratio Difference (95% CI)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('HRP Sharpe - Baseline Sharpe')
        axes[1].set_xticks(range(len(baselines)))
        axes[1].set_xticklabels(baselines)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'statistical_tests.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: {output_path / 'statistical_tests.png'}")
        plt.close()

def plot_dendrogram(linkage_matrix, returns, save_path='outputs/dendrogram.png'):
    output_path = Path(save_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    plt.figure(figsize=(20, 8))
    dendrogram(
        linkage_matrix,
        labels=returns.columns.tolist(),
        leaf_rotation=90,
        leaf_font_size=8
    )
    plt.title('Hierarchical Clustering Dendrogram', fontsize=16, fontweight='bold')
    plt.xlabel('Assets', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}")
    plt.close()

def save_comprehensive_results(results: Dict, hrp_weights: pd.Series, output_dir: str = 'outputs'):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    logger.info("\n" + "="*70)
    logger.info("SAVING RESULTS")
    logger.info("="*70)
    
    hrp_weights.sort_values(ascending=False).to_csv(
        output_path / 'hrp_weights.csv',
        header=['Weight']
    )
    logger.info(f"✓ HRP weights: {output_path / 'hrp_weights.csv'}")
    
    strategies = ['HRP', '1/N', 'Inverse Vol', 'Min Variance', 'Risk Parity']
    metrics_df = pd.DataFrame({
        s: results[s]['metrics'] for s in strategies if s in results
    }).T
    
    metrics_df.to_csv(output_path / 'in_sample_metrics.csv')
    logger.info(f"✓ In-sample metrics: {output_path / 'in_sample_metrics.csv'}")
    
    if 'oos' in results:
        oos_summary = {}
        for name in ['HRP', '1/N', 'Min Variance']:
            oos_returns = results['oos'][name]['oos_returns']
            oos_summary[name] = {
                'OOS Sharpe': oos_returns.mean() / oos_returns.std() * np.sqrt(252),
                'OOS Annual Return': oos_returns.mean() * 252,
                'OOS Annual Vol': oos_returns.std() * np.sqrt(252),
                'OOS Total Return': (1 + oos_returns).prod() - 1
            }
        
        pd.DataFrame(oos_summary).T.to_csv(output_path / 'oos_summary.csv')
        logger.info(f"✓ OOS summary: {output_path / 'oos_summary.csv'}")
    
    if 'statistical_tests' in results:
        stat_summary = {}
        for baseline, tests in results['statistical_tests'].items():
            stat_summary[baseline] = {
                'P-value (Bootstrap)': tests['bootstrap']['p_value'],
                'CI Lower': tests['bootstrap']['ci_lower'],
                'CI Upper': tests['bootstrap']['ci_upper'],
                'Actual Difference': tests['bootstrap']['actual_difference'],
                'Significant': tests['bootstrap']['significant'],
                'T-statistic': tests['ttest']['t_statistic'],
                'P-value (t-test)': tests['ttest']['p_value']
            }
        
        pd.DataFrame(stat_summary).T.to_csv(output_path / 'statistical_tests.csv')
        logger.info(f"✓ Statistical tests: {output_path / 'statistical_tests.csv'}")
    
    if 'stability' in results:
        turnover_df = pd.DataFrame({
            name: {'Avg Turnover': results['stability'][name]['avg_turnover'],
                   'Max Turnover': results['stability'][name]['max_turnover'],
                   'Min Turnover': results['stability'][name]['min_turnover']}
            for name in results['stability'].keys()
        }).T
        
        turnover_df.to_csv(output_path / 'turnover_analysis.csv')
        logger.info(f"✓ Turnover analysis: {output_path / 'turnover_analysis.csv'}")
    
    with open(output_path / 'validation_report.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("HIERARCHICAL RISK PARITY - COMPREHENSIVE VALIDATION REPORT\n")
        f.write("="*70 + "\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        
        f.write("PAPER CLAIM:\n")
        f.write("-"*70 + "\n")
        f.write("HRP outperforms naive 1/N diversification out-of-sample\n")
        f.write("with superior risk-adjusted returns and lower turnover.\n\n")
        
        f.write("IN-SAMPLE RESULTS:\n")
        f.write("-"*70 + "\n")
        f.write(metrics_df.to_string())
        f.write("\n\n")
        
        if 'oos' in results:
            f.write("OUT-OF-SAMPLE RESULTS:\n")
            f.write("-"*70 + "\n")
            hrp_oos = results['oos']['HRP']['oos_returns']
            eq_oos = results['oos']['1/N']['oos_returns']
            
            hrp_sharpe_oos = hrp_oos.mean() / hrp_oos.std() * np.sqrt(252)
            eq_sharpe_oos = eq_oos.mean() / eq_oos.std() * np.sqrt(252)
            
            f.write(f"HRP OOS Sharpe: {hrp_sharpe_oos:.4f}\n")
            f.write(f"1/N OOS Sharpe: {eq_sharpe_oos:.4f}\n")
            f.write(f"Improvement: {(hrp_sharpe_oos/eq_sharpe_oos - 1)*100:.2f}%\n\n")
        
        if 'statistical_tests' in results:
            f.write("STATISTICAL SIGNIFICANCE:\n")
            f.write("-"*70 + "\n")
            for baseline, tests in results['statistical_tests'].items():
                f.write(f"\nHRP vs {baseline}:\n")
                f.write(f"  P-value: {tests['bootstrap']['p_value']:.4f}\n")
                f.write(f"  Significant: {'YES' if tests['bootstrap']['significant'] else 'NO'}\n")
                f.write(f"  95% CI: [{tests['bootstrap']['ci_lower']:.4f}, {tests['bootstrap']['ci_upper']:.4f}]\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("VALIDATION CONCLUSION:\n")
        f.write("="*70 + "\n")
        
        if 'oos' in results and 'statistical_tests' in results:
            hrp_oos_sharpe = results['oos']['HRP']['oos_returns'].mean() / results['oos']['HRP']['oos_returns'].std() * np.sqrt(252)
            eq_oos_sharpe = results['oos']['1/N']['oos_returns'].mean() / results['oos']['1/N']['oos_returns'].std() * np.sqrt(252)
            p_val = results['statistical_tests']['1/N']['bootstrap']['p_value']
            
            if hrp_oos_sharpe > eq_oos_sharpe and p_val < 0.05:
                f.write("✓ PAPER CLAIM VALIDATED\n")
                f.write("HRP demonstrates statistically significant out-of-sample\n")
                f.write("outperformance vs naive diversification.\n")
            else:
                f.write("⚠ MIXED RESULTS\n")
                f.write("Results do not fully support paper's claims in this sample.\n")
        
        f.write("\n" + "="*70 + "\n")
    
    logger.info(f"✓ Validation report: {output_path / 'validation_report.txt'}")
    logger.info("\n" + "="*70)
    logger.info("✓ ALL RESULTS SAVED")
    logger.info("="*70)

def main(
    csv_path: str = 'data/financials.csv',
    start_date: str = '2020-01-01',
    end_date: str = '2024-12-31',
    output_dir: str = 'outputs'
):
    start_time = time.time()
    
    logger.info("\n" + "="*70)
    logger.info("ENHANCED HRP IMPLEMENTATION - JANE STREET CHALLENGE")
    logger.info("="*70)
    
    config.start_date = start_date
    config.end_date = end_date
    config.output_dir = Path(output_dir)
    
    try:
        tickers = load_tickers(csv_path)
        prices = fetch_historical_data(tickers, start_date, end_date)
        returns = calculate_returns(prices)
        
        logger.info("\n" + "="*70)
        logger.info("RUNNING HRP ALGORITHM")
        logger.info("="*70)
        hrp_weights, linkage_matrix, corr_matrix = hrp_algorithm(returns, verbose=True)
        
        results = run_complete_comparison(
            returns,
            include_costs=True,
            walk_forward=True,
            statistical_tests=True,
            verbose=True
        )
        
        logger.info("\n" + "="*70)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*70)
        
        plot_dendrogram(linkage_matrix, returns, save_path=f'{output_dir}/dendrogram.png')
        plot_comprehensive_comparison(results, save_dir=output_dir)
        
        save_comprehensive_results(results, hrp_weights, output_dir=output_dir)
        
        elapsed = time.time() - start_time
        logger.info("\n" + "="*70)
        logger.info("✓ PIPELINE COMPLETE")
        logger.info("="*70)
        logger.info(f"Execution time: {elapsed:.2f} seconds")
        logger.info(f"\nFiles saved in '{output_dir}/' directory:")
        logger.info("  - hrp_weights.csv")
        logger.info("  - in_sample_metrics.csv")
        logger.info("  - oos_summary.csv")
        logger.info("  - statistical_tests.csv")
        logger.info("  - turnover_analysis.csv")
        logger.info("  - validation_report.txt")
        logger.info("  - comprehensive_comparison.png")
        logger.info("  - oos_analysis.png")
        logger.info("  - statistical_tests.png")
        logger.info("  - dendrogram.png")
        
        return hrp_weights, results
        
    except Exception as e:
        logger.error(f"\n Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    hrp_weights, results = main()
