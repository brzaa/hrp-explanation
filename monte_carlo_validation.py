"""
Comprehensive Monte Carlo Simulation Framework for HRP Validation

This module implements systematic simulations to identify the specific conditions
under which Hierarchical Risk Parity outperforms naive diversification.

Following López de Prado's validation philosophy:
"Always look for simulation-based validations of a theory, and question the
soundness of the assumptions in the simulation"

Author: HRP Validation Study
Date: 2025-11-18
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from hrp import (
    hrp_algorithm,
    create_equal_weight_portfolio,
    create_inverse_vol_portfolio,
    create_min_variance_portfolio,
    create_risk_parity_portfolio
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulations"""
    n_simulations: int = 10000
    n_assets: int = 50
    n_observations: int = 260  # 1 year of daily returns
    random_seed: int = 42


@dataclass
class SimulationResult:
    """Results from a single simulation run"""
    hrp_var: float
    equal_weight_var: float
    inverse_vol_var: float
    min_var_var: float
    risk_parity_var: float
    correlation_structure: str
    avg_correlation: float
    condition_number: float


class CovarianceGenerator:
    """Generate covariance matrices with specific structures"""

    @staticmethod
    def random_correlation(n_assets: int, method='uniform') -> np.ndarray:
        """Generate a random valid correlation matrix"""
        if method == 'uniform':
            # Marsaglia & Olkin (1984) method
            A = np.random.randn(n_assets, n_assets)
            A = A @ A.T  # Guarantee positive semi-definite
            D = np.sqrt(np.diag(A))
            corr = A / np.outer(D, D)
            return corr
        elif method == 'vine':
            # Vine method (Lewandowski, Kurowicka, Joe 2009)
            P = np.eye(n_assets)
            S = np.eye(n_assets)

            for k in range(n_assets - 1):
                for i in range(k + 1, n_assets):
                    P[k, i] = np.random.uniform(-1, 1)
                    p = P[k, i]
                    for l in range(k - 1, -1, -1):
                        p = p * np.sqrt((1 - P[l, i]**2) * (1 - P[l, k]**2)) + P[l, i] * P[l, k]
                    S[k, i] = p
                    S[i, k] = p

            return S

    @staticmethod
    def block_diagonal(n_blocks: int = 5, block_size: int = 10,
                       within_corr: float = 0.7, between_corr: float = 0.1) -> np.ndarray:
        """
        Generate block-diagonal correlation structure
        This mimics assets from different sectors (high intra-sector, low inter-sector correlation)
        """
        n_assets = n_blocks * block_size
        corr = np.ones((n_assets, n_assets)) * between_corr

        for i in range(n_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            corr[start:end, start:end] = within_corr

        np.fill_diagonal(corr, 1.0)

        # Ensure positive definite
        min_eig = np.min(np.linalg.eigvals(corr))
        if min_eig < 0:
            corr += (-min_eig + 0.01) * np.eye(n_assets)
            # Re-normalize
            D = np.sqrt(np.diag(corr))
            corr = corr / np.outer(D, D)

        return corr

    @staticmethod
    def single_factor(n_assets: int, factor_loading_range: Tuple[float, float] = (0.3, 0.8)) -> np.ndarray:
        """
        Generate correlation from a single-factor model
        r_i = β_i * f + ε_i
        where f is the common factor
        """
        betas = np.random.uniform(factor_loading_range[0], factor_loading_range[1], n_assets)

        # Cov(r_i, r_j) = β_i * β_j * Var(f) + 0 (assuming independent epsilons)
        # Assuming Var(f) = 1 and Var(ε_i) such that Var(r_i) = 1
        corr = np.outer(betas, betas)

        # Set idiosyncratic variance such that total variance = 1
        for i in range(n_assets):
            corr[i, i] = 1.0

        return corr

    @staticmethod
    def multi_factor(n_assets: int, n_factors: int = 3) -> np.ndarray:
        """
        Generate correlation from a multi-factor model
        """
        # Random factor loadings
        B = np.random.uniform(0.2, 0.8, (n_assets, n_factors))

        # Factor covariance (assume factors are uncorrelated for simplicity)
        F = np.eye(n_factors)

        # Covariance matrix
        cov = B @ F @ B.T

        # Add idiosyncratic variance to make diagonal = 1
        for i in range(n_assets):
            cov[i, i] = 1.0

        # Convert to correlation
        D = np.sqrt(np.diag(cov))
        corr = cov / np.outer(D, D)

        return corr

    @staticmethod
    def high_correlation(n_assets: int, avg_corr: float = 0.8) -> np.ndarray:
        """Generate uniformly high correlation (like single-sector portfolios)"""
        corr = np.ones((n_assets, n_assets)) * avg_corr
        np.fill_diagonal(corr, 1.0)

        # Add small random noise to make it positive definite
        noise = np.random.randn(n_assets, n_assets) * 0.01
        noise = (noise + noise.T) / 2
        corr = corr + noise

        # Ensure positive definite
        min_eig = np.min(np.linalg.eigvals(corr))
        if min_eig < 0:
            corr += (-min_eig + 0.01) * np.eye(n_assets)

        # Re-normalize
        D = np.sqrt(np.diag(corr))
        corr = corr / np.outer(D, D)

        return corr

    @staticmethod
    def low_correlation(n_assets: int, avg_corr: float = 0.2) -> np.ndarray:
        """Generate uniformly low correlation (diversified portfolio)"""
        corr = np.ones((n_assets, n_assets)) * avg_corr
        np.fill_diagonal(corr, 1.0)

        # Add small random noise
        noise = np.random.randn(n_assets, n_assets) * 0.05
        noise = (noise + noise.T) / 2
        corr = corr + noise

        # Ensure valid correlation matrix
        corr = np.clip(corr, -0.99, 0.99)
        np.fill_diagonal(corr, 1.0)

        # Ensure positive definite
        min_eig = np.min(np.linalg.eigvals(corr))
        if min_eig < 0:
            corr += (-min_eig + 0.01) * np.eye(n_assets)

        # Re-normalize
        D = np.sqrt(np.diag(corr))
        corr = corr / np.outer(D, D)

        return corr

    @staticmethod
    def market_crash(n_assets: int) -> np.ndarray:
        """
        Simulate market crash: all correlations go to 1
        (Markowitz's curse: when you need diversification most, it works least)
        """
        corr = np.ones((n_assets, n_assets)) * 0.95
        np.fill_diagonal(corr, 1.0)

        # Small noise for numerical stability
        noise = np.random.randn(n_assets, n_assets) * 0.001
        noise = (noise + noise.T) / 2
        corr = corr + noise

        # Ensure positive definite
        min_eig = np.min(np.linalg.eigvals(corr))
        if min_eig < 0:
            corr += (-min_eig + 0.01) * np.eye(n_assets)

        D = np.sqrt(np.diag(corr))
        corr = corr / np.outer(D, D)

        return corr

    @staticmethod
    def equal_volatility(n_assets: int, vol: float = 0.2) -> np.ndarray:
        """Equal volatility for all assets (isolate correlation effect)"""
        return np.ones(n_assets) * vol

    @staticmethod
    def heterogeneous_volatility(n_assets: int) -> np.ndarray:
        """Heterogeneous volatility (realistic)"""
        return np.random.uniform(0.1, 0.5, n_assets)


def generate_returns(corr: np.ndarray, vols: np.ndarray, n_obs: int,
                     mean_return: float = 0.0) -> pd.DataFrame:
    """
    Generate synthetic returns given correlation and volatility structure

    Args:
        corr: Correlation matrix (N x N)
        vols: Volatility vector (N,)
        n_obs: Number of observations
        mean_return: Mean daily return (default 0 for simplicity)

    Returns:
        DataFrame of returns (n_obs x N)
    """
    n_assets = len(vols)

    # Convert to covariance matrix
    D = np.diag(vols)
    cov = D @ corr @ D

    # Generate returns from multivariate normal
    mean_vec = np.ones(n_assets) * mean_return
    returns = np.random.multivariate_normal(mean_vec, cov, n_obs)

    return pd.DataFrame(returns, columns=[f'Asset_{i}' for i in range(n_assets)])


def run_single_simulation(corr: np.ndarray, vols: np.ndarray,
                         config: SimulationConfig,
                         structure_name: str) -> SimulationResult:
    """
    Run a single simulation: generate data, fit strategies, evaluate OOS variance

    This mimics López de Prado's methodology:
    1. Generate true covariance matrix
    2. Sample data (with estimation error)
    3. Estimate covariance from sample
    4. Construct portfolios using estimated covariance
    5. Evaluate OOS using TRUE covariance
    """
    n_assets = len(vols)

    # True covariance matrix
    D = np.diag(vols)
    true_cov = D @ corr @ D

    # Generate sample returns (with noise)
    sample_returns = generate_returns(corr, vols, config.n_observations)

    # Construct portfolios
    try:
        hrp_weights, _, _ = hrp_algorithm(sample_returns, verbose=False)
    except:
        hrp_weights = create_equal_weight_portfolio(sample_returns)

    eq_weights = create_equal_weight_portfolio(sample_returns)
    iv_weights = create_inverse_vol_portfolio(sample_returns)

    try:
        mv_weights = create_min_variance_portfolio(sample_returns)
    except:
        mv_weights = eq_weights  # Fallback if singular

    try:
        rp_weights = create_risk_parity_portfolio(sample_returns)
    except:
        rp_weights = iv_weights  # Fallback

    # Evaluate OOS variance using TRUE covariance
    def oos_variance(weights, true_cov):
        w = weights.values
        return w @ true_cov @ w

    hrp_var = oos_variance(hrp_weights, true_cov)
    eq_var = oos_variance(eq_weights, true_cov)
    iv_var = oos_variance(iv_weights, true_cov)
    mv_var = oos_variance(mv_weights, true_cov)
    rp_var = oos_variance(rp_weights, true_cov)

    # Compute correlation stats
    avg_corr = (corr.sum() - n_assets) / (n_assets * (n_assets - 1))

    # Compute condition number
    eigenvalues = np.linalg.eigvals(true_cov)
    cond_num = np.max(eigenvalues) / np.min(eigenvalues)

    return SimulationResult(
        hrp_var=hrp_var,
        equal_weight_var=eq_var,
        inverse_vol_var=iv_var,
        min_var_var=mv_var,
        risk_parity_var=rp_var,
        correlation_structure=structure_name,
        avg_correlation=avg_corr,
        condition_number=cond_num
    )


def run_scenario(scenario_name: str,
                 corr_func: Callable,
                 vol_func: Callable,
                 config: SimulationConfig) -> List[SimulationResult]:
    """
    Run Monte Carlo simulation for a specific scenario

    Args:
        scenario_name: Descriptive name
        corr_func: Function that generates correlation matrix
        vol_func: Function that generates volatility vector
        config: Simulation configuration

    Returns:
        List of simulation results
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Running scenario: {scenario_name}")
    logger.info(f"{'='*70}")

    results = []

    np.random.seed(config.random_seed)

    for i in range(config.n_simulations):
        if i % 1000 == 0:
            logger.info(f"  Simulation {i}/{config.n_simulations}")

        # Generate correlation and volatility
        corr = corr_func(config.n_assets)
        vols = vol_func(config.n_assets)

        # Run simulation
        result = run_single_simulation(corr, vols, config, scenario_name)
        results.append(result)

    logger.info(f"✓ Completed {config.n_simulations} simulations for {scenario_name}")

    return results


def analyze_results(results: List[SimulationResult]) -> Dict:
    """Analyze simulation results and compute statistics"""

    hrp_vars = np.array([r.hrp_var for r in results])
    eq_vars = np.array([r.equal_weight_var for r in results])
    iv_vars = np.array([r.inverse_vol_var for r in results])
    mv_vars = np.array([r.min_var_var for r in results])
    rp_vars = np.array([r.risk_parity_var for r in results])

    avg_corrs = np.array([r.avg_correlation for r in results])
    cond_nums = np.array([r.condition_number for r in results])

    # Success rates (HRP beats baseline)
    hrp_beats_eq = (hrp_vars < eq_vars).mean()
    hrp_beats_iv = (hrp_vars < iv_vars).mean()
    hrp_beats_mv = (hrp_vars < mv_vars).mean()

    # Performance improvements
    hrp_vs_eq_improvement = ((eq_vars - hrp_vars) / eq_vars * 100).mean()
    hrp_vs_iv_improvement = ((iv_vars - hrp_vars) / iv_vars * 100).mean()

    # Statistical tests
    t_stat_eq, p_value_eq = stats.ttest_rel(eq_vars, hrp_vars)
    t_stat_iv, p_value_iv = stats.ttest_rel(iv_vars, hrp_vars)

    analysis = {
        'mean_vars': {
            'HRP': np.mean(hrp_vars),
            '1/N': np.mean(eq_vars),
            'Inverse Vol': np.mean(iv_vars),
            'Min Variance': np.mean(mv_vars),
            'Risk Parity': np.mean(rp_vars)
        },
        'median_vars': {
            'HRP': np.median(hrp_vars),
            '1/N': np.median(eq_vars),
            'Inverse Vol': np.median(iv_vars),
            'Min Variance': np.median(mv_vars),
            'Risk Parity': np.median(rp_vars)
        },
        'success_rates': {
            'HRP_vs_1/N': hrp_beats_eq,
            'HRP_vs_Inverse_Vol': hrp_beats_iv,
            'HRP_vs_Min_Var': hrp_beats_mv
        },
        'improvements': {
            'HRP_vs_1/N_pct': hrp_vs_eq_improvement,
            'HRP_vs_Inverse_Vol_pct': hrp_vs_iv_improvement
        },
        'statistical_tests': {
            '1/N': {'t_stat': t_stat_eq, 'p_value': p_value_eq},
            'Inverse_Vol': {'t_stat': t_stat_iv, 'p_value': p_value_iv}
        },
        'correlation_stats': {
            'mean': np.mean(avg_corrs),
            'median': np.median(avg_corrs),
            'std': np.std(avg_corrs)
        },
        'condition_number_stats': {
            'mean': np.mean(cond_nums),
            'median': np.median(cond_nums),
            'std': np.std(cond_nums)
        }
    }

    return analysis


def print_analysis(scenario_name: str, analysis: Dict):
    """Pretty-print analysis results"""
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*70}")

    print(f"\nMean Out-of-Sample Variance:")
    for method, var in analysis['mean_vars'].items():
        print(f"  {method:15s}: {var:.6f}")

    print(f"\nSuccess Rates (HRP beats baseline):")
    for comparison, rate in analysis['success_rates'].items():
        print(f"  {comparison:25s}: {rate*100:.1f}%")

    print(f"\nPerformance Improvement:")
    for comparison, improvement in analysis['improvements'].items():
        print(f"  {comparison:30s}: {improvement:+.2f}%")

    print(f"\nStatistical Significance:")
    for baseline, test in analysis['statistical_tests'].items():
        significant = "✓ YES" if test['p_value'] < 0.05 else "✗ NO"
        print(f"  HRP vs {baseline:12s}: p-value = {test['p_value']:.4f}  {significant}")

    print(f"\nCorrelation Structure:")
    print(f"  Mean correlation: {analysis['correlation_stats']['mean']:.3f}")
    print(f"  Condition number: {analysis['condition_number_stats']['mean']:.0f}")


def run_comprehensive_study(output_dir: str = 'simulation_results'):
    """
    Run comprehensive Monte Carlo study across multiple scenarios
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    config = SimulationConfig(
        n_simulations=10000,
        n_assets=50,
        n_observations=260,
        random_seed=42
    )

    logger.info("\n" + "="*70)
    logger.info("COMPREHENSIVE MONTE CARLO VALIDATION STUDY")
    logger.info("="*70)
    logger.info(f"Configuration:")
    logger.info(f"  Simulations per scenario: {config.n_simulations}")
    logger.info(f"  Assets: {config.n_assets}")
    logger.info(f"  Observations: {config.n_observations}")
    logger.info("="*70)

    # Define scenarios
    scenarios = {
        '1. Block-Diagonal (5 sectors)': (
            lambda n: CovarianceGenerator.block_diagonal(5, 10, 0.7, 0.1),
            CovarianceGenerator.heterogeneous_volatility
        ),
        '2. Block-Diagonal (10 sectors)': (
            lambda n: CovarianceGenerator.block_diagonal(10, 5, 0.7, 0.1),
            CovarianceGenerator.heterogeneous_volatility
        ),
        '3. Single-Factor Model': (
            CovarianceGenerator.single_factor,
            CovarianceGenerator.heterogeneous_volatility
        ),
        '4. Multi-Factor (3 factors)': (
            lambda n: CovarianceGenerator.multi_factor(n, 3),
            CovarianceGenerator.heterogeneous_volatility
        ),
        '5. High Correlation (0.8)': (
            lambda n: CovarianceGenerator.high_correlation(n, 0.8),
            CovarianceGenerator.heterogeneous_volatility
        ),
        '6. Low Correlation (0.2)': (
            lambda n: CovarianceGenerator.low_correlation(n, 0.2),
            CovarianceGenerator.heterogeneous_volatility
        ),
        '7. Market Crash (corr→1)': (
            CovarianceGenerator.market_crash,
            CovarianceGenerator.heterogeneous_volatility
        ),
        '8. Equal Volatility + Block': (
            lambda n: CovarianceGenerator.block_diagonal(5, 10, 0.7, 0.1),
            lambda n: CovarianceGenerator.equal_volatility(n, 0.2)
        ),
        '9. Random Correlation': (
            lambda n: CovarianceGenerator.random_correlation(n, 'uniform'),
            CovarianceGenerator.heterogeneous_volatility
        ),
        '10. Vine Copula': (
            lambda n: CovarianceGenerator.random_correlation(n, 'vine'),
            CovarianceGenerator.heterogeneous_volatility
        )
    }

    all_results = {}
    all_analyses = {}

    start_time = time.time()

    for scenario_name, (corr_func, vol_func) in scenarios.items():
        results = run_scenario(scenario_name, corr_func, vol_func, config)
        analysis = analyze_results(results)

        all_results[scenario_name] = results
        all_analyses[scenario_name] = analysis

        print_analysis(scenario_name, analysis)

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*70}")
    logger.info(f"✓ ALL SCENARIOS COMPLETE")
    logger.info(f"  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"{'='*70}")

    # Save results
    save_results(all_results, all_analyses, output_path)

    # Generate plots
    generate_plots(all_results, all_analyses, output_path)

    # Generate summary report
    generate_summary_report(all_analyses, output_path)

    return all_results, all_analyses


def save_results(all_results: Dict, all_analyses: Dict, output_path: Path):
    """Save results to CSV files"""
    logger.info("\nSaving results...")

    # Summary table
    summary_data = []
    for scenario, analysis in all_analyses.items():
        row = {
            'Scenario': scenario,
            'HRP_Mean_Var': analysis['mean_vars']['HRP'],
            '1/N_Mean_Var': analysis['mean_vars']['1/N'],
            'HRP_Success_Rate': analysis['success_rates']['HRP_vs_1/N'],
            'HRP_Improvement_Pct': analysis['improvements']['HRP_vs_1/N_pct'],
            'P_Value': analysis['statistical_tests']['1/N']['p_value'],
            'Significant': analysis['statistical_tests']['1/N']['p_value'] < 0.05,
            'Avg_Correlation': analysis['correlation_stats']['mean'],
            'Condition_Number': analysis['condition_number_stats']['mean']
        }
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path / 'monte_carlo_summary.csv', index=False)
    logger.info(f"  ✓ Saved: {output_path / 'monte_carlo_summary.csv'}")


def generate_plots(all_results: Dict, all_analyses: Dict, output_path: Path):
    """Generate visualization plots"""
    logger.info("\nGenerating plots...")

    # Plot 1: HRP vs 1/N across scenarios
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    scenarios = list(all_analyses.keys())
    hrp_means = [all_analyses[s]['mean_vars']['HRP'] for s in scenarios]
    eq_means = [all_analyses[s]['mean_vars']['1/N'] for s in scenarios]
    success_rates = [all_analyses[s]['success_rates']['HRP_vs_1/N'] * 100 for s in scenarios]
    improvements = [all_analyses[s]['improvements']['HRP_vs_1/N_pct'] for s in scenarios]

    # Subplot 1: Mean variances
    x = range(len(scenarios))
    width = 0.35
    axes[0, 0].bar([i - width/2 for i in x], hrp_means, width, label='HRP', color='steelblue')
    axes[0, 0].bar([i + width/2 for i in x], eq_means, width, label='1/N', color='coral')
    axes[0, 0].set_xlabel('Scenario')
    axes[0, 0].set_ylabel('Mean OOS Variance')
    axes[0, 0].set_title('Out-of-Sample Variance: HRP vs 1/N')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([s.split('.')[0] for s in scenarios], rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Subplot 2: Success rates
    colors = ['green' if sr > 50 else 'red' for sr in success_rates]
    axes[0, 1].bar(x, success_rates, color=colors, edgecolor='black')
    axes[0, 1].axhline(y=50, color='black', linestyle='--', label='50% baseline')
    axes[0, 1].set_xlabel('Scenario')
    axes[0, 1].set_ylabel('Success Rate (%)')
    axes[0, 1].set_title('HRP Success Rate (% of times HRP beats 1/N)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([s.split('.')[0] for s in scenarios], rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Subplot 3: Improvements
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    axes[1, 0].bar(x, improvements, color=colors, edgecolor='black')
    axes[1, 0].axhline(y=0, color='black', linestyle='--')
    axes[1, 0].set_xlabel('Scenario')
    axes[1, 0].set_ylabel('Improvement (%)')
    axes[1, 0].set_title('HRP Performance Improvement over 1/N')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([s.split('.')[0] for s in scenarios], rotation=45, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Subplot 4: Correlation vs Success Rate
    avg_corrs = [all_analyses[s]['correlation_stats']['mean'] for s in scenarios]
    axes[1, 1].scatter(avg_corrs, success_rates, s=100, c=success_rates,
                       cmap='RdYlGn', edgecolors='black', linewidths=2)
    axes[1, 1].axhline(y=50, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Average Correlation')
    axes[1, 1].set_ylabel('HRP Success Rate (%)')
    axes[1, 1].set_title('Success Rate vs Correlation Structure')
    axes[1, 1].grid(alpha=0.3)

    # Add text labels
    for i, scenario in enumerate(scenarios):
        axes[1, 1].annotate(scenario.split('.')[0], (avg_corrs[i], success_rates[i]),
                           fontsize=8, ha='right')

    plt.tight_layout()
    plt.savefig(output_path / 'monte_carlo_analysis.png', dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ Saved: {output_path / 'monte_carlo_analysis.png'}")
    plt.close()


def generate_summary_report(all_analyses: Dict, output_path: Path):
    """Generate a text summary report"""
    with open(output_path / 'monte_carlo_report.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("COMPREHENSIVE MONTE CARLO VALIDATION STUDY\n")
        f.write("Hierarchical Risk Parity vs Naive Diversification\n")
        f.write("="*70 + "\n\n")

        for scenario, analysis in all_analyses.items():
            f.write(f"{scenario}\n")
            f.write("-"*70 + "\n")
            f.write(f"HRP Mean Variance:     {analysis['mean_vars']['HRP']:.6f}\n")
            f.write(f"1/N Mean Variance:     {analysis['mean_vars']['1/N']:.6f}\n")
            f.write(f"Success Rate:          {analysis['success_rates']['HRP_vs_1/N']*100:.1f}%\n")
            f.write(f"Improvement:           {analysis['improvements']['HRP_vs_1/N_pct']:+.2f}%\n")
            p_val = analysis['statistical_tests']['1/N']['p_value']
            sig = "YES" if p_val < 0.05 else "NO"
            f.write(f"Statistically Sig:     {sig} (p={p_val:.4f})\n")
            f.write(f"Avg Correlation:       {analysis['correlation_stats']['mean']:.3f}\n")
            f.write(f"Condition Number:      {analysis['condition_number_stats']['mean']:.0f}\n")
            f.write("\n")

        f.write("="*70 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("="*70 + "\n\n")

        # Identify best and worst scenarios for HRP
        best_scenario = max(all_analyses.items(),
                          key=lambda x: x[1]['improvements']['HRP_vs_1/N_pct'])
        worst_scenario = min(all_analyses.items(),
                           key=lambda x: x[1]['improvements']['HRP_vs_1/N_pct'])

        f.write(f"Best scenario for HRP:\n")
        f.write(f"  {best_scenario[0]}\n")
        f.write(f"  Improvement: {best_scenario[1]['improvements']['HRP_vs_1/N_pct']:+.2f}%\n\n")

        f.write(f"Worst scenario for HRP:\n")
        f.write(f"  {worst_scenario[0]}\n")
        f.write(f"  Improvement: {worst_scenario[1]['improvements']['HRP_vs_1/N_pct']:+.2f}%\n\n")

    logger.info(f"  ✓ Saved: {output_path / 'monte_carlo_report.txt'}")


if __name__ == '__main__':
    all_results, all_analyses = run_comprehensive_study()
