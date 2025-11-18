"""
Simplified Monte Carlo Study for HRP Validation
Standalone implementation without external dependencies on hrp.py
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_distance_matrix(corr_matrix):
    """Convert correlation to distance metric"""
    return np.sqrt(0.5 * (1 - corr_matrix))


def tree_clustering(corr_matrix):
    """Hierarchical clustering on correlation matrix"""
    distance_matrix = get_distance_matrix(corr_matrix)
    dist_condensed = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(dist_condensed, method='single')
    return linkage_matrix


def quasi_diagonalization(linkage_matrix):
    """Get sorted order from dendrogram"""
    dendro = dendrogram(linkage_matrix, no_plot=True)
    return dendro['leaves']


def get_inverse_variance_weights(cov_matrix, indices):
    """Inverse variance weights for a subset of assets"""
    cov_slice = cov_matrix[np.ix_(indices, indices)]
    inv_diag = 1 / np.diag(cov_slice)
    weights = inv_diag / inv_diag.sum()
    return weights


def get_cluster_variance(cov_matrix, indices):
    """Compute cluster variance using inverse variance weighting"""
    weights = get_inverse_variance_weights(cov_matrix, indices)
    cov_slice = cov_matrix[np.ix_(indices, indices)]
    cluster_var = weights @ cov_slice @ weights
    return cluster_var


def recursive_bisection(cov_matrix, sorted_indices):
    """Recursive bisection with inverse variance allocation"""
    n_assets = len(sorted_indices)
    weights = np.ones(n_assets)
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

    weights = weights / weights.sum()
    return weights


def hrp_weights(cov_matrix, corr_matrix):
    """Complete HRP algorithm"""
    linkage_matrix = tree_clustering(corr_matrix)
    sorted_indices = quasi_diagonalization(linkage_matrix)
    weights = recursive_bisection(cov_matrix, sorted_indices)
    return weights


def generate_block_diagonal_corr(n_blocks=5, block_size=10, within_corr=0.7, between_corr=0.1):
    """Generate block-diagonal correlation structure"""
    n = n_blocks * block_size
    corr = np.ones((n, n)) * between_corr

    for i in range(n_blocks):
        start = i * block_size
        end = (i + 1) * block_size
        corr[start:end, start:end] = within_corr

    np.fill_diagonal(corr, 1.0)

    # Ensure positive definite
    min_eig = np.min(np.linalg.eigvals(corr))
    if min_eig < 0:
        corr += (-min_eig + 0.01) * np.eye(n)
        D = np.sqrt(np.diag(corr))
        corr = corr / np.outer(D, D)

    return corr


def generate_high_corr(n_assets=50, avg_corr=0.8):
    """Generate uniformly high correlation"""
    corr = np.ones((n_assets, n_assets)) * avg_corr
    np.fill_diagonal(corr, 1.0)

    noise = np.random.randn(n_assets, n_assets) * 0.01
    noise = (noise + noise.T) / 2
    corr = corr + noise

    min_eig = np.min(np.linalg.eigvals(corr))
    if min_eig < 0:
        corr += (-min_eig + 0.01) * np.eye(n_assets)
        D = np.sqrt(np.diag(corr))
        corr = corr / np.outer(D, D)

    return corr


def generate_low_corr(n_assets=50, avg_corr=0.2):
    """Generate uniformly low correlation"""
    corr = np.ones((n_assets, n_assets)) * avg_corr
    np.fill_diagonal(corr, 1.0)

    noise = np.random.randn(n_assets, n_assets) * 0.05
    noise = (noise + noise.T) / 2
    corr = corr + noise
    corr = np.clip(corr, -0.99, 0.99)
    np.fill_diagonal(corr, 1.0)

    min_eig = np.min(np.linalg.eigvals(corr))
    if min_eig < 0:
        corr += (-min_eig + 0.01) * np.eye(n_assets)
        D = np.sqrt(np.diag(corr))
        corr = corr / np.outer(D, D)

    return corr


def run_monte_carlo(scenario_name, corr_func, n_sims=10000, n_assets=50, n_obs=260):
    """
    Run Monte Carlo simulation for a scenario

    Following López de Prado's methodology:
    1. Generate true covariance
    2. Sample data with noise
    3. Estimate covariance from sample
    4. Construct portfolios
    5. Evaluate OOS using true covariance
    """
    logger.info(f"\nRunning: {scenario_name}")
    logger.info(f"  Simulations: {n_sims}, Assets: {n_assets}, Observations: {n_obs}")

    hrp_vars = []
    eq_vars = []
    iv_vars = []

    np.random.seed(42)

    for i in range(n_sims):
        if i % 1000 == 0 and i > 0:
            logger.info(f"  Progress: {i}/{n_sims}")

        # Generate true correlation and volatilities
        true_corr = corr_func(n_assets)
        true_vols = np.random.uniform(0.1, 0.3, n_assets)
        D = np.diag(true_vols)
        true_cov = D @ true_corr @ D

        # Sample returns
        returns = np.random.multivariate_normal(np.zeros(n_assets), true_cov, n_obs)

        # Estimate covariance from sample (with noise)
        sample_cov = np.cov(returns.T)
        sample_corr = np.corrcoef(returns.T)

        # Construct portfolios using SAMPLE covariance
        try:
            w_hrp = hrp_weights(sample_cov, sample_corr)
        except:
            w_hrp = np.ones(n_assets) / n_assets

        w_eq = np.ones(n_assets) / n_assets

        sample_vols = np.sqrt(np.diag(sample_cov))
        inv_vols = 1 / sample_vols
        w_iv = inv_vols / inv_vols.sum()

        # Evaluate OOS using TRUE covariance
        hrp_var = w_hrp @ true_cov @ w_hrp
        eq_var = w_eq @ true_cov @ w_eq
        iv_var = w_iv @ true_cov @ w_iv

        hrp_vars.append(hrp_var)
        eq_vars.append(eq_var)
        iv_vars.append(iv_var)

    # Analysis
    hrp_vars = np.array(hrp_vars)
    eq_vars = np.array(eq_vars)
    iv_vars = np.array(iv_vars)

    hrp_mean = np.mean(hrp_vars)
    eq_mean = np.mean(eq_vars)
    iv_mean = np.mean(iv_vars)

    hrp_beats_eq = (hrp_vars < eq_vars).mean()
    hrp_beats_iv = (hrp_vars < iv_vars).mean()

    improvement_vs_eq = (eq_mean - hrp_mean) / eq_mean * 100
    improvement_vs_iv = (iv_mean - hrp_mean) / iv_mean * 100

    # Compute average correlation
    sample_corr_test = corr_func(n_assets)
    n_off_diag = n_assets * (n_assets - 1)
    avg_corr = (sample_corr_test.sum() - n_assets) / n_off_diag

    logger.info(f"  ✓ Complete")
    logger.info(f"  HRP mean var:      {hrp_mean:.6f}")
    logger.info(f"  1/N mean var:      {eq_mean:.6f}")
    logger.info(f"  Inv-Vol mean var:  {iv_mean:.6f}")
    logger.info(f"  HRP beats 1/N:     {hrp_beats_eq*100:.1f}%")
    logger.info(f"  HRP improvement:   {improvement_vs_eq:+.2f}%")
    logger.info(f"  Avg correlation:   {avg_corr:.3f}")

    return {
        'scenario': scenario_name,
        'hrp_mean': hrp_mean,
        'eq_mean': eq_mean,
        'iv_mean': iv_mean,
        'hrp_beats_eq_pct': hrp_beats_eq * 100,
        'improvement_pct': improvement_vs_eq,
        'avg_corr': avg_corr
    }


def main():
    logger.info("="*70)
    logger.info("MONTE CARLO VALIDATION STUDY: HRP vs Naive Diversification")
    logger.info("="*70)

    scenarios = [
        ("Block-Diagonal (5 sectors, high intra-corr)",
         lambda n: generate_block_diagonal_corr(5, n//5, 0.7, 0.1)),

        ("Block-Diagonal (10 sectors, moderate intra-corr)",
         lambda n: generate_block_diagonal_corr(10, n//10, 0.5, 0.1)),

        ("High Correlation (0.8 - single sector)",
         lambda n: generate_high_corr(n, 0.8)),

        ("Moderate Correlation (0.5)",
         lambda n: generate_high_corr(n, 0.5)),

        ("Low Correlation (0.2 - diversified)",
         lambda n: generate_low_corr(n, 0.2)),

        ("Very High Correlation (0.9 - market crash)",
         lambda n: generate_high_corr(n, 0.9)),
    ]

    results = []

    for scenario_name, corr_func in scenarios:
        result = run_monte_carlo(scenario_name, corr_func, n_sims=10000)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    print(f"{'Scenario':<50s} {'HRP Beats 1/N':<15s} {'Improvement':<12s}")
    print("-"*70)

    for r in results:
        print(f"{r['scenario']:<50s} {r['hrp_beats_eq_pct']:>6.1f}%         {r['improvement_pct']:>+6.2f}%")

    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('simulation_results/monte_carlo_summary.csv', index=False)

    # Generate plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scenarios_short = [r['scenario'][:30] + '...' if len(r['scenario']) > 30 else r['scenario'] for r in results]
    success_rates = [r['hrp_beats_eq_pct'] for r in results]
    improvements = [r['improvement_pct'] for r in results]

    # Plot 1: Success rates
    colors = ['green' if sr > 50 else 'red' for sr in success_rates]
    axes[0].barh(range(len(scenarios_short)), success_rates, color=colors, edgecolor='black')
    axes[0].axvline(x=50, color='black', linestyle='--', label='50% baseline')
    axes[0].set_xlabel('Success Rate (%)')
    axes[0].set_ylabel('Scenario')
    axes[0].set_yticks(range(len(scenarios_short)))
    axes[0].set_yticklabels(scenarios_short, fontsize=9)
    axes[0].set_title('HRP Success Rate (% of times HRP beats 1/N)')
    axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)

    # Plot 2: Improvements
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    axes[1].barh(range(len(scenarios_short)), improvements, color=colors, edgecolor='black')
    axes[1].axvline(x=0, color='black', linestyle='--')
    axes[1].set_xlabel('Improvement (%)')
    axes[1].set_ylabel('Scenario')
    axes[1].set_yticks(range(len(scenarios_short)))
    axes[1].set_yticklabels(scenarios_short, fontsize=9)
    axes[1].set_title('HRP Performance Improvement over 1/N')
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('simulation_results/monte_carlo_results.png', dpi=300, bbox_inches='tight')
    logger.info("\n✓ Saved: simulation_results/monte_carlo_results.png")
    logger.info("✓ Saved: simulation_results/monte_carlo_summary.csv")

    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)

    best = max(results, key=lambda x: x['improvement_pct'])
    worst = min(results, key=lambda x: x['improvement_pct'])

    print(f"\n✓ Best scenario for HRP:")
    print(f"  {best['scenario']}")
    print(f"  Success rate: {best['hrp_beats_eq_pct']:.1f}%")
    print(f"  Improvement: {best['improvement_pct']:+.2f}%")

    print(f"\n✗ Worst scenario for HRP:")
    print(f"  {worst['scenario']}")
    print(f"  Success rate: {worst['hrp_beats_eq_pct']:.1f}%")
    print(f"  Improvement: {worst['improvement_pct']:+.2f}%")

    print("\n" + "="*70)


if __name__ == '__main__':
    import os
    os.makedirs('simulation_results', exist_ok=True)
    main()
