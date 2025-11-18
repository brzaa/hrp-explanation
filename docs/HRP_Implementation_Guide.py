"""
Hierarchical Risk Parity (HRP) Implementation
Complete, documented implementation from scratch
Based on López de Prado (2016)

Author: Educational Implementation
Purpose: Understanding HRP through code
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)


# ============================================================================
# PART 1: DATA GENERATION AND PREPARATION
# ============================================================================

def generate_sample_data(n_assets=10, n_observations=252, block_structure=True):
    """
    Generate sample return data for testing HRP.

    Parameters:
    -----------
    n_assets : int
        Number of assets in the portfolio
    n_observations : int
        Number of time periods (e.g., 252 trading days = 1 year)
    block_structure : bool
        If True, creates assets with block correlation structure
        (some assets highly correlated within groups)

    Returns:
    --------
    returns_df : pandas.DataFrame
        DataFrame with returns for each asset
    """

    if block_structure:
        # Create a block diagonal covariance structure
        # This mimics real markets where assets cluster (e.g., tech stocks, energy stocks)

        # We'll create 3 groups of assets
        n_groups = 3
        assets_per_group = n_assets // n_groups

        # Base correlation within groups and between groups
        corr_within = 0.7  # High correlation within same sector
        corr_between = 0.2  # Low correlation between sectors

        # Build correlation matrix
        corr_matrix = np.ones((n_assets, n_assets)) * corr_between

        for i in range(n_groups):
            start_idx = i * assets_per_group
            end_idx = start_idx + assets_per_group
            corr_matrix[start_idx:end_idx, start_idx:end_idx] = corr_within

        # Set diagonal to 1
        np.fill_diagonal(corr_matrix, 1.0)

        # Generate random volatilities (annual volatility between 15% and 40%)
        volatilities = np.random.uniform(0.15, 0.40, n_assets)

        # Convert correlation to covariance
        cov_matrix = np.outer(volatilities, volatilities) * corr_matrix

        # Generate returns from multivariate normal
        # Scale to daily returns (divide annual vol by sqrt(252))
        daily_cov = cov_matrix / 252
        returns = np.random.multivariate_normal(
            mean=np.zeros(n_assets),
            cov=daily_cov,
            size=n_observations
        )
    else:
        # Simple random returns
        returns = np.random.randn(n_observations, n_assets) * 0.01

    # Create DataFrame with asset names
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    returns_df = pd.DataFrame(returns, columns=asset_names)

    return returns_df


def compute_covariance_matrix(returns_df):
    """
    Compute the covariance matrix from returns.

    Parameters:
    -----------
    returns_df : pandas.DataFrame
        DataFrame of asset returns

    Returns:
    --------
    cov_matrix : pandas.DataFrame
        Covariance matrix
    """
    return returns_df.cov()


def compute_correlation_matrix(returns_df):
    """
    Compute the correlation matrix from returns.

    Parameters:
    -----------
    returns_df : pandas.DataFrame
        DataFrame of asset returns

    Returns:
    --------
    corr_matrix : pandas.DataFrame
        Correlation matrix
    """
    return returns_df.corr()


# ============================================================================
# PART 2: HRP STEP 1 - TREE CLUSTERING
# ============================================================================

def correlation_to_distance(corr_matrix):
    """
    Convert correlation matrix to distance matrix.

    The formula is: d_ij = sqrt(0.5 * (1 - rho_ij))
    This ensures the result is a proper metric (satisfies triangle inequality).

    Parameters:
    -----------
    corr_matrix : pandas.DataFrame or numpy.ndarray
        Correlation matrix

    Returns:
    --------
    dist_matrix : numpy.ndarray
        Distance matrix
    """
    # Convert to numpy array if pandas DataFrame
    if isinstance(corr_matrix, pd.DataFrame):
        corr_array = corr_matrix.values
    else:
        corr_array = corr_matrix

    # Apply the distance formula
    dist_matrix = np.sqrt(0.5 * (1 - corr_array))

    return dist_matrix


def perform_hierarchical_clustering(dist_matrix, method='single'):
    """
    Perform hierarchical clustering on the distance matrix.

    Parameters:
    -----------
    dist_matrix : numpy.ndarray
        Distance matrix (symmetric, with zeros on diagonal)
    method : str
        Linkage method: 'single', 'complete', 'average', or 'ward'
        HRP paper uses 'single'

    Returns:
    --------
    linkage_matrix : numpy.ndarray
        Linkage matrix from scipy (encodes the dendrogram)
    """
    # Convert square distance matrix to condensed form
    # (scipy requires upper triangular part as 1D array)
    dist_condensed = squareform(dist_matrix, checks=False)

    # Perform hierarchical clustering
    linkage_matrix = sch.linkage(dist_condensed, method=method)

    return linkage_matrix


def get_cluster_ordering(linkage_matrix):
    """
    Get the optimal ordering of assets from the dendrogram.

    This ordering is used in Step 2 (quasi-diagonalization).

    Parameters:
    -----------
    linkage_matrix : numpy.ndarray
        Linkage matrix from hierarchical clustering

    Returns:
    --------
    ordered_indices : list
        Indices of assets in the order they appear in the dendrogram
    """
    # Get the optimal leaf ordering
    ordered_indices = sch.leaves_list(linkage_matrix)

    return ordered_indices


# ============================================================================
# PART 3: HRP STEP 2 - QUASI-DIAGONALIZATION
# ============================================================================

def quasi_diagonalize(cov_matrix, ordered_indices):
    """
    Reorder the covariance matrix according to the dendrogram structure.

    This creates a quasi-diagonal structure where similar assets
    are placed next to each other.

    Parameters:
    -----------
    cov_matrix : pandas.DataFrame
        Original covariance matrix
    ordered_indices : list
        Ordering from the dendrogram

    Returns:
    --------
    reordered_cov : pandas.DataFrame
        Reordered covariance matrix
    """
    # Reorder both rows and columns
    asset_names = cov_matrix.index
    ordered_names = [asset_names[i] for i in ordered_indices]

    reordered_cov = cov_matrix.loc[ordered_names, ordered_names]

    return reordered_cov


# ============================================================================
# PART 4: HRP STEP 3 - RECURSIVE BISECTION
# ============================================================================

def get_inverse_variance_weights(cov_matrix):
    """
    Compute inverse-variance portfolio weights.

    For a set of assets, allocate inversely to variance:
    w_i = (1/var_i) / sum(1/var_j)

    Parameters:
    -----------
    cov_matrix : pandas.DataFrame or numpy.ndarray
        Covariance matrix (or sub-matrix for a cluster)

    Returns:
    --------
    weights : numpy.ndarray
        Inverse-variance weights (sum to 1)
    """
    # Get variances (diagonal of covariance matrix)
    if isinstance(cov_matrix, pd.DataFrame):
        variances = np.diag(cov_matrix.values)
    else:
        variances = np.diag(cov_matrix)

    # Compute inverse-variance weights
    inv_var = 1.0 / variances
    weights = inv_var / inv_var.sum()

    return weights


def get_cluster_variance(cov_matrix, cluster_indices=None):
    """
    Compute the variance of a cluster using inverse-variance weighting.

    Parameters:
    -----------
    cov_matrix : pandas.DataFrame
        Full covariance matrix
    cluster_indices : list or None
        Indices of assets in this cluster (None means all assets)

    Returns:
    --------
    cluster_var : float
        Variance of the cluster portfolio
    """
    if cluster_indices is None:
        cluster_indices = list(range(len(cov_matrix)))

    # Extract sub-covariance matrix for this cluster
    if isinstance(cov_matrix, pd.DataFrame):
        cluster_cov = cov_matrix.iloc[cluster_indices, cluster_indices].values
    else:
        cluster_cov = cov_matrix[np.ix_(cluster_indices, cluster_indices)]

    # Get inverse-variance weights for the cluster
    weights = get_inverse_variance_weights(cluster_cov)

    # Compute portfolio variance: w^T * Sigma * w
    cluster_var = np.dot(weights, np.dot(cluster_cov, weights))

    return cluster_var


def recursive_bisection(cov_matrix, ordered_indices):
    """
    Allocate weights using recursive bisection down the dendrogram.

    This is the core of HRP. We recursively split clusters and allocate
    weights inversely proportional to cluster variance.

    Parameters:
    -----------
    cov_matrix : pandas.DataFrame
        Covariance matrix (should be reordered)
    ordered_indices : list
        Ordering from the dendrogram

    Returns:
    --------
    weights : pandas.Series
        HRP portfolio weights (sum to 1)
    """
    # Initialize weights to 1 for all assets
    weights = pd.Series(1.0, index=ordered_indices)

    # Build the cluster tree by recursively splitting
    clusters = [ordered_indices]  # Start with all assets in one cluster

    while len(clusters) > 0:
        # Split each cluster into two sub-clusters
        new_clusters = []

        for cluster in clusters:
            if len(cluster) > 1:
                # Split cluster in half (based on dendrogram ordering)
                mid = len(cluster) // 2
                left_cluster = cluster[:mid]
                right_cluster = cluster[mid:]

                # Compute variance of each sub-cluster
                left_var = get_cluster_variance(cov_matrix, left_cluster)
                right_var = get_cluster_variance(cov_matrix, right_cluster)

                # Allocate weight to sub-clusters inversely to their variance
                # (lower variance cluster gets more weight)
                total_var = left_var + right_var
                left_weight = right_var / total_var
                right_weight = left_var / total_var

                # Update weights: multiply current cluster weight by allocation
                current_weight = weights[cluster[0]]  # All assets in cluster have same weight
                weights[left_cluster] *= left_weight
                weights[right_cluster] *= right_weight

                # Add sub-clusters to the list for further splitting
                new_clusters.append(left_cluster)
                new_clusters.append(right_cluster)

        clusters = new_clusters

    # Normalize weights to sum to 1 (should already be normalized)
    weights = weights / weights.sum()

    return weights


# ============================================================================
# PART 5: COMPLETE HRP ALGORITHM
# ============================================================================

def hierarchical_risk_parity(returns_df, linkage_method='single'):
    """
    Complete HRP algorithm: combines all three steps.

    Parameters:
    -----------
    returns_df : pandas.DataFrame
        Asset returns (each column is an asset)
    linkage_method : str
        Hierarchical clustering method ('single', 'complete', 'average', 'ward')

    Returns:
    --------
    weights : pandas.Series
        HRP portfolio weights
    additional_info : dict
        Dictionary with intermediate results for analysis
    """
    print("=" * 60)
    print("HIERARCHICAL RISK PARITY (HRP) ALGORITHM")
    print("=" * 60)

    # Compute correlation and covariance matrices
    print("\n[1/4] Computing correlation and covariance matrices...")
    corr_matrix = compute_correlation_matrix(returns_df)
    cov_matrix = compute_covariance_matrix(returns_df)

    # STEP 1: Tree Clustering
    print("[2/4] Step 1: Tree Clustering...")
    dist_matrix = correlation_to_distance(corr_matrix)
    linkage_matrix = perform_hierarchical_clustering(dist_matrix, method=linkage_method)
    ordered_indices = get_cluster_ordering(linkage_matrix)
    print(f"    Clustering complete. Asset ordering: {ordered_indices}")

    # STEP 2: Quasi-Diagonalization
    print("[3/4] Step 2: Quasi-Diagonalization...")
    reordered_cov = quasi_diagonalize(cov_matrix, ordered_indices)
    print("    Covariance matrix reordered.")

    # STEP 3: Recursive Bisection
    print("[4/4] Step 3: Recursive Bisection...")
    weights = recursive_bisection(reordered_cov, ordered_indices)

    # Reindex weights to original asset order
    weights = weights.reindex(returns_df.columns)

    print("\n" + "=" * 60)
    print("HRP ALGORITHM COMPLETE")
    print("=" * 60)

    # Package additional information
    additional_info = {
        'correlation_matrix': corr_matrix,
        'covariance_matrix': cov_matrix,
        'distance_matrix': dist_matrix,
        'linkage_matrix': linkage_matrix,
        'ordered_indices': ordered_indices,
        'reordered_covariance': reordered_cov
    }

    return weights, additional_info


# ============================================================================
# PART 6: COMPARISON WITH OTHER METHODS
# ============================================================================

def equal_weight_portfolio(returns_df):
    """
    Simple equal-weight (1/N) portfolio.

    Parameters:
    -----------
    returns_df : pandas.DataFrame
        Asset returns

    Returns:
    --------
    weights : pandas.Series
        Equal weights
    """
    n_assets = len(returns_df.columns)
    weights = pd.Series(1.0 / n_assets, index=returns_df.columns)
    return weights


def inverse_variance_portfolio(returns_df):
    """
    Inverse-variance portfolio (IVP / Risk Parity without correlations).

    Allocates inversely to variance: w_i ∝ 1/σ_i²

    Parameters:
    -----------
    returns_df : pandas.DataFrame
        Asset returns

    Returns:
    --------
    weights : pandas.Series
        Inverse-variance weights
    """
    variances = returns_df.var()
    inv_var = 1.0 / variances
    weights = inv_var / inv_var.sum()
    return weights


def minimum_variance_portfolio(returns_df):
    """
    Markowitz minimum variance portfolio.

    Solves: min w^T Σ w  subject to  w^T 1 = 1
    Solution: w = Σ^(-1) 1 / (1^T Σ^(-1) 1)

    Parameters:
    -----------
    returns_df : pandas.DataFrame
        Asset returns

    Returns:
    --------
    weights : pandas.Series
        Minimum variance weights (may have negative values if short-selling allowed)
    """
    cov_matrix = returns_df.cov().values
    n_assets = len(cov_matrix)

    # Invert covariance matrix
    try:
        cov_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        print("Warning: Covariance matrix is singular. Adding small regularization.")
        cov_inv = np.linalg.inv(cov_matrix + np.eye(n_assets) * 1e-6)

    # Compute minimum variance weights
    ones = np.ones(n_assets)
    weights = cov_inv @ ones / (ones @ cov_inv @ ones)

    # Convert to pandas Series
    weights = pd.Series(weights, index=returns_df.columns)

    return weights


# ============================================================================
# PART 7: PORTFOLIO ANALYSIS AND METRICS
# ============================================================================

def portfolio_performance(weights, returns_df, annualization_factor=252):
    """
    Compute portfolio performance metrics.

    Parameters:
    -----------
    weights : pandas.Series
        Portfolio weights
    returns_df : pandas.DataFrame
        Historical returns
    annualization_factor : int
        Factor to annualize returns and volatility (252 for daily data)

    Returns:
    --------
    metrics : dict
        Dictionary with performance metrics
    """
    # Portfolio returns
    portfolio_returns = (returns_df * weights).sum(axis=1)

    # Annualized return
    mean_return = portfolio_returns.mean() * annualization_factor

    # Annualized volatility
    volatility = portfolio_returns.std() * np.sqrt(annualization_factor)

    # Sharpe ratio (assuming risk-free rate = 0)
    sharpe_ratio = mean_return / volatility if volatility > 0 else 0

    # Maximum drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    metrics = {
        'Annualized Return': mean_return,
        'Annualized Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Maximum Drawdown': max_drawdown
    }

    return metrics, portfolio_returns


def compare_portfolios(returns_df):
    """
    Compare HRP with other portfolio construction methods.

    Parameters:
    -----------
    returns_df : pandas.DataFrame
        Asset returns

    Returns:
    --------
    comparison_df : pandas.DataFrame
        Comparison table
    """
    print("\n" + "=" * 60)
    print("PORTFOLIO COMPARISON")
    print("=" * 60)

    # Compute weights for each method
    print("\nComputing portfolio weights...")
    weights_hrp, _ = hierarchical_risk_parity(returns_df)
    weights_equal = equal_weight_portfolio(returns_df)
    weights_ivp = inverse_variance_portfolio(returns_df)
    weights_mv = minimum_variance_portfolio(returns_df)

    # Compute performance metrics
    print("\nComputing performance metrics...")
    metrics_hrp, _ = portfolio_performance(weights_hrp, returns_df)
    metrics_equal, _ = portfolio_performance(weights_equal, returns_df)
    metrics_ivp, _ = portfolio_performance(weights_ivp, returns_df)
    metrics_mv, _ = portfolio_performance(weights_mv, returns_df)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Equal Weight (1/N)': metrics_equal,
        'Inverse Variance': metrics_ivp,
        'Minimum Variance': metrics_mv,
        'HRP': metrics_hrp
    }).T

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(comparison_df.round(4))

    return comparison_df


# ============================================================================
# PART 8: VISUALIZATION
# ============================================================================

def plot_dendrogram(linkage_matrix, asset_names):
    """
    Plot the hierarchical clustering dendrogram.

    Parameters:
    -----------
    linkage_matrix : numpy.ndarray
        Linkage matrix from hierarchical clustering
    asset_names : list
        Names of assets
    """
    plt.figure(figsize=(12, 6))
    sch.dendrogram(linkage_matrix, labels=asset_names, leaf_rotation=90)
    plt.title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
    plt.xlabel('Assets', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrices(corr_original, corr_reordered, asset_names):
    """
    Plot original and reordered correlation matrices side by side.

    Parameters:
    -----------
    corr_original : pandas.DataFrame
        Original correlation matrix
    corr_reordered : pandas.DataFrame
        Reordered correlation matrix
    asset_names : list
        Names of assets
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Original correlation matrix
    sns.heatmap(corr_original, annot=False, cmap='RdYlGn', center=0,
                square=True, ax=axes[0], cbar_kws={'label': 'Correlation'})
    axes[0].set_title('Original Correlation Matrix', fontsize=12, fontweight='bold')

    # Reordered correlation matrix
    sns.heatmap(corr_reordered, annot=False, cmap='RdYlGn', center=0,
                square=True, ax=axes[1], cbar_kws={'label': 'Correlation'})
    axes[1].set_title('Reordered Correlation Matrix\n(Quasi-Diagonalized)',
                     fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_portfolio_weights(weights_dict):
    """
    Plot portfolio weights comparison across methods.

    Parameters:
    -----------
    weights_dict : dict
        Dictionary mapping method names to weight Series
    """
    weights_df = pd.DataFrame(weights_dict)

    fig, ax = plt.subplots(figsize=(12, 6))
    weights_df.plot(kind='bar', ax=ax)
    ax.set_title('Portfolio Weights Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Assets', fontsize=12)
    ax.set_ylabel('Weight', fontsize=12)
    ax.legend(title='Method')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_cumulative_returns(returns_df, weights_dict):
    """
    Plot cumulative returns for different portfolio strategies.

    Parameters:
    -----------
    returns_df : pandas.DataFrame
        Asset returns
    weights_dict : dict
        Dictionary mapping method names to weight Series
    """
    plt.figure(figsize=(12, 6))

    for method_name, weights in weights_dict.items():
        portfolio_returns = (returns_df * weights).sum(axis=1)
        cumulative = (1 + portfolio_returns).cumprod()
        plt.plot(cumulative, label=method_name, linewidth=2)

    plt.title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# PART 9: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to demonstrate HRP implementation.
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "HIERARCHICAL RISK PARITY (HRP)")
    print(" " * 20 + "Implementation Demo")
    print("=" * 70)

    # Generate sample data
    print("\n[STEP 1] Generating sample data...")
    print("-" * 70)
    n_assets = 10
    n_observations = 252  # 1 year of daily data
    returns_df = generate_sample_data(n_assets, n_observations, block_structure=True)
    print(f"Generated returns for {n_assets} assets over {n_observations} periods")
    print(f"\nFirst 5 rows of returns:\n{returns_df.head()}")

    # Run HRP
    print("\n[STEP 2] Running HRP Algorithm...")
    print("-" * 70)
    weights_hrp, info = hierarchical_risk_parity(returns_df)

    print("\n[STEP 3] HRP Weights:")
    print("-" * 70)
    for asset, weight in weights_hrp.items():
        print(f"{asset}: {weight:.4f} ({weight*100:.2f}%)")

    # Compare with other methods
    print("\n[STEP 4] Comparing with Other Methods...")
    print("-" * 70)
    weights_equal = equal_weight_portfolio(returns_df)
    weights_ivp = inverse_variance_portfolio(returns_df)
    weights_mv = minimum_variance_portfolio(returns_df)

    comparison_df = compare_portfolios(returns_df)

    # Visualizations
    print("\n[STEP 5] Generating Visualizations...")
    print("-" * 70)

    # Plot dendrogram
    plot_dendrogram(info['linkage_matrix'], list(returns_df.columns))

    # Plot correlation matrices
    plot_correlation_matrices(
        info['correlation_matrix'],
        info['reordered_covariance'].corr(),
        list(returns_df.columns)
    )

    # Plot weights comparison
    weights_dict = {
        'Equal Weight': weights_equal,
        'Inverse Variance': weights_ivp,
        'Min Variance': weights_mv,
        'HRP': weights_hrp
    }
    plot_portfolio_weights(weights_dict)

    # Plot cumulative returns
    plot_cumulative_returns(returns_df, weights_dict)

    print("\n" + "=" * 70)
    print(" " * 25 + "DEMO COMPLETE")
    print("=" * 70)
    print("\nKey Insights:")
    print("1. HRP produces diversified weights (all assets have positive weight)")
    print("2. HRP typically has lower out-of-sample volatility than Min Variance")
    print("3. The dendrogram shows the hierarchical structure HRP exploits")
    print("4. Quasi-diagonalization creates block structure in covariance matrix")
    print("\nNext steps:")
    print("- Try with your own return data")
    print("- Experiment with different linkage methods")
    print("- Implement out-of-sample testing")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
