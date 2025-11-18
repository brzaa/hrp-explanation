"""
Unit tests for HRP implementation
Run with: pytest tests/test_hrp.py -v
"""


import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hrp_implementation_enhanced import (
    get_distance_matrix,
    tree_clustering,
    quasi_diagonalization,
    recursive_bisection,
    hrp_algorithm,
    create_equal_weight_portfolio,
    create_inverse_vol_portfolio,
    calculate_portfolio_returns,
    bootstrap_sharpe_test
)

@pytest.fixture
def sample_returns():
    """Generate sample return data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    
    # Create correlated returns
    n_assets = 10
    mean_returns = np.random.randn(n_assets) * 0.001
    cov_matrix = np.random.randn(n_assets, n_assets)
    cov_matrix = (cov_matrix @ cov_matrix.T) / 100
    
    returns_data = np.random.multivariate_normal(mean_returns, cov_matrix, size=252)
    
    returns = pd.DataFrame(
        returns_data,
        index=dates,
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    return returns

class TestDistanceMatrix:
    """Test distance matrix calculation."""
    
    def test_distance_range(self, sample_returns):
        """Distance should be in [0, 1]."""
        corr = sample_returns.corr()
        dist = get_distance_matrix(corr)
        
        assert np.all(dist >= 0), "Distance should be non-negative"
        assert np.all(dist <= 1), "Distance should be <= 1"
    
    def test_perfect_correlation(self):
        """Perfect correlation should give zero distance."""
        corr = pd.DataFrame(np.ones((3, 3)))
        dist = get_distance_matrix(corr)
        
        np.testing.assert_almost_equal(dist, 0, decimal=10)
    
    def test_zero_correlation(self):
        """Zero correlation should give distance ~0.707."""
        corr = pd.DataFrame(np.eye(3))
        dist = get_distance_matrix(corr)
        
        # Diagonal should be 0
        assert np.allclose(np.diag(dist), 0)
        
        # Off-diagonal should be sqrt(0.5)
        expected = np.sqrt(0.5)
        assert np.allclose(dist[0, 1], expected)

class TestTreeClustering:
    """Test hierarchical clustering."""
    
    def test_linkage_shape(self, sample_returns):
        """Linkage matrix should have correct shape."""
        linkage_matrix, corr_matrix = tree_clustering(sample_returns)
        
        n_assets = len(sample_returns.columns)
        expected_shape = (n_assets - 1, 4)
        
        assert linkage_matrix.shape == expected_shape
    
    def test_correlation_matrix(self, sample_returns):
        """Correlation matrix should be valid."""
        _, corr_matrix =
