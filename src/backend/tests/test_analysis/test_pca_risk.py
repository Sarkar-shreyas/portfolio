import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from src.backend.analysis.pca_risk import pca_analysis, cov_shrinkage


@pytest.fixture
def covariance_matrix(multi_asset_returns) -> pd.DataFrame:
    return multi_asset_returns.cov()


# ---------------------------------------------------------------------------
# pca_analysis()
# ---------------------------------------------------------------------------


def test_pca_analysis_returns_expected_shapes(config, covariance_matrix):
    eigval_df, eigvec_df = pca_analysis(config, covariance_matrix)
    n_assets = len(covariance_matrix)
    assert eigval_df.shape == (n_assets, 3)
    assert list(eigval_df.columns) == ["eigenvalues", "explained_var", "cum_var"]
    assert eigvec_df.shape == (n_assets, n_assets)


def test_pca_analysis_eigenvalues_sorted_descending(config, covariance_matrix):
    eigval_df, _ = pca_analysis(config, covariance_matrix)
    eigenvalues = eigval_df["eigenvalues"].values
    assert np.all(np.diff(eigenvalues) <= 0)


def test_pca_analysis_explained_variance_sums_to_one(config, covariance_matrix):
    eigval_df, _ = pca_analysis(config, covariance_matrix)
    assert eigval_df["explained_var"].sum() == pytest.approx(1.0)


def test_pca_analysis_cumulative_variance_is_monotonic_and_ends_at_one(
    config, covariance_matrix
):
    eigval_df, _ = pca_analysis(config, covariance_matrix)
    cum_var = eigval_df["cum_var"].values
    assert np.all(np.diff(cum_var) >= -1e-12)
    assert cum_var[-1] == pytest.approx(1.0)


def test_pca_analysis_eigenvectors_are_orthonormal(config, covariance_matrix):
    _, eigvec_df = pca_analysis(config, covariance_matrix)
    identity_approx = eigvec_df.values.T @ eigvec_df.values
    np.testing.assert_allclose(identity_approx, np.eye(len(eigvec_df)), atol=1e-8)


def test_pca_analysis_reconstructs_original_covariance(config, covariance_matrix):
    eigval_df, eigvec_df = pca_analysis(config, covariance_matrix)
    reconstructed = (
        eigvec_df.values @ np.diag(eigval_df["eigenvalues"].values) @ eigvec_df.values.T
    )
    np.testing.assert_allclose(reconstructed, covariance_matrix.values, atol=1e-10)


def test_pca_analysis_eigvec_index_matches_covariance_index(config, covariance_matrix):
    _, eigvec_df = pca_analysis(config, covariance_matrix)
    assert list(eigvec_df.index) == list(covariance_matrix.index)


# ---------------------------------------------------------------------------
# cov_shrinkage()
# ---------------------------------------------------------------------------


def test_cov_shrinkage_matches_manual_ledoit_wolf_call(config, covariance_matrix):
    result = cov_shrinkage(config, covariance_matrix)
    expected = pd.DataFrame(
        LedoitWolf().fit(covariance_matrix).covariance_, index=covariance_matrix.index
    )
    pd.testing.assert_frame_equal(result, expected)


def test_cov_shrinkage_output_shape_matches_input(config, covariance_matrix):
    result = cov_shrinkage(config, covariance_matrix)
    assert result.shape == covariance_matrix.shape


def test_cov_shrinkage_output_is_symmetric(config, covariance_matrix):
    result = cov_shrinkage(config, covariance_matrix)
    np.testing.assert_allclose(result.values, result.values.T)


def test_cov_shrinkage_output_index_matches_input(config, covariance_matrix):
    result = cov_shrinkage(config, covariance_matrix)
    assert list(result.index) == list(covariance_matrix.index)
