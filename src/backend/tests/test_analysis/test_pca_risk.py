import numpy as np
import pandas as pd
import pytest

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


def test_cov_shrinkage_is_square_in_the_number_of_assets(config, multi_asset_returns):
    result = cov_shrinkage(config, multi_asset_returns)
    n_assets = multi_asset_returns.shape[1]
    assert result.shape == (n_assets, n_assets)


def test_cov_shrinkage_is_labelled_by_ticker_on_both_axes(config, multi_asset_returns):
    result = cov_shrinkage(config, multi_asset_returns)
    assert list(result.index) == list(multi_asset_returns.columns)
    assert list(result.columns) == list(multi_asset_returns.columns)


def test_cov_shrinkage_output_is_symmetric(config, multi_asset_returns):
    result = cov_shrinkage(config, multi_asset_returns)
    np.testing.assert_allclose(result.values, result.values.T)


def test_cov_shrinkage_is_positive_definite(config, multi_asset_returns):
    result = cov_shrinkage(config, multi_asset_returns)
    assert np.all(np.linalg.eigvalsh(result.values) > 0)


def test_cov_shrinkage_is_a_convex_blend_of_sample_cov_and_scaled_identity(
    config, multi_asset_returns
):
    # Ledoit-Wolf by definition returns (1 - d) * S + d * mu * I, with S the
    # ddof=0 sample covariance, mu = tr(S) / p and a single intensity d in [0, 1].
    result = cov_shrinkage(config, multi_asset_returns).values
    sample = multi_asset_returns.cov(ddof=0).values
    p = sample.shape[0]
    mu = np.trace(sample) / p
    target = mu * np.eye(p)

    d = 1 - result[0, 1] / sample[0, 1]
    assert 0.0 < d < 1.0
    np.testing.assert_allclose(result, (1 - d) * sample + d * target, rtol=1e-10)


def test_cov_shrinkage_preserves_total_variance(config, multi_asset_returns):
    # A consequence of shrinking toward mu * I: the trace is unchanged for any d.
    result = cov_shrinkage(config, multi_asset_returns)
    sample = multi_asset_returns.cov(ddof=0)
    assert np.trace(result.values) == pytest.approx(np.trace(sample.values), rel=1e-12)


def test_cov_shrinkage_drops_the_leading_nan_row_of_simple_returns(
    config, multi_asset_returns
):
    with_nan = multi_asset_returns.copy()
    with_nan.iloc[0] = np.nan
    result = cov_shrinkage(config, with_nan)
    expected = cov_shrinkage(config, multi_asset_returns.iloc[1:])
    assert np.isfinite(result.values).all()
    pd.testing.assert_frame_equal(result, expected)
