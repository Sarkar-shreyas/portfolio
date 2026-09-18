import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
from typing import Optional

from src.backend.config import DevConfig
from src.backend.analysis.risk_metrics import (
    get_correlation,
    get_covariance,
    rolling_correlation,
    rolling_covariance,
)


def pca_analysis(
    config: DevConfig, covariances: pd.DataFrame, n_components: Optional[int] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Performs a principal component analysis on the given covariance matrix and returns the specified number of components."""
    if n_components is None:
        n_components = config.pca_components

    eig_val, eig_vec = np.linalg.eigh(covariances)
    sorted_index = np.argsort(eig_val)[::-1]
    sorted_eigval = eig_val[sorted_index]
    sorted_eigvec = eig_vec[:, sorted_index]

    explained_variance = sorted_eigval / sorted_eigval.sum()
    cum_variance = explained_variance.cumsum()
    eigval_index = [f"PC{i + 1}" for i in range(len(sorted_eigval))]
    eigval_df = pd.DataFrame(
        {
            "eigenvalues": sorted_eigval,
            "explained_var": explained_variance,
            "cum_var": cum_variance,
        },
        index=eigval_index,
    )

    eigvec_df = pd.DataFrame(
        sorted_eigvec, index=covariances.index, columns=eigval_index
    )

    return eigval_df, eigvec_df


def cov_shrinkage(config: DevConfig, covariances: pd.DataFrame) -> pd.DataFrame:
    """Performs a Ledoit-Wolf shrinkage on the input covariance matrix."""
    cov = LedoitWolf().fit(covariances)
    shrunken_cov = cov.covariance_
    shrunken_cov_df = pd.DataFrame(shrunken_cov, index=covariances.index)
    return shrunken_cov_df
