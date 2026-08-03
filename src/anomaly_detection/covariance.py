import numpy as np
from scipy import linalg


def corr_and_scale(cov: np.ndarray) -> tuple:
    """Decompose cov as D @ Corr @ D. Returns (D diagonal vector, Corr)."""
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    return d, corr


def stable_inv(cov: np.ndarray) -> np.ndarray:
    """Invert a covariance matrix via its correlation form for numerical stability."""
    if cov.shape[0] == 1:
        return np.array([[1.0 / cov[0, 0]]])
    d, corr = corr_and_scale(cov)
    inv_corr = linalg.inv(corr)
    d_inv = 1.0 / d
    return inv_corr * np.outer(d_inv, d_inv)


def sparse_mean_cov(data: np.ndarray) -> tuple:
    """
    Compute mean vector and sparse covariance matrix from data with missing values.

    Replicates the R runRecAnalysis sparse covariance estimator:
      - mu[j] = sum of present values / count of present values per column
      - S = sum_i H_i^T (y_i - mu[inds_i])^T (y_i - mu[inds_i]) H_i
        where H_i is the row-selection matrix for row i's present indices
      - R = N^{-1/2} S N^{-1/2}, N[j,j] = count of present values for column j

    Parameters
    ----------
    data : (n, k) array with np.nan for missing values

    Returns
    -------
    mu : (k,) mean vector
    R  : (k, k) sparse covariance matrix
    """
    n, k = data.shape

    present_mask = ~np.isnan(data)
    count_present = present_mask.sum(axis=0).astype(float)

    if np.any(count_present == 0):
        raise ValueError("One or more columns have no non-missing values.")

    mu = np.nansum(data, axis=0) / count_present

    I = np.eye(k)
    S = np.zeros((k, k))

    for i in range(n):
        inds = np.where(present_mask[i])[0]
        if len(inds) == 0:
            continue
        yt = data[i, inds] - mu[inds]       # (p,)
        Hyt = I[inds, :]                    # (p, k)
        S += Hyt.T @ np.outer(yt, yt) @ Hyt

    N_sqrt_inv = np.diag(1.0 / np.sqrt(count_present))
    R = N_sqrt_inv @ S @ N_sqrt_inv

    return mu, R
