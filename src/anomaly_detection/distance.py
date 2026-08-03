import numpy as np
from scipy import linalg

from .covariance import stable_inv


def mahalanobis_missing(data: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Compute squared Mahalanobis distance for each row using only present values.

    For row i with present indices `inds`:
      D^2_i = (x[inds] - mu[inds])^T @ inv(cov[inds,:][:,inds]) @ (x[inds] - mu[inds])

    Returns array of squared distances (to compare against chi-square quantile).
    """
    n = data.shape[0]
    distances_sq = np.full(n, np.nan)

    for i in range(n):
        inds = np.where(~np.isnan(data[i]))[0]
        if len(inds) < 2:
            continue
        x = data[i, inds] - mu[inds]
        sub_cov = cov[np.ix_(inds, inds)]
        try:
            inv_cov = stable_inv(sub_cov)
            d_sq = float(x @ inv_cov @ x)
            distances_sq[i] = max(d_sq, 0.0)
        except linalg.LinAlgError:
            continue

    return distances_sq
