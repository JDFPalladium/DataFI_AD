import numpy as np
import pandas as pd
from scipy import linalg

from .covariance import stable_inv


def estimate_values(data: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    For each present value in each row, estimate the conditional expectation
    given all other present values in that row:
      E[x_j | x_{-j}] = mu_j + cov[j, -j] @ inv(cov[-j, -j]) @ (x_{-j} - mu[-j])

    Returns array of same shape as data; NaN where the original value was missing.
    """
    n, k = data.shape
    estimates = np.full((n, k), np.nan)

    for i in range(n):
        inds = np.where(~np.isnan(data[i]))[0]
        if len(inds) < 2:
            continue
        for j in inds:
            other = inds[inds != j]
            cov_j_other = cov[j, other]
            cov_other = cov[np.ix_(other, other)]
            try:
                inv_cov_other = stable_inv(cov_other)
                diff = data[i, other] - mu[other]
                estimates[i, j] = mu[j] + cov_j_other @ inv_cov_other @ diff
            except linalg.LinAlgError:
                estimates[i, j] = mu[j]

    return estimates


def compute_deviations(data: np.ndarray, estimates: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Normalized absolute deviation per cell: |actual - estimated| / var[j].
    NaN where the original value was missing.
    """
    variances = np.diag(cov)
    with np.errstate(divide="ignore", invalid="ignore"):
        deviations = np.abs(data - estimates) / variances
    return deviations


def build_output(
    df_processed: pd.DataFrame,
    id_cols: list,
    distances_sq: np.ndarray,
    estimates: np.ndarray,
    deviations: np.ndarray,
    outlier_flags: np.ndarray,
    min_value_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Assemble the final output DataFrame, sorted by outlier flag then distance.

    Columns:
      <id_cols>  — original ID fields
      <metric_cols> — reported values
      mahalanobis_distance_sq — squared Mahalanobis distance
      is_outlier — 1/0 flag
      top_deviation_indicator — metric column with the highest normalized deviation
                                (only for outliers; ignores values <= min_value_threshold)
      E_<col>  — estimated value for each metric column
      D_<col>  — normalized deviation for each metric column
    """
    metric_cols = [c for c in df_processed.columns if c not in id_cols]

    result = df_processed.copy().reset_index(drop=True)
    result["mahalanobis_distance_sq"] = distances_sq
    result["is_outlier"] = outlier_flags.astype(int)

    est_df = pd.DataFrame(estimates, columns=[f"E_{c}" for c in metric_cols])
    dev_df = pd.DataFrame(deviations, columns=[f"D_{c}" for c in metric_cols])
    result = pd.concat([result, est_df, dev_df], axis=1)

    data_vals = df_processed[metric_cols].values
    result["top_deviation_indicator"] = None

    for i in range(len(result)):
        if not outlier_flags[i]:
            continue
        dev_row = deviations[i].copy()
        dev_row[np.isnan(dev_row)] = 0.0
        dev_row[data_vals[i] <= min_value_threshold] = 0.0
        if dev_row.sum() == 0:
            continue
        result.at[i, "top_deviation_indicator"] = metric_cols[int(np.argmax(dev_row))]

    result = result.sort_values(
        ["is_outlier", "mahalanobis_distance_sq"], ascending=[False, False]
    ).reset_index(drop=True)

    return result
