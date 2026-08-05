import numpy as np
import pandas as pd
from scipy.stats import chi2

from .preprocessing import validate_columns, preprocess
from .covariance import sparse_mean_cov
from .distance import mahalanobis_missing
from .output import estimate_values, compute_deviations, build_output
from .validation import (
    check_dataset_size, check_missing_data, check_covariance_matrix,
    check_non_negative_data, format_validation_report,
)


def run_detection(
    df: pd.DataFrame,
    id_cols: list,
    confidence: float = 0.95,
    min_value_threshold: float = 0.0,
) -> tuple:
    """
    Run the full anomaly detection pipeline on a tabular dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. Must contain all id_cols plus numeric metric columns.
    id_cols : list of str
        Columns that identify rows (excluded from distance calculation).
    confidence : float
        Chi-square quantile used as the outlier cutoff (default 0.95).
    min_value_threshold : float
        Reported values at or below this are ignored when identifying the
        top deviation indicator (useful to suppress noise from near-zero cells).

    Returns
    -------
    result_df : pd.DataFrame
        All rows sorted by anomaly flag (desc) then Mahalanobis distance (desc),
        with added columns: mahalanobis_distance, anomalous,
        top_deviation_indicator, and E_<col>.
    report : dict
        Summary of preprocessing steps and detection results.
    """
    validation = validate_columns(df, id_cols)

    if validation["missing_id_cols"]:
        raise ValueError(f"ID columns not found in data: {validation['missing_id_cols']}")

    # Drop non-numeric columns before preprocessing
    cols_to_use = id_cols + validation["metric_cols"]
    df_numeric = df[cols_to_use].copy()

    # Must run before preprocess(), since drop_collinear_columns applies log1p internally
    raw_metric_cols = validation["metric_cols"]
    raw_data_pre = df_numeric[raw_metric_cols].values.astype(float)
    neg_errors, _ = check_non_negative_data(raw_data_pre, raw_metric_cols)
    if neg_errors:
        raise ValueError(format_validation_report(neg_errors, []))

    df_clean, prep_report = preprocess(df_numeric, id_cols)
    metric_cols = [c for c in df_clean.columns if c not in id_cols]

    if len(metric_cols) < 2:
        raise ValueError(
            f"Only {len(metric_cols)} metric column(s) remain after preprocessing. "
            "At least 2 are required to compute Mahalanobis distance."
        )

    # All downstream computation happens in log1p space to correct for the heavy-tailed,
    # skewed distribution of raw facility counts (see chi-square calibration analysis).
    data_raw = df_clean[metric_cols].values.astype(float)
    data = np.log1p(data_raw)
    n, k = data.shape

    errors, warnings = [], []

    size_errors, size_warnings = check_dataset_size(n, k)
    errors.extend(size_errors)
    warnings.extend(size_warnings)

    missing_errors, missing_warnings = check_missing_data(data, metric_cols)
    errors.extend(missing_errors)
    warnings.extend(missing_warnings)

    if errors:
        raise ValueError(format_validation_report(errors, warnings))

    mu, cov = sparse_mean_cov(data)

    cov_errors, cov_warnings = check_covariance_matrix(cov)
    warnings.extend(cov_warnings)
    if cov_errors:
        raise ValueError(format_validation_report(cov_errors, cov_warnings))

    distances_sq = mahalanobis_missing(data, mu, cov)

    # With missing values, each row has an effective dimension p_i = number of observed metrics.
    # Compare D^2_i against chi2(p_i), not chi2(k), and rank by tail probability.
    observed_metrics = np.sum(~np.isnan(data), axis=1).astype(int)
    valid_mask = np.isfinite(distances_sq) & (observed_metrics >= 2)

    row_cutoffs = np.full(len(distances_sq), np.nan, dtype=float)
    row_cutoffs[valid_mask] = chi2.ppf(confidence, df=observed_metrics[valid_mask])

    outlier_flags = np.zeros(len(distances_sq), dtype=int)
    outlier_flags[valid_mask] = (distances_sq[valid_mask] > row_cutoffs[valid_mask]).astype(int)

    mahalanobis_pvalue = np.full(len(distances_sq), np.nan, dtype=float)
    mahalanobis_pvalue[valid_mask] = 1.0 - chi2.cdf(distances_sq[valid_mask], df=observed_metrics[valid_mask])

    # Larger score = more anomalous; clipped to avoid inf for extreme tail probabilities.
    anomaly_score = np.full(len(distances_sq), np.nan, dtype=float)
    mahalanobis_pvalue_safe = np.clip(mahalanobis_pvalue[valid_mask], 1e-300, 1.0)
    anomaly_score[valid_mask] = -np.log10(mahalanobis_pvalue_safe)

    estimates = estimate_values(data, mu, cov)
    deviations = compute_deviations(data, estimates, cov)

    # Estimates are in log1p space; convert back to raw units for display.
    # Deviations stay in log1p space — they're already an abstract normalized score,
    # not a raw-unit quantity, and that's the space top_deviation_indicator should rank in.
    estimates_display = np.expm1(estimates)

    result = build_output(
        df_processed=df_clean,
        id_cols=id_cols,
        distances_sq=distances_sq,
        estimates=estimates_display,
        deviations=deviations,
        outlier_flags=outlier_flags,
        observed_metrics=observed_metrics,
        mahalanobis_pvalue=mahalanobis_pvalue,
        anomaly_score=anomaly_score,
        min_value_threshold=min_value_threshold,
    )

    report = {
        **prep_report,
        "non_numeric_cols_dropped": validation["non_numeric_cols"],
        "n_outliers": int(outlier_flags.sum()),
        "chi2_cutoff": float(chi2.ppf(confidence, df=len(metric_cols))),
        "chi2_cutoff_by_observed_metrics": {
            int(p): float(chi2.ppf(confidence, df=int(p))) for p in sorted(set(observed_metrics.tolist())) if p >= 2
        },
        "confidence": confidence,
        "metric_cols_used": metric_cols,
        "analysis_data": data_raw,
        "estimated_data": estimates_display,
        "covariance_matrix": cov,
        "deviations": deviations,
        "observed_metrics": observed_metrics,
        "mahalanobis_pvalue": mahalanobis_pvalue,
        "anomaly_score": anomaly_score,
        "id_cols": id_cols,
        "warnings": [str(w) for w in warnings],
    }

    return result, report
