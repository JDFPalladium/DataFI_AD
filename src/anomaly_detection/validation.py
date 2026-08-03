import numpy as np
import pandas as pd

from .covariance import corr_and_scale


# Thresholds
MIN_ROWS_ABSOLUTE = 20
MIN_ROWS_RELIABLE = 30       # for chi-square approximation
N_TO_K_HARD_FAIL = 5        # n < 5k → hard fail
N_TO_K_WARN = 10            # n < 10k → warning
MAX_MISSING_PER_COL = 0.30  # warn if any column > 30% missing
MAX_MISSING_OVERALL = 0.20  # warn if overall missingness > 20%
MAX_CONDITION_NUMBER = 1000  # hard fail if cov matrix is ill-conditioned


class DatasetError(ValueError):
    """Raised when the dataset fails a hard requirement for Mahalanobis distance."""


class DatasetWarning:
    """Represents a non-fatal concern about result reliability."""
    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message


def check_dataset_size(n: int, k: int) -> tuple:
    """
    Check sample size requirements against the number of metric columns.

    Returns (errors, warnings) — lists of DatasetError and DatasetWarning.
    Errors must be resolved before analysis; warnings are informational.
    """
    errors, warnings = [], []

    if n < MIN_ROWS_ABSOLUTE:
        errors.append(DatasetError(
            f"Dataset has only {n} rows. At least {MIN_ROWS_ABSOLUTE} are required for "
            "Mahalanobis distance to be meaningful."
        ))
        return errors, warnings  # no point checking ratios if n is tiny

    if n <= k:
        errors.append(DatasetError(
            f"Dataset has {n} rows and {k} metric columns (n ≤ k). "
            "The covariance matrix cannot be full-rank and therefore cannot be inverted. "
            "Either add more rows or reduce the number of metric columns."
        ))
        return errors, warnings

    ratio = n / k
    if ratio < N_TO_K_HARD_FAIL:
        errors.append(DatasetError(
            f"Dataset has {n} rows and {k} metric columns (ratio {ratio:.1f}:1). "
            f"A minimum of {N_TO_K_HARD_FAIL} rows per column is required for reliable "
            "covariance estimation. Either add more data or reduce the number of metric columns."
        ))
    elif ratio < N_TO_K_WARN:
        warnings.append(DatasetWarning(
            f"Dataset has {n} rows and {k} metric columns (ratio {ratio:.1f}:1). "
            f"A ratio of at least {N_TO_K_WARN}:1 is preferred. Results may be less reliable."
        ))

    if n < MIN_ROWS_RELIABLE:
        warnings.append(DatasetWarning(
            f"Dataset has only {n} rows. At least {MIN_ROWS_RELIABLE} rows are recommended "
            "for the chi-square approximation underlying the outlier cutoff to be accurate."
        ))

    return errors, warnings


def check_missing_data(data: np.ndarray, col_names: list) -> tuple:
    """
    Check missingness levels across the dataset.

    Returns (errors, warnings).
    """
    errors, warnings = [], []
    n, k = data.shape

    per_col_missing = np.isnan(data).mean(axis=0)
    overall_missing = np.isnan(data).mean()

    high_missing_cols = [
        col_names[j] for j in range(k) if per_col_missing[j] > MAX_MISSING_PER_COL
    ]
    if high_missing_cols:
        worst = max(per_col_missing)
        warnings.append(DatasetWarning(
            f"The following columns have >{MAX_MISSING_PER_COL:.0%} missing values, "
            f"which may reduce the reliability of the sparse covariance estimate: "
            f"{high_missing_cols}. (Worst: {worst:.0%} missing.)"
        ))

    if overall_missing > MAX_MISSING_OVERALL:
        warnings.append(DatasetWarning(
            f"Overall missing data rate is {overall_missing:.0%} "
            f"(threshold: {MAX_MISSING_OVERALL:.0%}). "
            "Results may be less reliable."
        ))

    return errors, warnings


def check_non_negative_data(data: np.ndarray, col_names: list) -> tuple:
    """
    log1p requires values > -1; flag any column that violates this before transforming.

    Returns (errors, warnings).
    """
    errors, warnings = [], []

    with np.errstate(invalid="ignore"):
        min_vals = np.nanmin(data, axis=0)

    bad_cols = [col_names[j] for j in range(len(col_names)) if min_vals[j] <= -1]
    if bad_cols:
        errors.append(DatasetError(
            f"The following columns contain values ≤ -1, which cannot be log-transformed: "
            f"{bad_cols}. This tool assumes non-negative count-style indicators; exclude or "
            "pre-process columns that can be negative (e.g. rates of change) before running."
        ))

    return errors, warnings


def check_covariance_matrix(cov: np.ndarray) -> tuple:
    """
    Check that the covariance matrix is well-conditioned and positive definite.

    Returns (errors, warnings).
    """
    errors, warnings = [], []

    _, corr = corr_and_scale(cov)
    eigenvalues = np.linalg.eigvalsh(corr)
    min_eig = float(eigenvalues.min())

    if min_eig <= 0:
        errors.append(DatasetError(
            f"The covariance matrix is not positive definite (smallest eigenvalue: {min_eig:.4g}). "
            "This usually means there are redundant or perfectly correlated columns remaining "
            "after preprocessing. Check your data for duplicate or linearly dependent columns."
        ))
        return errors, warnings

    condition_number = float(eigenvalues.max() / min_eig)
    if condition_number > MAX_CONDITION_NUMBER:
        errors.append(DatasetError(
            f"The covariance matrix is ill-conditioned (condition number: {condition_number:.1f}, "
            f"threshold: {MAX_CONDITION_NUMBER}). Matrix inversion will be numerically unstable. "
            "This can be caused by near-collinear columns or extreme differences in scale between "
            "variables. Consider standardizing your data or removing near-duplicate columns."
        ))

    return errors, warnings


def format_validation_report(errors: list, warnings: list) -> str:
    """Format errors and warnings into a human-readable string."""
    lines = []
    if errors:
        lines.append("ERRORS (analysis cannot proceed):")
        for e in errors:
            lines.append(f"  - {e}")
    if warnings:
        lines.append("WARNINGS (analysis will run, but review results carefully):")
        for w in warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines)
