import numpy as np
import pandas as pd


def validate_columns(df: pd.DataFrame, id_cols: list) -> dict:
    """
    Check that id_cols exist and classify remaining columns as numeric or not.
    Returns dict with keys: missing_id_cols, non_numeric_cols, metric_cols.
    """
    missing_id_cols = [c for c in id_cols if c not in df.columns]
    remaining = [c for c in df.columns if c not in id_cols]

    non_numeric, numeric = [], []
    for col in remaining:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            non_numeric.append(col)

    return {
        "missing_id_cols": missing_id_cols,
        "non_numeric_cols": non_numeric,
        "metric_cols": numeric,
    }


def drop_sparse_columns(df: pd.DataFrame, id_cols: list, min_present_frac: float = 0.10) -> pd.DataFrame:
    """Drop metric columns present in fewer than min_present_frac of rows."""
    metric_cols = [c for c in df.columns if c not in id_cols]
    present_frac = df[metric_cols].notna().mean()
    keep = present_frac[present_frac >= min_present_frac].index.tolist()
    return df[id_cols + keep]


def drop_zero_variance_columns(df: pd.DataFrame, id_cols: list) -> pd.DataFrame:
    """Drop metric columns with zero variance."""
    metric_cols = [c for c in df.columns if c not in id_cols]
    variances = df[metric_cols].var(skipna=True)
    keep = variances[variances > 0].index.tolist()
    return df[id_cols + keep]


def drop_collinear_columns(df: pd.DataFrame, id_cols: list, threshold: float = 0.95) -> pd.DataFrame:
    """
    Iteratively drop one column from each highly-correlated pair (|r| > threshold).

    Correlation is judged on log1p-transformed values, since that's the space
    covariance/distance are actually computed in downstream — a nonlinear transform
    can change which columns look collinear.
    """
    metric_cols = [c for c in df.columns if c not in id_cols]
    data = np.log1p(df[metric_cols].copy())

    while True:
        corr = data.corr(method="pearson", min_periods=1).abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        high = upper.stack()
        high = high[high > threshold]
        if high.empty:
            break
        col_to_drop = high.index[0][0]
        data = data.drop(columns=[col_to_drop])

    return df[id_cols + data.columns.tolist()]


def drop_sparse_rows(df: pd.DataFrame, id_cols: list, min_present: int = 4) -> pd.DataFrame:
    """Drop rows with fewer than min_present non-null metric values."""
    metric_cols = [c for c in df.columns if c not in id_cols]
    present_counts = df[metric_cols].notna().sum(axis=1)
    return df[present_counts >= min_present].reset_index(drop=True)


def preprocess(df: pd.DataFrame, id_cols: list) -> tuple:
    """
    Run full preprocessing pipeline. Returns (cleaned_df, report_dict).
    """
    original_metric = set(df.columns) - set(id_cols)
    original_rows = len(df)

    df = drop_sparse_columns(df, id_cols)
    after_sparse = set(df.columns) - set(id_cols)

    df = drop_zero_variance_columns(df, id_cols)
    after_var = set(df.columns) - set(id_cols)

    df = drop_collinear_columns(df, id_cols)
    after_collinear = set(df.columns) - set(id_cols)

    df = drop_sparse_rows(df, id_cols)
    after_rows = len(df)

    report = {
        "dropped_sparse_cols": sorted(original_metric - after_sparse),
        "dropped_zero_var_cols": sorted(after_sparse - after_var),
        "dropped_collinear_cols": sorted(after_var - after_collinear),
        "dropped_sparse_rows": original_rows - after_rows,
        "remaining_metric_cols": sorted(after_collinear),
        "remaining_rows": after_rows,
    }
    return df, report
