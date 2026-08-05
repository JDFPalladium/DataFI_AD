#!/usr/bin/env python3
"""
Anomaly Detection CLI

Usage:
  python main.py input.csv output.csv --id-cols facility_id facility_name

The tool detects multivariate outliers using Mahalanobis distance with a
sparse covariance estimator that handles missing values.
"""
import argparse
import sys

import pandas as pd

from src.anomaly_detection.detection import run_detection
from src.anomaly_detection.output import write_output
from src.anomaly_detection.preprocessing import validate_columns


def main():
    parser = argparse.ArgumentParser(
        description="Detect anomalies in tabular data using Mahalanobis distance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Path to input CSV file")
    parser.add_argument("output", help="Path to output CSV file")
    parser.add_argument(
        "--id-cols", required=True, nargs="+",
        metavar="COL",
        help="Names of ID/key columns (space-separated). These identify rows and are excluded from the distance calculation.",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.95,
        help="Chi-square confidence level for the outlier cutoff (default: 0.95).",
    )
    parser.add_argument(
        "--min-value", type=float, default=0.0,
        help=(
            "Reported values at or below this threshold are ignored when identifying "
            "the top deviation indicator. Useful to suppress near-zero noise (default: 0)."
        ),
    )
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    print(f"  {len(df)} rows, {len(df.columns)} columns\n")

    # Report on column types before running
    validation = validate_columns(df, args.id_cols)

    if validation["missing_id_cols"]:
        print(f"ERROR: The following ID columns were not found in the data:")
        for col in validation["missing_id_cols"]:
            print(f"  - {col}")
        print("\nAvailable columns:")
        for col in df.columns:
            print(f"  - {col}")
        sys.exit(1)

    if validation["non_numeric_cols"]:
        print("WARNING: The following non-ID columns are not numeric and will be dropped:")
        for col in validation["non_numeric_cols"]:
            print(f"  - {col}")
        print(
            "\n  If any of these should be numeric, check your input data for formatting "
            "issues (e.g. commas in numbers, text values) and re-run.\n"
        )

    print("Running anomaly detection ...")
    try:
        result, report = run_detection(
            df=df,
            id_cols=args.id_cols,
            confidence=args.confidence,
            min_value_threshold=args.min_value,
        )
    except ValueError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    if report["warnings"]:
        print("\n--- Warnings ---")
        for w in report["warnings"]:
            print(f"  ! {w}")

    # Preprocessing summary
    print("\n--- Preprocessing ---")
    if report["dropped_sparse_cols"]:
        print(f"  Columns dropped (< 10% present):  {report['dropped_sparse_cols']}")
    if report["dropped_zero_var_cols"]:
        print(f"  Columns dropped (zero variance):   {report['dropped_zero_var_cols']}")
    if report["dropped_collinear_cols"]:
        print(f"  Columns dropped (collinear r>0.95):{report['dropped_collinear_cols']}")
    if report["dropped_sparse_rows"]:
        print(f"  Rows dropped (< 4 values present): {report['dropped_sparse_rows']}")
    print(f"  Metric columns used: {len(report['metric_cols_used'])}")
    print(f"  Rows analyzed:       {report['remaining_rows']}")

    # Detection summary
    print("\n--- Results ---")
    print(f"  Confidence level:      {report['confidence']:.0%}")
    print(f"  Chi-square cutoff:     {report['chi2_cutoff']:.3f}")
    print(f"  Outliers flagged:      {report['n_outliers']} of {report['remaining_rows']} rows")

    output_path = write_output(result, report, args.output)
    print(f"\nOutput written to: {output_path}")


if __name__ == "__main__":
    main()
