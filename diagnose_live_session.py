"""Read-only comparison of live flow features with the fitted scaler baseline.

This tool reports model-output counts and standardized feature differences. It
does not label flows, recalculate predictions, or write any project files.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import joblib

from live_backend import FEATURE_MAP, SCALER_FILE


EXPECTED_SKLEARN_VERSION = "1.7.2"
DEFAULT_TOP_FEATURES = 10


def _load_single_csv(session_dir, pattern, description):
    matches = sorted(session_dir.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {description} matching {pattern!r} in "
            f"{session_dir}, found {len(matches)}."
        )
    return pd.read_csv(matches[0]), matches[0]


def analyze_session(flows, predictions, scaler, top_n=DEFAULT_TOP_FEATURES):
    """Return prediction counts and feature-drift summaries without mutation."""
    expected = list(getattr(scaler, "feature_names_in_", []))
    if len(expected) != 61:
        raise ValueError(
            f"Expected a scaler configured for 61 named features, found {len(expected)}."
        )

    missing = []
    mapped = []
    for feature in expected:
        column = FEATURE_MAP.get(feature)
        if column is None:
            missing.append(f"No live mapping for {feature}")
        elif column not in flows.columns:
            missing.append(f"Missing live column {column} for {feature}")
        else:
            mapped.append(column)
    if missing:
        raise ValueError("Cannot compare this session's features:\n" + "\n".join(missing))

    values = (
        flows[mapped]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .to_numpy(dtype=np.float64)
    )
    feature_frame = pd.DataFrame(values, columns=expected)
    standardized = scaler.transform(feature_frame)
    absolute_z = np.abs(standardized)

    paired_rows = min(len(flows), len(predictions))
    paired_labels = predictions.iloc[:paired_rows]["prediction"].astype(str).to_numpy()
    feature_rows = absolute_z[:paired_rows]

    drift = []
    for column_index, feature in enumerate(expected):
        overall = absolute_z[:, column_index]
        flagged = feature_rows[paired_labels == "ATTACK", column_index]
        benign = feature_rows[paired_labels == "BENIGN", column_index]
        drift.append(
            {
                "feature": feature,
                "median_abs_z": float(np.median(overall)) if len(overall) else 0.0,
                "flagged_median_abs_z": (
                    float(np.median(flagged)) if len(flagged) else None
                ),
                "benign_median_abs_z": (
                    float(np.median(benign)) if len(benign) else None
                ),
                "all_over_3z_pct": (
                    float(np.mean(overall > 3) * 100) if len(overall) else 0.0
                ),
                "flagged_over_3z_pct": (
                    float(np.mean(flagged > 3) * 100) if len(flagged) else None
                ),
                "benign_over_3z_pct": (
                    float(np.mean(benign > 3) * 100) if len(benign) else None
                ),
            }
        )
    drift.sort(key=lambda item: item["median_abs_z"], reverse=True)

    family_counts = {}
    if "attack_family" in predictions.columns:
        family_counts = {
            str(name): int(count)
            for name, count in predictions["attack_family"].value_counts().items()
        }

    probability_by_output = {}
    if (
        "attack_probability" in predictions.columns
        and "prediction" in predictions.columns
        and len(predictions)
    ):
        probabilities = pd.to_numeric(
            predictions["attack_probability"], errors="coerce"
        )
        for output in ("ATTACK", "BENIGN"):
            selected = probabilities[
                predictions["prediction"].astype(str).to_numpy() == output
            ].dropna()
            if len(selected):
                probability_by_output[output] = float(selected.median())

    return {
        "flow_rows": len(flows),
        "prediction_rows": len(predictions),
        "paired_rows": paired_rows,
        "prediction_counts": {
            str(name): int(count)
            for name, count in predictions["prediction"].astype(str).value_counts().items()
        } if "prediction" in predictions.columns else {},
        "family_counts": family_counts,
        "median_attack_probability_by_output": probability_by_output,
        "feature_drift": drift[:max(0, int(top_n))],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare one live session's model outputs and features to the scaler baseline."
    )
    parser.add_argument(
        "session_dir",
        type=Path,
        help="Session directory containing one live_flows_*.csv and one live_predictions_*.csv.",
    )
    parser.add_argument(
        "--scaler",
        type=Path,
        default=SCALER_FILE,
        help=f"Fitted scaler path (default: {SCALER_FILE}).",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=DEFAULT_TOP_FEATURES,
        help="Number of highest-shift features to display.",
    )
    args = parser.parse_args(argv)

    try:
        if not args.session_dir.is_dir():
            raise FileNotFoundError(f"Session directory not found: {args.session_dir}")
        if not args.scaler.is_file():
            raise FileNotFoundError(f"Scaler not found: {args.scaler}")

        flows, flow_path = _load_single_csv(
            args.session_dir, "live_flows_*.csv", "live flow CSV"
        )
        predictions, prediction_path = _load_single_csv(
            args.session_dir, "live_predictions_*.csv", "prediction CSV"
        )
        scaler = joblib.load(args.scaler)
        summary = analyze_session(flows, predictions, scaler, args.top_features)
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        parser.error(str(exc))

    print("AI-NIDS live-session diagnostic (read-only)")
    print(f"Flow file: {flow_path}")
    print(f"Prediction file: {prediction_path}")
    print(
        f"Rows: {summary['flow_rows']} flows, {summary['prediction_rows']} predictions, "
        f"{summary['paired_rows']} paired"
    )
    if summary["flow_rows"] != summary["prediction_rows"]:
        print("WARNING: flow and prediction row counts differ; inspect the session before drawing conclusions.")
    print(f"Model output counts: {summary['prediction_counts']}")
    print(f"Model family output counts: {summary['family_counts']}")
    if summary["median_attack_probability_by_output"]:
        probability_text = ", ".join(
            f"{label}={value:.4f}"
            for label, value in summary["median_attack_probability_by_output"].items()
        )
        print(f"Median attack probability by model output: {probability_text}")
    print(
        f"Python {sys.version.split()[0]} / scikit-learn {sklearn.__version__} "
        f"(scaler artifact version expected: {EXPECTED_SKLEARN_VERSION})"
    )
    if sklearn.__version__ != EXPECTED_SKLEARN_VERSION:
        print("WARNING: runtime scikit-learn does not match the saved scaler's version.")
    print("Top feature shifts (absolute standardized distance from scaler mean):")
    print(
        "Feature | All rows median | ATTACK median | BENIGN median | "
        "All rows > 3 SD | ATTACK > 3 SD | BENIGN > 3 SD"
    )
    for item in summary["feature_drift"]:
        flagged = item["flagged_median_abs_z"]
        benign = item["benign_median_abs_z"]
        flagged_pct = item["flagged_over_3z_pct"]
        print(
            f"{item['feature']} | {item['median_abs_z']:.2f} | "
            f"{flagged if flagged is not None else float('nan'):.2f} | "
            f"{benign if benign is not None else float('nan'):.2f} | "
            f"{item['all_over_3z_pct']:.1f}% | "
            f"{flagged_pct if flagged_pct is not None else float('nan'):.1f}% | "
            f"{item['benign_over_3z_pct'] if item['benign_over_3z_pct'] is not None else float('nan'):.1f}%"
        )
    print(
        "These are model outputs and feature-shift indicators, not ground-truth labels, "
        "accuracy, or proof that a flagged flow is malicious."
    )


if __name__ == "__main__":
    main()
