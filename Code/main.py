#!/usr/bin/env python3

"""Main workflow: compare models, check the winner, and save the results."""

from __future__ import annotations

import argparse
import json
import platform
import warnings
from math import ceil
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from physics_surrogate import (
    FEATURES,
    INPUT_LIMITS,
    TARGETS,
    PhysicsConstrainedSurrogate,
    PhysicsProjectedSurrogate,
    component_balance_residual,
)


RANDOM_STATE = 20260817
CONFIRMATION_FRACTION = 0.20
CALIBRATION_ROWS = 1_000


def scaled_regressor(regressor):
    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=StandardScaler(),
    )


def make_mlp(
    hidden_layers,
    activation="relu",
    alpha=1e-4,
    batch_size=64,
    max_iter=600,
):
    return scaled_regressor(
        Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=hidden_layers,
                        activation=activation,
                        alpha=alpha,
                        batch_size=batch_size,
                        learning_rate_init=1e-3,
                        max_iter=max_iter,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=30,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    )


def candidate_factories(quick=False):
    trees = 150 if quick else 500
    boosting_iterations = 150 if quick else 500
    max_iter = 250 if quick else 600
    return {
        "Polynomial Ridge": lambda: PhysicsProjectedSurrogate(
            scaled_regressor(
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("polynomial", PolynomialFeatures(2, include_bias=False)),
                        ("ridge", Ridge(alpha=1e-3)),
                    ]
                )
            )
        ),
        "Extra Trees": lambda: PhysicsProjectedSurrogate(
            scaled_regressor(
                ExtraTreesRegressor(
                    n_estimators=trees,
                    max_features=1.0,
                    min_samples_leaf=1,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            )
        ),
        "CatBoost": lambda: PhysicsProjectedSurrogate(
            MultiOutputRegressor(
                CatBoostRegressor(
                    iterations=boosting_iterations,
                    depth=7,
                    learning_rate=0.05,
                    loss_function="RMSE",
                    random_seed=RANDOM_STATE,
                    verbose=False,
                    thread_count=-1,
                    allow_writing_files=False,
                ),
                n_jobs=1,
            )
        ),
        "ANN 64-32 ReLU": lambda: PhysicsProjectedSurrogate(
            make_mlp((64, 32), max_iter=max_iter)
        ),
        "ANN 128-64 ReLU": lambda: PhysicsProjectedSurrogate(
            make_mlp((128, 64), max_iter=max_iter)
        ),
        "ANN 128-64 tanh": lambda: PhysicsProjectedSurrogate(
            make_mlp((128, 64), activation="tanh", max_iter=max_iter)
        ),
        "ANN 256-128 ReLU": lambda: PhysicsProjectedSurrogate(
            make_mlp((256, 128), batch_size=128, max_iter=max_iter)
        ),
        "ANN latent split": lambda: PhysicsConstrainedSurrogate(
            make_mlp((64, 32), max_iter=max_iter)
        ),
    }


def prediction_metrics(model_name, split_name, actual, predicted, normalizers):
    records = []
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    for index, target in enumerate(TARGETS):
        errors = np.abs(actual[:, index] - predicted[:, index])
        rmse = np.sqrt(mean_squared_error(actual[:, index], predicted[:, index]))
        records.append(
            {
                "model": model_name,
                "split": split_name,
                "target": target,
                "MAE": mean_absolute_error(actual[:, index], predicted[:, index]),
                "RMSE": rmse,
                "R2": r2_score(actual[:, index], predicted[:, index]),
                "NRMSE": rmse / normalizers[index],
                "P95_absolute_error": np.quantile(errors, 0.95),
            }
        )
    return records


def cross_validate_candidates(factories, features, targets, normalizers, folds):
    splitter = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    records = []
    for model_name, factory in factories.items():
        print(f"Cross-validating {model_name}...")
        for fold, (train_positions, validation_positions) in enumerate(
            splitter.split(features), start=1
        ):
            model = factory()
            start = perf_counter()
            model.fit(features.iloc[train_positions], targets.iloc[train_positions])
            elapsed = perf_counter() - start
            predicted = model.predict(features.iloc[validation_positions])
            fold_records = prediction_metrics(
                model_name,
                f"cv_fold_{fold}",
                targets.iloc[validation_positions],
                predicted,
                normalizers,
            )
            for record in fold_records:
                record["fold"] = fold
                record["fit_seconds"] = elapsed
            records.extend(fold_records)
    return pd.DataFrame(records)


def summarize_cross_validation(metrics):
    fold_scores = (
        metrics.groupby(["model", "fold"], as_index=False)
        .agg(mean_NRMSE=("NRMSE", "mean"), mean_R2=("R2", "mean"))
    )
    summary = (
        fold_scores.groupby("model", as_index=False)
        .agg(
            mean_NRMSE=("mean_NRMSE", "mean"),
            std_NRMSE=("mean_NRMSE", "std"),
            mean_R2=("mean_R2", "mean"),
        )
        .sort_values("mean_NRMSE")
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary


def physical_checks(name, features, predictions):
    predictions = np.asarray(predictions)
    residual = component_balance_residual(features, predictions)
    benzene_feed = 100.0 * features["benzene_feed_fraction"].to_numpy()
    invalid = (predictions[:, :2] < 0.0) | (predictions[:, :2] > 1.0)
    return {
        "model": name,
        "invalid_distillate_purities": int(invalid[:, 0].sum()),
        "invalid_bottoms_purities": int(invalid[:, 1].sum()),
        "max_abs_balance_residual_mol_s": float(np.max(np.abs(residual))),
        "p95_relative_balance_error": float(
            np.quantile(np.abs(residual) / benzene_feed, 0.95)
        ),
    }


def non_saturated_purity_metrics(actual, predicted, threshold=0.995):
    """Score each purity only where its DWSIM value is below the threshold."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    records = []
    for index, target in enumerate(TARGETS[:2]):
        mask = actual[:, index] < threshold
        records.append(
            {
                "target": target,
                "condition": f"DWSIM purity < {threshold}",
                "rows": int(mask.sum()),
                "MAE": mean_absolute_error(actual[mask, index], predicted[mask, index]),
                "RMSE": np.sqrt(
                    mean_squared_error(actual[mask, index], predicted[mask, index])
                ),
                "R2": r2_score(actual[mask, index], predicted[mask, index]),
            }
        )
    return pd.DataFrame(records)


def projection_checks(model, features):
    if not hasattr(model, "predict_raw"):
        return {
            "raw_invalid_distillate_purities": 0,
            "raw_invalid_bottoms_purities": 0,
            "adjusted_prediction_percent": 0.0,
            "mean_abs_distillate_adjustment": 0.0,
            "mean_abs_bottoms_adjustment": 0.0,
            "p95_abs_distillate_adjustment": 0.0,
            "p95_abs_bottoms_adjustment": 0.0,
        }

    raw = model.predict_raw(features)
    projected = model.predict(features)
    invalid = (raw[:, :2] < 0.0) | (raw[:, :2] > 1.0)
    adjustments = np.abs(projected[:, :2] - raw[:, :2])
    return {
        "raw_invalid_distillate_purities": int(invalid[:, 0].sum()),
        "raw_invalid_bottoms_purities": int(invalid[:, 1].sum()),
        "adjusted_prediction_percent": float(
            100.0 * np.any(adjustments > 1e-12, axis=1).mean()
        ),
        "mean_abs_distillate_adjustment": float(adjustments[:, 0].mean()),
        "mean_abs_bottoms_adjustment": float(adjustments[:, 1].mean()),
        "p95_abs_distillate_adjustment": float(
            np.quantile(adjustments[:, 0], 0.95)
        ),
        "p95_abs_bottoms_adjustment": float(
            np.quantile(adjustments[:, 1], 0.95)
        ),
    }


def conformal_intervals(actual_calibration, predicted_calibration, predicted_test):
    alpha = 0.10
    residuals = np.abs(
        np.asarray(actual_calibration, dtype=float)
        - np.asarray(predicted_calibration, dtype=float)
    )
    level = min(1.0, ceil((len(residuals) + 1) * (1.0 - alpha)) / len(residuals))
    quantiles = np.quantile(residuals, level, axis=0, method="higher")
    lower = predicted_test - quantiles
    upper = predicted_test + quantiles
    lower[:, :2] = np.maximum(lower[:, :2], 0.0)
    upper[:, :2] = np.minimum(upper[:, :2], 1.0)
    return lower, upper, quantiles


def run_region_holdouts(factory, data, normalizers):
    holdouts = {
        "high feed composition": "benzene_feed_fraction",
        "high reflux ratio": "reflux_ratio",
        "high stage count": "number_of_stages",
        "high bottoms flow": "bottoms_flow_mol_s",
    }
    records = []
    for name, feature in holdouts.items():
        cutoff = data[feature].quantile(0.80)
        test_mask = data[feature] >= cutoff
        model = factory()
        model.fit(data.loc[~test_mask, FEATURES], data.loc[~test_mask, TARGETS])
        predicted = model.predict(data.loc[test_mask, FEATURES])
        scenario_records = prediction_metrics(
            "selected model",
            name,
            data.loc[test_mask, TARGETS],
            predicted,
            normalizers,
        )
        for record in scenario_records:
            record["holdout_feature"] = feature
            record["cutoff"] = cutoff
            record["training_rows"] = int((~test_mask).sum())
            record["test_rows"] = int(test_mask.sum())
        records.extend(scenario_records)
    return pd.DataFrame(records)


def monotonic_trend_tests(model, reference_features, feature_ranges):
    expectations = [
        ("reflux_ratio", 0, 1, "distillate purity increases"),
        ("reflux_ratio", 1, 1, "bottoms purity increases"),
        ("reflux_ratio", 2, 1, "condenser duty increases"),
        ("reflux_ratio", 3, -1, "signed reboiler duty decreases"),
        ("number_of_stages", 0, 1, "distillate purity increases"),
        ("number_of_stages", 1, 1, "bottoms purity increases"),
    ]
    records = []
    for feature, target_index, direction, expectation in expectations:
        lower, upper = feature_ranges.loc[feature, ["min", "max"]]
        if feature == "number_of_stages":
            grid = np.arange(int(lower), int(upper) + 1)
        else:
            grid = np.linspace(lower, upper, 31)

        violating_steps = 0
        significant_violating_steps = 0
        total_steps = 0
        violating_sweeps = 0
        significant_violating_sweeps = 0
        largest_opposite_change = 0.0
        for _, reference in reference_features.iterrows():
            sweep = pd.DataFrame(
                np.repeat(reference.to_numpy()[None, :], len(grid), axis=0),
                columns=FEATURES,
            )
            sweep[feature] = grid
            values = model.predict(sweep)[:, target_index]
            directed_changes = direction * np.diff(values)
            tolerance = 1e-5 if target_index < 2 else 1.0
            significant_tolerance = 1e-3 if target_index < 2 else 10.0
            violations = directed_changes < -tolerance
            significant_violations = directed_changes < -significant_tolerance
            violating_steps += int(violations.sum())
            significant_violating_steps += int(significant_violations.sum())
            total_steps += len(violations)
            violating_sweeps += int(violations.any())
            significant_violating_sweeps += int(significant_violations.any())
            largest_opposite_change = max(
                largest_opposite_change,
                float(np.maximum(-directed_changes, 0.0).max()),
            )
        records.append(
            {
                "feature": feature,
                "target": TARGETS[target_index],
                "expected_trend": expectation,
                "step_violation_percent": 100.0 * violating_steps / total_steps,
                "sweeps_with_violation_percent": (
                    100.0 * violating_sweeps / len(reference_features)
                ),
                "significant_step_violation_percent": (
                    100.0 * significant_violating_steps / total_steps
                ),
                "sweeps_with_significant_violation_percent": (
                    100.0 * significant_violating_sweeps / len(reference_features)
                ),
                "largest_opposite_step": largest_opposite_change,
            }
        )
    return pd.DataFrame(records)


def analyze_failures(successful, failed):
    common_features = [
        "feed_temperature_K",
        "feed_pressure_Pa",
        "benzene_feed_fraction",
        "number_of_stages",
        "feed_stage",
        "reflux_ratio",
        "bottoms_flow_mol_s",
    ]
    good = successful[common_features].copy()
    good["failed"] = 0
    bad = failed[common_features].copy()
    bad["failed"] = 1
    attempts = pd.concat([good, bad], ignore_index=True)

    records = []
    for feature in common_features:
        bins = pd.qcut(attempts[feature], q=5, duplicates="drop")
        grouped = attempts.assign(bin=bins).groupby("bin", observed=True)
        for interval, group in grouped:
            records.append(
                {
                    "feature": feature,
                    "bin": str(interval),
                    "attempts": len(group),
                    "failures": int(group["failed"].sum()),
                    "failure_percent": 100.0 * group["failed"].mean(),
                }
            )
    return pd.DataFrame(records)


def save_plots(results_dir, cv_summary, metrics, actual, predicted, region_metrics):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ordered = cv_summary.sort_values("mean_NRMSE", ascending=False)
    ax.barh(ordered["model"], ordered["mean_NRMSE"], color="#315b7d")
    ax.set_xlabel("Five-fold mean NRMSE")
    ax.set_title("Development-set model selection")
    fig.tight_layout()
    fig.savefig(results_dir / "cross_validation_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for index, ax in enumerate(axes.flat):
        low = min(actual[:, index].min(), predicted[:, index].min())
        high = max(actual[:, index].max(), predicted[:, index].max())
        ax.scatter(actual[:, index], predicted[:, index], s=10, alpha=0.45)
        ax.plot([low, high], [low, high], "--", color="#b24c32")
        target_metrics = metrics[metrics["target"] == TARGETS[index]].iloc[0]
        ax.set_title(f"{TARGETS[index]}\nR2={target_metrics['R2']:.5f}")
        ax.set_xlabel("DWSIM")
        ax.set_ylabel("Surrogate")
    fig.tight_layout()
    fig.savefig(results_dir / "confirmation_predicted_vs_actual.png", dpi=180)
    plt.close(fig)

    aggregate_regions = (
        region_metrics.groupby("split", as_index=False)["NRMSE"].mean()
        .sort_values("NRMSE", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(aggregate_regions["split"], aggregate_regions["NRMSE"], color="#9b6a3d")
    ax.set_xlabel("Mean NRMSE")
    ax.set_title("Edge-region extrapolation stress tests")
    fig.tight_layout()
    fig.savefig(results_dir / "region_holdout_comparison.png", dpi=180)
    plt.close(fig)


def markdown_table(frame):
    """Render a small DataFrame without requiring pandas' tabulate extra."""

    formatted = frame.copy()
    for column in formatted.select_dtypes(include=[np.number]).columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.6g}")
    headers = [str(column) for column in formatted.columns]
    rows = [[str(value) for value in row] for row in formatted.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_summary(
    path,
    selected_name,
    cv_summary,
    confirmation_metrics,
    non_saturated_metrics,
    checks,
    uncertainty,
    region_metrics,
    trend_tests,
    projection,
    data,
    failed,
):
    aggregate_confirmation = confirmation_metrics.groupby("model").agg(
        mean_NRMSE=("NRMSE", "mean"), mean_R2=("R2", "mean")
    )
    selected_aggregate = aggregate_confirmation.loc[selected_name]
    selected_checks = checks[checks["model"] == selected_name].iloc[0]
    region_aggregate = region_metrics.groupby("split")["NRMSE"].mean()
    lines = [
        "# Results Summary",
        "",
        f"- Selected model: {selected_name}",
        f"- Selection rule: lowest development-set cross-validation mean NRMSE ({cv_summary.iloc[0]['mean_NRMSE']:.6f} +/- {cv_summary.iloc[0]['std_NRMSE']:.6f})",
        f"- Locked confirmation mean NRMSE: {selected_aggregate['mean_NRMSE']:.6f}",
        f"- Locked confirmation mean R2: {selected_aggregate['mean_R2']:.6f}",
        f"- Invalid purity predictions: {int(selected_checks['invalid_distillate_purities'] + selected_checks['invalid_bottoms_purities'])}",
        f"- Maximum benzene-balance residual: {selected_checks['max_abs_balance_residual_mol_s']:.3e} mol/s",
        f"- Mean projection adjustment (distillate/bottoms purity): {projection['mean_abs_distillate_adjustment']:.3e} / {projection['mean_abs_bottoms_adjustment']:.3e}",
        f"- Dataset: {len(data):,} converged simulations and {len(failed):,} failed attempts",
        "",
        "## Confirmation Metrics",
        "",
        markdown_table(
            confirmation_metrics[confirmation_metrics["model"] == selected_name]
            .drop(columns=["split"])
        ),
        "",
        "## Non-Saturated Purity Metrics",
        "",
        "Each purity is scored only on confirmation rows where its corresponding DWSIM value is below 0.995.",
        "",
        markdown_table(non_saturated_metrics),
        "",
        "## Model Selection",
        "",
        markdown_table(cv_summary),
        "",
        "## 90% Prediction Ranges",
        "",
        markdown_table(uncertainty),
        "",
        "## Tests on Unseen Edge Regions",
        "",
        markdown_table(region_aggregate.rename("mean_NRMSE").reset_index()),
        "",
        "## Physical Trend Tests",
        "",
        markdown_table(trend_tests),
        "",
        "## Scope Warning",
        "",
        "The confirmation score covers cases similar to the training data. The edge-region "
        "tests are harder because each model is trained without one end of an input range. "
        "Do not use the model outside the sampled ranges or claim plant accuracy.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset.csv"))
    parser.add_argument("--failed", type=Path, default=Path("failed_cases.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("analysis_results"))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use three folds and smaller ensembles for a fast smoke test",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    data = pd.read_csv(args.dataset)
    failed = pd.read_csv(args.failed)
    required = FEATURES + TARGETS
    if data[required].isna().any().any() or not np.isfinite(data[required]).all().all():
        raise ValueError("Dataset contains missing or non-finite model values")
    if data.duplicated(subset=FEATURES).any():
        raise ValueError("Dataset contains duplicate model inputs")

    development, confirmation = train_test_split(
        data,
        test_size=CONFIRMATION_FRACTION,
        random_state=RANDOM_STATE,
    )
    training, calibration = train_test_split(
        development,
        test_size=CALIBRATION_ROWS,
        random_state=RANDOM_STATE + 1,
    )
    normalizers = np.ptp(training[TARGETS].to_numpy(), axis=0)
    if np.any(normalizers <= 0):
        raise ValueError("At least one target has zero training range")

    split_assignments = pd.DataFrame(
        {
            "row_index": np.concatenate(
                [training.index, calibration.index, confirmation.index]
            ),
            "split": (
                ["training"] * len(training)
                + ["calibration"] * len(calibration)
                + ["confirmation"] * len(confirmation)
            ),
        }
    ).sort_values("row_index")
    split_assignments.to_csv(args.results_dir / "split_assignments.csv", index=False)

    factories = candidate_factories(args.quick)
    folds = 3 if args.quick else 5
    cv_metrics = cross_validate_candidates(
        factories,
        training[FEATURES],
        training[TARGETS],
        normalizers,
        folds,
    )
    cv_summary = summarize_cross_validation(cv_metrics)
    selected_name = cv_summary.iloc[0]["model"]
    selected_factory = factories[selected_name]
    print(f"Selected by cross-validation: {selected_name}")

    selected_model = selected_factory()
    selected_model.fit(training[FEATURES], training[TARGETS])
    calibration_predictions = selected_model.predict(calibration[FEATURES])
    confirmation_predictions = selected_model.predict(confirmation[FEATURES])

    direct_ann = make_mlp((128, 64), max_iter=250 if args.quick else 600)
    direct_ann.fit(training[FEATURES], training[TARGETS])
    direct_predictions = direct_ann.predict(confirmation[FEATURES])

    confirmation_records = prediction_metrics(
        selected_name,
        "locked confirmation",
        confirmation[TARGETS],
        confirmation_predictions,
        normalizers,
    )
    confirmation_records.extend(
        prediction_metrics(
            "Unconstrained ANN",
            "locked confirmation",
            confirmation[TARGETS],
            direct_predictions,
            normalizers,
        )
    )
    confirmation_metrics = pd.DataFrame(confirmation_records)
    non_saturated_metrics = non_saturated_purity_metrics(
        confirmation[TARGETS], confirmation_predictions
    )

    checks = pd.DataFrame(
        [
            physical_checks(selected_name, confirmation[FEATURES], confirmation_predictions),
            physical_checks(
                "Unconstrained ANN", confirmation[FEATURES], direct_predictions
            ),
        ]
    )
    projection = projection_checks(selected_model, confirmation[FEATURES])

    lower, upper, interval_quantiles = conformal_intervals(
        calibration[TARGETS],
        calibration_predictions,
        confirmation_predictions,
    )
    actual_confirmation = confirmation[TARGETS].to_numpy()
    uncertainty_records = []
    for index, target in enumerate(TARGETS):
        covered = (actual_confirmation[:, index] >= lower[:, index]) & (
            actual_confirmation[:, index] <= upper[:, index]
        )
        uncertainty_records.append(
            {
                "target": target,
                "nominal_coverage": 0.90,
                "empirical_coverage": covered.mean(),
                "calibrated_absolute_error": interval_quantiles[index],
                "mean_interval_width": np.mean(upper[:, index] - lower[:, index]),
            }
        )
    uncertainty = pd.DataFrame(uncertainty_records)

    region_metrics = run_region_holdouts(selected_factory, data, normalizers)
    references = confirmation[FEATURES].sample(
        n=min(100, len(confirmation)), random_state=RANDOM_STATE
    )
    feature_ranges = training[FEATURES].agg(["min", "max"]).T
    input_limits = pd.DataFrame.from_dict(
        INPUT_LIMITS, orient="index", columns=["min", "max"]
    )
    trend_tests = monotonic_trend_tests(selected_model, references, feature_ranges)
    failure_rates = analyze_failures(data, failed)

    sample_positions = np.linspace(0, len(confirmation) - 1, 10, dtype=int)
    samples = confirmation.iloc[sample_positions][FEATURES + TARGETS].reset_index()
    for index, target in enumerate(TARGETS):
        samples[f"predicted_{target}"] = confirmation_predictions[sample_positions, index]
        samples[f"lower90_{target}"] = lower[sample_positions, index]
        samples[f"upper90_{target}"] = upper[sample_positions, index]

    cv_metrics.to_csv(args.results_dir / "cv_fold_metrics.csv", index=False)
    cv_summary.to_csv(args.results_dir / "cv_summary.csv", index=False)
    confirmation_metrics.to_csv(
        args.results_dir / "confirmation_metrics.csv", index=False
    )
    checks.to_csv(args.results_dir / "physical_checks.csv", index=False)
    pd.DataFrame([projection]).to_csv(
        args.results_dir / "projection_checks.csv", index=False
    )
    uncertainty.to_csv(args.results_dir / "uncertainty_intervals.csv", index=False)
    region_metrics.to_csv(args.results_dir / "region_holdout_metrics.csv", index=False)
    trend_tests.to_csv(args.results_dir / "trend_tests.csv", index=False)
    failure_rates.to_csv(args.results_dir / "failure_rates_by_region.csv", index=False)
    samples.to_csv(args.results_dir / "sample_predictions.csv", index=False)

    bundle = {
        "model": selected_model,
        "model_name": selected_name,
        "features": FEATURES,
        "targets": TARGETS,
        "input_ranges": input_limits,
        "training_input_ranges": feature_ranges,
        "training_rows": len(training),
        "random_state": RANDOM_STATE,
    }
    joblib.dump(bundle, args.results_dir / "final_surrogate.joblib")

    metadata = {
        "random_state": RANDOM_STATE,
        "cv_folds": folds,
        "training_rows": len(training),
        "calibration_rows": len(calibration),
        "confirmation_rows": len(confirmation),
        "selected_model": selected_name,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (args.results_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="ascii"
    )

    selected_metrics = confirmation_metrics[
        confirmation_metrics["model"] == selected_name
    ]
    save_plots(
        args.results_dir,
        cv_summary,
        selected_metrics,
        actual_confirmation,
        confirmation_predictions,
        region_metrics,
    )
    write_summary(
        Path("Results_Summary_generated.md"),
        selected_name,
        cv_summary,
        confirmation_metrics,
        non_saturated_metrics,
        checks,
        uncertainty,
        region_metrics,
        trend_tests,
        projection,
        data,
        failed,
    )

    print()
    print(cv_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print(selected_metrics.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"Results written to {args.results_dir}")


if __name__ == "__main__":
    main()
