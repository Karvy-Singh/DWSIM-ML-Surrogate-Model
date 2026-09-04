#!/usr/bin/env python3

"""Predict DWSIM outputs with the saved model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--model", type=Path, default=Path("Results/final_surrogate.joblib"))
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = joblib.load(args.model)
    data = pd.read_csv(args.input_csv)
    features = bundle["features"]
    public_inputs = [feature for feature in features if feature != "feed_stage_fraction"]
    public_inputs.append("feed_stage")
    missing = sorted(set(public_inputs) - set(data.columns))
    if missing:
        raise ValueError(f"Missing input columns: {', '.join(missing)}")

    stages = data["number_of_stages"]
    feed_stage = data["feed_stage"]
    if (stages != stages.round()).any() or (feed_stage != feed_stage.round()).any():
        raise ValueError("number_of_stages and feed_stage must be integers")
    if ((feed_stage < 1) | (feed_stage > stages - 2)).any():
        raise ValueError("feed_stage must be between 1 and number_of_stages - 2")

    model_inputs = data.copy()
    model_inputs["feed_stage_fraction"] = feed_stage / (stages - 1)
    model_inputs = model_inputs[features]
    ranges = bundle["input_ranges"]
    outside = pd.DataFrame(
        {
            feature: (model_inputs[feature] < ranges.loc[feature, "min"])
            | (model_inputs[feature] > ranges.loc[feature, "max"])
            for feature in features
        }
    )
    if outside.any().any():
        details = []
        for row_index, row in outside[outside.any(axis=1)].iterrows():
            invalid_features = ", ".join(row.index[row])
            details.append(f"row {row_index}: {invalid_features}")
        raise ValueError(
            "Inputs are outside the ranges used to train the model: "
            + "; ".join(details[:10])
        )

    predicted = bundle["model"].predict(model_inputs)
    result = data.copy()
    for index, target in enumerate(bundle["targets"]):
        result[target] = predicted[:, index]
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
