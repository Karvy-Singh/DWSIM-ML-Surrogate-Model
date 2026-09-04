#!/usr/bin/env python3

"""Compare real DWSIM and surrogate predictions along two simple sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

START_DIRECTORY = Path.cwd()

from generate_dataset import automation, run_case


BASE_CASE = {
    "feed_temperature_K": 365.0,
    "feed_pressure_Pa": 150662.5,
    "benzene_feed_fraction": 0.5,
    "number_of_stages": 26,
    "reflux_ratio": 3.0,
    "bottoms_flow_mol_s": 60.0,
}

TARGET_LABELS = [
    "Distillate benzene purity",
    "Bottoms toluene purity",
    "Condenser duty (kW)",
    "Signed reboiler duty (kW)",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("analysis_results/final_surrogate.joblib"),
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("analysis_results")
    )
    return parser.parse_args()


def evaluate_case(bundle, case):
    stages = int(case["number_of_stages"])
    feed_stage = int(round(0.5 * (stages - 1)))
    model_inputs = pd.DataFrame(
        [
            {
                **case,
                "feed_stage_fraction": feed_stage / (stages - 1),
            }
        ]
    )[bundle["features"]]
    surrogate = bundle["model"].predict(model_inputs)[0]

    dwsim = run_case(
        temperature=case["feed_temperature_K"],
        pressure=case["feed_pressure_Pa"],
        benzene_fraction=case["benzene_feed_fraction"],
        stages=stages,
        feed_stage=feed_stage,
        reflux_ratio=case["reflux_ratio"],
        bottoms_flow=case["bottoms_flow_mol_s"],
    )
    dwsim_values = np.array([dwsim[target] for target in bundle["targets"]])
    return dwsim_values, surrogate


def run_sweep(bundle, variable, values):
    dwsim_rows = []
    surrogate_rows = []
    for value in values:
        case = BASE_CASE.copy()
        case[variable] = int(value) if variable == "number_of_stages" else float(value)
        dwsim, surrogate = evaluate_case(bundle, case)
        dwsim_rows.append(dwsim)
        surrogate_rows.append(surrogate)
        print(f"Solved {variable} = {value}")
    return np.asarray(dwsim_rows), np.asarray(surrogate_rows)


def save_plot(path, values, dwsim, surrogate, xlabel):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for index, ax in enumerate(axes.flat):
        ax.plot(values, dwsim[:, index], "o-", color="#1f4e79", label="DWSIM")
        ax.plot(
            values,
            surrogate[:, index],
            "s--",
            color="#c65d21",
            label="Surrogate",
        )
        ax.set_title(TARGET_LABELS[index])
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    model_path = args.model if args.model.is_absolute() else START_DIRECTORY / args.model
    results_dir = (
        args.results_dir
        if args.results_dir.is_absolute()
        else START_DIRECTORY / args.results_dir
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(model_path)

    try:
        reflux_values = np.linspace(1.5, 4.5, 13)
        reflux_dwsim, reflux_surrogate = run_sweep(
            bundle, "reflux_ratio", reflux_values
        )
        save_plot(
            results_dir / "dwsim_surrogate_reflux_sweep.png",
            reflux_values,
            reflux_dwsim,
            reflux_surrogate,
            "Reflux ratio",
        )

        stage_values = np.arange(18, 36)
        stage_dwsim, stage_surrogate = run_sweep(
            bundle, "number_of_stages", stage_values
        )
        save_plot(
            results_dir / "dwsim_surrogate_stage_sweep.png",
            stage_values,
            stage_dwsim,
            stage_surrogate,
            "Number of stages",
        )
    finally:
        automation.ReleaseResources()

    print(f"Sweep plots written to {results_dir}")


if __name__ == "__main__":
    main()
