#!/usr/bin/env python3

import os
import sys
import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import qmc

DWSIM_LAUNCHER = "/usr/bin/dwsim"

TARGET_ROWS = int(os.environ.get("TARGET_ROWS", "10000"))
MAX_CONSECUTIVE_FAILURES = int(os.environ.get("MAX_CONSECUTIVE_FAILURES", "25"))
MAX_COLUMN_ITERATIONS = int(os.environ.get("MAX_COLUMN_ITERATIONS", "300"))
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", os.getcwd() + "/dataset.csv")
FAILED_FILE = os.environ.get("FAILED_FILE", os.getcwd() + "/failed_cases.csv")
METADATA_FILE = os.environ.get("METADATA_FILE", os.getcwd() + "/dataset_metadata.json")
REPRESENTATIVE_FLOWSHEET = os.environ.get(
    "REPRESENTATIVE_FLOWSHEET", os.getcwd() + "/DWSIM_Flowsheet.dwxmz"
)

FEED_MOLAR_FLOW = 100.0  # mol/s
COLUMN_PRESSURE = 101325.0  # Pa

RANDOM_SEED = 42


def find_dwsim_directory():

    # Environment variable takes priority
    env_dir = os.environ.get("DWSIM_DIR")

    if env_dir:
        p = Path(env_dir)

        if (p / "DWSIM.Automation.dll").exists():
            return p

    launcher = Path(DWSIM_LAUNCHER)

    candidates = []

    if launcher.exists():

        try:
            resolved = launcher.resolve()
            candidates.append(resolved.parent)
        except Exception:
            pass

    # Common Linux locations
    candidates += [
        Path("/usr/lib/dwsim"),
        Path("/usr/lib/DWSIM"),
        Path("/usr/share/dwsim"),
        Path("/usr/share/DWSIM"),
        Path("/opt/dwsim"),
        Path("/opt/DWSIM"),
    ]

    for path in candidates:

        if (path / "DWSIM.Automation.dll").exists():
            return path

    # Last resort: ask Linux 'find'
    print("Searching for DWSIM.Automation.dll ...")

    result = subprocess.run(
        ["find", "/usr", "/opt", "-type", "f", "-name", "DWSIM.Automation.dll"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    locations = [Path(x.strip()) for x in result.stdout.splitlines() if x.strip()]

    if locations:
        return locations[0].parent

    raise RuntimeError(
        "Could not locate DWSIM.Automation.dll.\n"
        "Run:\n"
        "find /usr /opt -name DWSIM.Automation.dll 2>/dev/null"
    )


DWSIM_DIR = find_dwsim_directory()

print("DWSIM directory:", DWSIM_DIR)


# Must be done BEFORE import clr
from pythonnet import load

load("coreclr")

import clr

# Make DWSIM dependencies visible
sys.path.insert(0, str(DWSIM_DIR))

os.environ["PATH"] = str(DWSIM_DIR) + os.pathsep + os.environ.get("PATH", "")


assemblies = [
    "DWSIM.Interfaces.dll",
    "DWSIM.GlobalSettings.dll",
    "DWSIM.SharedClasses.dll",
    "DWSIM.Thermodynamics.dll",
    "DWSIM.UnitOperations.dll",
    "DWSIM.FlowsheetSolver.dll",
    "DWSIM.Automation.dll",
]


for dll in assemblies:

    path = DWSIM_DIR / dll

    if path.exists():

        print("Loading:", dll)

        clr.AddReference(str(path))


from System import Array, Double
from DWSIM.Automation import Automation3

automation = Automation3()
DWSIM_VERSION = str(automation.GetVersion())

print("DWSIM Automation loaded.")
print("DWSIM version:", DWSIM_VERSION)


def run_case(
    temperature,
    pressure,
    benzene_fraction,
    stages,
    feed_stage,
    reflux_ratio,
    bottoms_flow,
    save_path=None,
):

    flowsheet = None

    try:

        flowsheet = automation.CreateFlowsheet()
        flowsheet.AddCompound("Benzene")
        flowsheet.AddCompound("Toluene")

        # Name must be one returned by DWSIM's available
        # property-package collection.
        try:

            pp = flowsheet.CreateAndAddPropertyPackage("Peng-Robinson (PR)")

        except Exception:

            # Some DWSIM builds expose it under this shorter name.
            pp = flowsheet.CreateAndAddPropertyPackage("Peng-Robinson")

        feed = flowsheet.AddFlowsheetObject("Material Stream", "FEED").GetAsObject()

        distillate = flowsheet.AddFlowsheetObject(
            "Material Stream", "DISTILLATE"
        ).GetAsObject()

        bottoms = flowsheet.AddFlowsheetObject(
            "Material Stream", "BOTTOMS"
        ).GetAsObject()

        column = flowsheet.AddFlowsheetObject(
            "Distillation Column", "COLUMN"
        ).GetAsObject()

        feed.SetPropertyPackageInstance(pp)
        distillate.SetPropertyPackageInstance(pp)
        bottoms.SetPropertyPackageInstance(pp)
        column.SetPropertyPackageInstance(pp)

        feed.SetTemperature(float(temperature))

        feed.SetPressure(float(pressure))

        feed.SetMolarFlow(float(FEED_MOLAR_FLOW))

        composition = Array[Double](
            [
                float(benzene_fraction),
                float(1.0 - benzene_fraction),
            ]
        )

        feed.SetOverallMolarComposition(composition)

        column.SetNumberOfStages(int(stages))

        column.SetTopPressure(float(COLUMN_PRESSURE))

        column.MaxIterations = MAX_COLUMN_ITERATIONS

        # DWSIM does not initialize the remaining stage pressures from the
        # top pressure until the column pressure drop has been specified.
        column.ColumnPressureDrop = 0.0

        # DWSIM counts condenser as stage index 0.
        column.ConnectFeed(feed, int(feed_stage))

        column.ConnectDistillate(distillate)

        column.ConnectBottoms(bottoms)

        column.SetCondenserSpec("Reflux Ratio", float(reflux_ratio), "")
        column.SetReboilerSpec("Product Molar Flow Rate", float(bottoms_flow), "mol/s")
        errors = automation.CalculateFlowsheet4(flowsheet)

        if errors is not None:

            errors_list = [str(e) for e in errors if str(e).strip()]

            if errors_list:

                raise RuntimeError(" | ".join(errors_list))

        xD = list(distillate.GetOverallComposition())
        xB = list(bottoms.GetOverallComposition())

        # Component order:
        # 0 = Benzene
        # 1 = Toluene

        distillate_benzene = float(xD[0])

        bottoms_toluene = float(xB[1])

        condenser_duty = float(column.CondenserDuty)
        reboiler_duty = float(column.ReboilerDuty)
        values = [distillate_benzene, bottoms_toluene, condenser_duty, reboiler_duty]

        if not all(math.isfinite(x) for x in values):

            raise ValueError("Non-finite simulation result")

        if not (0 <= distillate_benzene <= 1):

            raise ValueError("Invalid distillate purity")

        if not (0 <= bottoms_toluene <= 1):

            raise ValueError("Invalid bottoms purity")

        if save_path:
            automation.SaveFlowsheet2(flowsheet, str(save_path))

        return {
            "distillate_benzene_purity": distillate_benzene,
            "bottoms_toluene_purity": bottoms_toluene,
            "condenser_duty": condenser_duty,
            "reboiler_duty": reboiler_duty,
        }

    finally:

        # Important when doing thousands of simulations
        if flowsheet is not None:

            try:
                flowsheet.Dispose()
            except Exception:
                pass


def make_samples(n, seed):

    sampler = qmc.LatinHypercube(d=7, seed=seed)

    X = sampler.random(n)

    for x in X:

        temperature = 345 + x[0] * (385 - 345)
        pressure = 101325 + x[1] * (200000 - 101325)
        benzene = 0.25 + x[2] * (0.75 - 0.25)
        stages = int(round(18 + x[3] * (35 - 18)))
        feed_fraction = 0.30 + x[4] * (0.70 - 0.30)
        feed_stage = int(round(feed_fraction * (stages - 1)))
        feed_stage = max(1, min(feed_stage, stages - 2))
        reflux = 1.5 + x[5] * (4.5 - 1.5)
        bottoms_flow = 25 + x[6] * (75 - 25)

        yield {
            "feed_temperature_K": temperature,
            "feed_pressure_Pa": pressure,
            "benzene_feed_fraction": benzene,
            "number_of_stages": stages,
            "feed_stage": feed_stage,
            "feed_stage_fraction": feed_stage / (stages - 1),
            "reflux_ratio": reflux,
            "bottoms_flow_mol_s": bottoms_flow,
        }


HEADERS = [
    "feed_temperature_K",
    "feed_pressure_Pa",
    "benzene_feed_fraction",
    "number_of_stages",
    "feed_stage",
    "feed_stage_fraction",
    "reflux_ratio",
    "bottoms_flow_mol_s",
    "distillate_benzene_purity",
    "bottoms_toluene_purity",
    "condenser_duty",
    "reboiler_duty",
]


def write_metadata(successful, failed):
    metadata = {
        "mixture": ["Benzene", "Toluene"],
        "property_package": "Peng-Robinson (PR)",
        "dwsim_version": DWSIM_VERSION,
        "random_seed": RANDOM_SEED,
        "successful_rows": successful,
        "failed_attempts": failed,
        "sampling": (
            "Independent 300-point Latin hypercube batches; failed simulations "
            "were logged and replaced until the target row count was reached"
        ),
        "fixed_conditions": {
            "feed_molar_flow_mol_s": FEED_MOLAR_FLOW,
            "column_top_pressure_Pa": COLUMN_PRESSURE,
            "column_pressure_drop_Pa": 0.0,
        },
        "input_ranges": {
            "feed_temperature_K": [345.0, 385.0],
            "feed_pressure_Pa": [101325.0, 200000.0],
            "benzene_feed_fraction": [0.25, 0.75],
            "number_of_stages": [18, 35],
            "sampled_feed_stage_fraction": [0.30, 0.70],
            "realized_feed_stage_fraction": [5.0 / 18.0, 13.0 / 18.0],
            "reflux_ratio": [1.5, 4.5],
            "bottoms_flow_mol_s": [25.0, 75.0],
        },
        "outputs": {
            "distillate_benzene_purity": "mole fraction",
            "bottoms_toluene_purity": "mole fraction",
            "condenser_duty": "kW, signed DWSIM value",
            "reboiler_duty": "kW, signed DWSIM value",
        },
    }
    Path(METADATA_FILE).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="ascii"
    )


def main():
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()

    with open(FAILED_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "feed_temperature_K",
                "feed_pressure_Pa",
                "benzene_feed_fraction",
                "number_of_stages",
                "feed_stage",
                "reflux_ratio",
                "bottoms_flow_mol_s",
                "error",
            ]
        )

    successful = 0
    failed = 0
    batch_number = 0
    consecutive_failures = 0

    try:
        while successful < TARGET_ROWS:
            batch_number += 1
            samples = make_samples(300, RANDOM_SEED + batch_number)

            for inputs in samples:
                if successful >= TARGET_ROWS:
                    break

                try:
                    result = run_case(
                        temperature=inputs["feed_temperature_K"],
                        pressure=inputs["feed_pressure_Pa"],
                        benzene_fraction=inputs["benzene_feed_fraction"],
                        stages=inputs["number_of_stages"],
                        feed_stage=inputs["feed_stage"],
                        reflux_ratio=inputs["reflux_ratio"],
                        bottoms_flow=inputs["bottoms_flow_mol_s"],
                        save_path=REPRESENTATIVE_FLOWSHEET if successful == 0 else None,
                    )
                    row = {**inputs, **result}

                    with open(OUTPUT_FILE, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=HEADERS)
                        writer.writerow(row)

                    successful += 1
                    consecutive_failures = 0
                    print(
                        f"\rGOOD: {successful}/{TARGET_ROWS}   FAILED: {failed}",
                        end="",
                        flush=True,
                    )
                except Exception as e:
                    failed += 1
                    consecutive_failures += 1
                    error = str(e).splitlines()[0]

                    with open(FAILED_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                inputs["feed_temperature_K"],
                                inputs["feed_pressure_Pa"],
                                inputs["benzene_feed_fraction"],
                                inputs["number_of_stages"],
                                inputs["feed_stage"],
                                inputs["reflux_ratio"],
                                inputs["bottoms_flow_mol_s"],
                                error,
                            ]
                        )

                    print(
                        f"\rGOOD: {successful}/{TARGET_ROWS}   FAILED: {failed}"
                        f"   LAST ERROR: {error[:120]}",
                        end="",
                        flush=True,
                    )

                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        raise RuntimeError(
                            f"Stopped after {consecutive_failures} consecutive failures. "
                            f"Last error: {error}"
                        ) from e
    finally:
        try:
            automation.ReleaseResources()
        except Exception:
            pass

    print()
    print()
    print("Dataset generation finished.")
    print("Successful rows:", successful)
    print("Failed simulations:", failed)
    print("Dataset:", OUTPUT_FILE)
    print("Failures:", FAILED_FILE)
    write_metadata(successful, failed)
    print("Metadata:", METADATA_FILE)
    print("Representative flowsheet:", REPRESENTATIVE_FLOWSHEET)


if __name__ == "__main__":
    main()
