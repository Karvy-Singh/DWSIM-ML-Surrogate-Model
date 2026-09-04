# DWSIM Distillation Model

This project trains a fast model to reproduce a DWSIM benzene-toluene
distillation column. Given the operating conditions, it predicts:

- benzene purity in the distillate,
- toluene purity in the bottoms,
- condenser duty, and
- reboiler duty.

The simulation uses DWSIM 9.0.5 and the Peng-Robinson property package.

## Main Result

The selected model is a neural network with two hidden layers of 128 and 64
units. It was selected by comparing models across five different data splits,
not from one lucky test result.

| Final check | Result |
| --- | ---: |
| Mean normalized error across four outputs | 0.005244 |
| Mean R2 across four outputs | 0.999493 |
| Invalid purity predictions | 0 / 4,000 |
| Largest benzene-balance error | 1.42e-14 mol/s |

The high scores are expected because the model is learning deterministic,
noise-free DWSIM results. They do not prove accuracy on a real plant.

## How the Data Are Used

The 10,000 successful simulations are divided into three groups:

- 7,000 rows train and compare the models.
- 1,000 rows estimate 90% prediction ranges.
- 2,000 rows provide the final check and are not used to choose the model.

The model first predicts both product purities. A small correction then makes
those two values obey the benzene material balance exactly. This avoids simply
cutting invalid values off at 0 or 1.

## Files

- `DWSIM_Flowsheet.dwxmz`: one solved DWSIM flowsheet.
- `dataset.csv`: 10,000 successful simulations.
- `failed_cases.csv`: 340 non-converged cases used by the failure analysis.
- `Code/generate_dataset.py`: creates and solves DWSIM cases.
- `Code/ml_model_comparison.ipynb`: short first comparison of four model types.
- `Code/main.py`: main entry point for final model comparison and checks.
- `Code/run_sweep_validation.py`: compares real DWSIM and ANN sweep curves.
- `Code/physics_surrogate.py`: applies the benzene material balance.
- `Code/predict_surrogate.py`: uses the saved model on new inputs.
- `Code/requirements.txt`: Python dependencies.
- `Report.pdf`: complete methodology, plots, tables, and discussion.
- `Results_Summary.pdf`: concise model results and sample predictions.

The notebook is useful for a simple walkthrough. The final reported numbers
come from `main.py`, which uses the stronger 7,000/1,000/2,000
method described above.

## Input Range

| Input | Range |
| --- | ---: |
| Feed temperature | 345-385 K |
| Feed pressure | 101,325-200,000 Pa |
| Benzene feed mole fraction | 0.25-0.75 |
| Number of stages | 18-35 |
| Sampled feed location | 30%-70% of column height |
| Reflux ratio | 1.5-4.5 |
| Bottoms flow | 25-75 mol/s |

Feed flow is fixed at 100 mol/s. Column top pressure is 101,325 Pa and the
specified pressure drop is zero. DWSIM counts the condenser as stage 0.

The generated integer feed stages produce actual relative locations from
0.27778 to 0.72222. Duties are the signed DWSIM values in kW, so most reboiler
duties are negative.

## Install

Dataset generation requires Linux and DWSIM. Set `DWSIM_DIR` if
`DWSIM.Automation.dll` is not in a standard DWSIM location. Training the model
does not require DWSIM.

```bash
python -m venv venv
venv/bin/python -m pip install -r Code/requirements.txt
```

## Reproduce the Results

Run the full comparison and all checks:

```bash
venv/bin/python Code/main.py --results-dir reproduced_results
```

For a quicker code check, use smaller models and three data splits:

```bash
venv/bin/python Code/main.py --quick --results-dir quick_results
```

The script regenerates the detailed CSV audit tables, plots, data-group
assignments, physical checks, sample predictions, and final fitted model in the
chosen results directory. These generated intermediates are summarized in the
submitted report and results summary rather than duplicated in the archive.

Run the two direct DWSIM-versus-surrogate validation sweeps after training:

```bash
venv/bin/python Code/run_sweep_validation.py \
  --model reproduced_results/final_surrogate.joblib \
  --results-dir reproduced_results
```

This evaluates a representative operating point while sweeping reflux ratio
and number of stages. It requires DWSIM because every plotted reference point
is solved again through DWSIM Automation.

## Use the Saved Model

The input CSV must contain these seven columns:

```text
feed_temperature_K
feed_pressure_Pa
benzene_feed_fraction
number_of_stages
feed_stage
reflux_ratio
bottoms_flow_mol_s
```

`number_of_stages` and `feed_stage` must be integers. The script calculates the
relative feed location internally and rejects values outside the training
range.

```bash
venv/bin/python Code/predict_surrogate.py new_cases.csv \
  --model reproduced_results/final_surrogate.joblib \
  --output predictions.csv
```

## Regenerate the Dataset

Every row requires a new DWSIM solve, so generating 10,000 rows takes time.

```bash
TARGET_ROWS=10000 venv/bin/python Code/generate_dataset.py
```

The generator writes `dataset.csv`, `failed_cases.csv`,
`dataset_metadata.json`, and `DWSIM_Flowsheet.dwxmz`. It samples 300 cases at a
time. Failed cases are logged and replaced until the requested number of
successful rows is reached.

## Limits

- The model represents only this DWSIM setup and property package.
- Do not use it outside the listed input ranges.
- Failed DWSIM cases are not represented by the model.
- The final score measures agreement with DWSIM, not with plant data.
- The simulation has not been checked against experimental or plant data.
