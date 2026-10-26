# RecSys Final Project - Group 26

DSAIT4335 project on MovieLens 100K: individual and hybrid recommenders,
independent accuracy and beyond-accuracy evaluation, and reranking for diversity,
calibration and popularity-based fairness.

## Setup

Requirements: Git and Python 3.11. Experiments run on CPU. The course RecBole
repository is included as a Git submodule pinned to **`7aa0ec9`**, the corrected
split-export version used for the results. No separate RecBole clone is needed.

```bash
git clone --recurse-submodules https://github.com/AndreiLeIttu/RecSys-Final-Project-Group-26.git Final_Project
cd Final_Project
```

Create and activate the environment:

```powershell
# Windows (PowerShell)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

Then install the pinned dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-project.txt
```

For an existing clone, run `git submodule update --init --recursive` after pulling
the project. Keep the pinned revision rather than updating RecBole independently.
The submodule includes the dataset; `data/movie_metadata.tsv` supplies movie
metadata for the feature-combination hybrid.

## Reproduce the experiments

Run all commands below **from the repository root with the environment active**.
Committed CSV/JSON results can be inspected immediately. Checkpoints and caches
under `artifacts/` are not committed: a fresh clone retrains missing models using
the saved best settings. CPU training is required and runtime depends on hardware.
Reruns write to the result locations below and reuse available local checkpoints
and caches where supported; small numerical differences between environments are
possible.

### 1. Individual models and hybrids (Task 1)

The eleven individual models are Random, Pop, ItemKNN, UserKNN, BPR, FISM,
SLIMElastic, EASE, NeuMF, NGCF and LightGCN. UserKNN uses RecBole's ItemKNN in
user-based mode. Hybrids combine scores through regression, combine collaborative
and metadata features, or switch between models by user activity.

```bash
# Export the corrected seed-2020 train/validation/test split.
python third_party/RecBole_DSAIT4335/save_split.py --model BPR --dataset ml-100k --config_files "task1/configs/common.yaml task1/configs/export_split.yaml" --output_dir saved/splits

# Recover selected individual models and fit the reference weighted hybrid.
python -m task1.train_hybrid

# Repeat the nine primary individual models with fixed settings on three seeds.
python -m task1.evaluate_seeds --seeds 2020 2021 2022

# Tune hybrids, run component/feature ablations, and repeat on two more seeds.
python -m task1.hybrid_experiments
python -m task1.hybrid_experiments --seed 2021
python -m task1.hybrid_experiments --seed 2022
```

This uses the committed individual-model selections in `results/best_models.json`.
The seed-2020 hybrid command also runs the eight feature-combination candidates.
To reproduce the **individual-model search itself**, run these two commands before
the commands above: `python -m task1.tune_models` (49 primary candidates) and
`python -m task1.tune_models --models FISM NGCF` (12 supplementary candidates).
The exact candidates are in `task1/search_spaces.yaml`.

### 2. Independent evaluation and group analysis (Task 2)

```bash
python -m task2.evaluate
python -m task2.group_analysis
```

The first command compares all 15 Task 1 variants using independently implemented
accuracy, coverage, genre diversity, novelty and popularity metrics. The second
compares training-defined user/item groups and evaluates a group-aware hybrid.

### 3. Reranking and group impact (Task 3)

```bash
python -m task3.evaluate
python -m task3.plot_results
python -m task3.evaluate_hybrid_reranking
python -m task3.plot_hybrid_reranking
python -m task3.analyze_groups
```

These evaluate four rerankers, compare reranking before/after hybrid aggregation,
and measure group impacts. Task 3.3 defaults to TunedWeightedHybrid; Task 3.4
requires its saved rankings, so retain this execution order.

## Results and protocol

| Location | Contents |
|---|---|
| `results/tuning_trials.csv`, `results/best_models.json` | Tuning history and selected settings |
| `results/model_summary.csv` | Three-seed individual-model accuracy |
| `results/weighted_hybrid.json` | Reference regression hybrid and coefficients |
| `results/hybrids/` | Hybrid tuning, three-seed comparison, coefficients and ablations |
| `results/task2/` | Independent accuracy and beyond-accuracy comparison |
| `results/task2_groups/` | User/item groups and group-aware hybrid |
| `results/task3/` | Reranker selection, test results and trade-off figures |
| `results/task3_3/` | Hybrid reranking order, group impacts and figures |

All ratings are implicit positives. The main protocol is a user-grouped,
random-order 80/10/10 split (seed 2020), full-catalog ranking, validation MRR@10
selection, and test metrics at K=10. Test recommendations exclude training and
validation items. Task 1 robustness seeds 2021/2022 change both splitting and
training randomness. FISM and NGCF are supplementary single-seed comparisons.

Task 2 group analysis and Task 3 refit regression weights on half the validation
users, reserving the other half for gate/reranker selection. Their hybrid results
can therefore differ from the original Task 2 comparison. Task 3's `--seed`
controls sampling and validation-user partitioning, not a new base-model split.
Protocol JSON files record settings and limitations. Interpretations belong in
the report; CSV/JSON files retain the experimental evidence.

## Checks

```bash
python -m unittest discover -s tests
```

This checks metrics, selection, tuning-history preservation, group analysis and
reranking without rerunning the full training experiments.
