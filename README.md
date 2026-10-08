# RecSys Final Project - Group 26

This repository contains the group project for DSAIT4335 Recommender Systems.
The course fork of RecBole is pinned as a Git submodule at
`third_party/RecBole_DSAIT4335`.

## Setup

```powershell
git clone --recurse-submodules https://github.com/AndreiLeIttu/RecSys-Final-Project-Group-26.git Final_Project
cd Final_Project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-project.txt
```

The committed results were produced with Python 3.11.9 on CPU.

For an existing clone, initialize or update the pinned submodule with:

```powershell
git submodule update --init --recursive
```

## Project Layout

- `task1/`: individual recommender models, tuning, and later hybrid models.

## Task 1.1 and 1.2

The implemented RecBole models are `Random`, `Pop`, `ItemKNN`, `UserKNN`,
`BPR`, `FISM`, `SLIMElastic`, `EASE`, `NeuMF`, `NGCF`, and `LightGCN`.
`UserKNN` uses RecBole's `ItemKNN` implementation in user-based mode.

All models use MovieLens 100K with seed 2020, an 80/10/10 random-order
user-grouped split, and full-ranking evaluation over unseen items. Models are
selected by validation MRR@10 and evaluated with Precision, Recall, nDCG, MRR,
and Hit@10. Neural models train for at most 20 epochs with per-epoch validation
and early stopping; only the selected validation winner is evaluated on the
test set. All observed interactions are treated as positive.

Run all 50 configured trials (47 individual-model trials and 3 hybrid trials):

```powershell
python -m task1.tune_models
```

Run selected models:

```powershell
python -m task1.tune_models --models LightGCN NGCF NeuMF
```

For a quick preliminary check, run only the first candidate per model:

```powershell
python -m task1.tune_models --models Random Pop --max-trials 1
```

Completed trials are reused when the command is run again.

The pipeline writes:

- `results/tuning_trials.csv`: every completed tuning trial.
- `results/best_models.csv` and `results/best_models.json`: selected settings
  and test results.
- `task1/configs/best/`: generated configuration for each selected model.
- `artifacts/best/`: selected checkpoints (stored locally and not committed).

After running the tuning pipeline, load a selected checkpoint with:

```python
from task1.experiment import load_data_and_model

config, model, dataset, train_data, valid_data, test_data = (
    load_data_and_model("artifacts/best/EASE.pth")
)
```
## Feature combination hybrid

`FeatureCombination` concatenates learned user/item embeddings and their
elementwise product, user/movie genres, user/movie text vectors, and movie
popularity. A neural network with one hidden layer produces a relevance score.
It trains with pairwise logistic loss and RecBole's uniform negative sampling.

The model is in `task1/feature_combination/model.py`, feature preparation is in
`task1/feature_combination/features.py`, and settings are in
`task1/configs/models/FeatureCombination.yaml`. It uses the same split and
validation MRR@10 selection as the individual models.

Movie metadata in `data/movie_metadata.tsv` comes from Assignment 1's
`movielens/items.txt` (original movie IDs, titles, genres, descriptions).
Text embeddings use TF-IDF followed by a 32-dimensional SVD projection, cached
under `artifacts/features/`. The text transform uses the available catalog
metadata, including held-out items, but no held-out interaction labels.
User profiles average the features of training-history movies; popularity is
the normalized log of training interaction counts. Profiles and popularity
are rebuilt for each training split and stored as checkpoint buffers.

Tune its three declared candidates and evaluate the validation winner:

```powershell
python -m task1.tune_models --models FeatureCombination
```

For a one-epoch validation-only installation check:

```powershell
python -c "from task1.experiment import train_model; from task1.model_registry import MODEL_SPECS; print(train_model(MODEL_SPECS['FeatureCombination'], {'epochs': 1, 'show_progress': False}))"
```

Load it through `task1.experiment.load_data_and_model`, just like other selected
models. No pretrained collaborative checkpoint or transformer download is needed.
After the smoke check, verify scoring, checkpoint reload, and feature switches with
`python -m task1.feature_combination.check`.
Feature-removal experiments use overrides `use_genres`, `use_text`, and
`use_popularity`: disable all three for the interaction-only comparison, or
disable each individually to measure its contribution. Select settings using
validation before conducting the final test comparisons.
