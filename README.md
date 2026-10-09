# RecSys Final Project - Group 26

This repository contains the group project for DSAIT4335 Recommender Systems. The course fork of RecBole is pinned as a
Git submodule at `third_party/RecBole_DSAIT4335`.

## Setup

```powershell
git clone --recurse-submodules https://github.com/AndreiLeIttu/RecSys-Final-Project-Group-26.git Final_Project
cd Final_Project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-project.txt
```

The committed Task 1.1-1.2 results were produced with Python 3.11.9 on CPU.

On macOS/Linux, from the repository root:

```bash
git submodule update --init --recursive
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-project.txt
```

If Python 3.11 is not installed, `uv` can provision it and install the same dependencies:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-project.txt
source .venv/bin/activate
```

For an existing clone, initialize or update the pinned submodule with:

```powershell
git submodule update --init --recursive
```

## Project Layout

- `task1/`: individual recommender models, tuning, and later hybrid models.

## Task 1.1 and 1.2

The implemented RecBole models are `Random`, `Pop`, `ItemKNN`, `UserKNN`, `BPR`, `FISM`, `SLIMElastic`, `EASE`, `NeuMF`,
`NGCF`, and `LightGCN`. `UserKNN` uses RecBole's `ItemKNN` implementation in user-based mode.

All models use MovieLens 100K with seed 2020, an 80/10/10 random-order user-grouped split, and full-ranking evaluation
over unseen items. Models are selected by validation MRR@10 and evaluated with Precision, Recall, nDCG, MRR, and Hit@10.
Neural models train for at most 20 epochs with per-epoch validation and early stopping; only the selected validation
winner is evaluated on the test set. All observed interactions are treated as positive.

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
- `results/best_models.csv` and `results/best_models.json`: selected settings and test results.
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

`FeatureCombination` concatenates learned user/item embeddings and their elementwise product, user/movie genres,
user/movie text vectors, and movie popularity. A neural network with one hidden layer produces a relevance score. It
trains with pairwise logistic loss and RecBole's uniform negative sampling.

The model is in `task1/feature_combination/model.py`, feature preparation is in `task1/feature_combination/features.py`,
and settings are in `task1/configs/models/FeatureCombination.yaml`. It uses the same split and validation MRR@10
selection as the individual models.

Movie metadata in `data/movie_metadata.tsv` comes from Assignment 1's `movielens/items.txt` (original movie IDs, titles,
genres, descriptions). Text embeddings use TF-IDF followed by a 32-dimensional SVD projection, cached under
`artifacts/features/`. The text transform uses the available catalog metadata, including held-out items, but no held-out
interaction labels. User profiles average the features of training-history movies; popularity is the normalized log of
training interaction counts. Profiles and popularity are rebuilt for each training split and stored as checkpoint
buffers.

Tune its three declared candidates and evaluate the validation winner:

```powershell
python -m task1.tune_models --models FeatureCombination
```

For a one-epoch validation-only installation check:

```powershell
python -c "from task1.experiment import train_model; from task1.model_registry import MODEL_SPECS; print(train_model(MODEL_SPECS['FeatureCombination'], {'epochs': 1, 'show_progress': False}))"
```

Load it through `task1.experiment.load_data_and_model`, just like other selected models. No pretrained collaborative
checkpoint or transformer download is needed. After the smoke check, verify scoring, checkpoint reload, and feature
switches with `python -m task1.feature_combination.check`. Feature-removal experiments use overrides `use_genres`,
`use_text`, and `use_popularity`: disable all three for the interaction-only comparison, or disable each individually to
measure its contribution. Select settings using validation before conducting the final test comparisons.

## Task 1.3: Weighted hybrid recommender

Run after setup, from the repository root:

```bash
python -m task1.train_hybrid
```

This combines all eleven individual recommenders. It reuses local checkpoints in `artifacts/best/`. Since checkpoints
are not committed, missing ones are automatically trained using the selected parameters in `results/best_models.json`
and the model configurations. You do not need to repeat the 47-trial search. The existing Task 1.2 result files are
preserved. Training all missing models can take several minutes on CPU.

For a smaller run, choose at least two distinct models:

```bash
python -m task1.train_hybrid --models Pop EASE
```

### How coefficients are learned

1. Freeze the individual models trained on the original 80% training split. Check that every checkpoint reproduces the
   same user/item token mappings and train/validation/test splits.
2. Collect each model's full-catalog prediction scores. Min-max normalize scores separately for each user and model
   using nonpadding items unseen in training. Constant score ranges become zero. This keeps different score scales
   comparable.
3. Build regression examples from all validation interactions (target 1) and five sampled unobserved items per positive
   (target 0), without replacement per user. Exclude padding, training interactions, and validation positives from
   negative sampling. Test labels are never used in fitting, normalization, or negative sampling. Unobserved items are
   assumed negatives and may include future test positives; they are not verified dislikes.
4. Fit a linear least-squares regression with an intercept, nonnegative model coefficients, and a sum-to-one constraint
   using SciPy's SLSQP optimizer. Minimize `mean((intercept + X @ weights - targets) ** 2)`.
5. Rank items by the learned weighted sum of normalized component scores. The intercept is saved for regression
   predictions but has no effect on ranking. Evaluate once on the test split with the existing RecBole metrics and
   history masking (training items for validation; training and validation items for test).

The validation metrics are **fit diagnostics**, since this split was used both for individual-model selection and
coefficient fitting. Test metrics are the held-out evaluation. This implements Task 1.3; hybrid hyperparameter tuning
(Task 1.5) and independently implemented evaluation metrics (Task 2) remain separate work. Regression predictions are
ranking scores, not calibrated probabilities, and need not fall in [0, 1].

Options: `--negative-ratio 5` controls the sampled negative count; `--user-batch-size 16` controls score collection
memory. Keep the batch size fixed for reproducibility: the course Random model draws new scores on each call.

Outputs:

- `results/weighted_hybrid.json`: weights, intercept, regression MSE, protocol, checkpoint paths, and validation/test
  metrics.
- `results/hybrid_coefficients.csv`: one learned coefficient per recommender.
- `results/weighted_hybrid.csv`: test accuracy metrics.
- `artifacts/hybrid/WeightedHybrid.npz`: frozen combined scores, coefficients, and user/item token mappings. This local
  artifact preserves the exact scores, including the Random component, without rerunning base-model inference.

Load the fitted hybrid for existing MovieLens users/items:

```python
from task1.train_hybrid import load_hybrid
from recbole.data.interaction import Interaction
import torch

hybrid = load_hybrid()
scores = hybrid.full_sort_predict(Interaction({"user_id": torch.tensor([1])}))
```

The example uses an internal RecBole user ID, not a raw MovieLens ID. The NPZ contains `user_tokens` and `item_tokens`
to map IDs back to MovieLens. For a recommendation list, exclude padding item 0 and the user's observed items before
sorting. The saved hybrid supports the existing catalog; adding users/items requires rebuilding it.

## Task 2.1 and 2.2: Independent evaluation and comparison

`task2.metrics` implements Precision@K, Recall@K, Hit@K, MRR@K, and nDCG@K from ordered item-ID lists and sets of
relevant test items. Each is macro- averaged over test users with at least one test interaction, matching RecBole's
ranking-evaluation user population. Users with no relevant items or empty recommendation lists score zero in the
reusable metric functions. Precision uses K as its denominator, including when fewer than K candidates exist. Unit tests
compare the implementations with RecBole for identical rankings. The evaluator also uses Task 1's full-sort test
batches, history mask, and PyTorch top-K ordering, so its accuracy results match the stored Task 1 test results to their
reported precision when evaluating the same checkpoint.

Beyond-accuracy definitions:

- **Coverage@K:** distinct recommended nonpadding items divided by the full nonpadding catalog size.
- **IntraListDiversity@K:** for each user, mean pairwise cosine distance of binary movie-genre vectors in their top-K
  list, then macro-averaged across evaluated users. Lists shorter than two items contribute zero.
- **Novelty@K:** mean self-information in bits, `-log2(p(item))`, where
  `p(item) = (training_count + 1) / (training_interactions + catalog_size)`.
- **AveragePopularity@K:** mean raw training-interaction count of recommended items, pooled across evaluated users.

All popularity values use training interactions only. Recommendations exclude padding, training items, and validation
items. Test interactions are used only as relevance labels, never to fit a model, select parameters, or normalize
scores. Rankings are cached by model checkpoint, split signature, and K under `artifacts/task2/recommendations/`.

Run the comparison for all Task 1 models, using saved best parameters to recover missing individual checkpoints when
possible:

```bash
python -m task2.evaluate
```

To evaluate only existing baseline checkpoints without triggering recovery:

```bash
python -m task2.evaluate --models Random Pop --topk 10 --no-recover-missing
```

`--models` accepts any of `Random`, `Pop`, `ItemKNN`, `UserKNN`, `BPR`, `FISM`, `SLIMElastic`, `EASE`, `NeuMF`, `NGCF`,
`LightGCN`, `FeatureCombination`, and `WeightedHybrid`. FeatureCombination and WeightedHybrid are included only when
their selected checkpoint/artifact is available (or recoverable from saved Task 1 settings). Missing or failed models
are listed in the evaluation summary.

The combined accuracy and beyond-accuracy table is written to `results/task2/all_metrics.csv`. Evaluated and skipped
models are also reported in the command output. Comparisons use only actual evaluation results, not Task 1's stored
metric values.

Run the Task 2 metric tests with:

```bash
python -m unittest discover -s tests -p 'test_task2_metrics.py'
```
