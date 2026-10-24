"""Load consistent checkpoints and fit a separate Task 3 hybrid."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from task1.experiment import load_data_and_model, train_model
from task1.hybrid_selection import switch_scores
from task1.model_registry import INDIVIDUAL_MODEL_NAMES, MODEL_SPECS, PROJECT_ROOT
from task1.train_hybrid import collect_scores, dataset_signature
from task1.weighted_hybrid import fit_weights, normalize_scores, regression_pairs
from task2.evaluate import _genre_features, _interaction_matrix
from recbole.utils import init_seed


ARTIFACTS = PROJECT_ROOT / "artifacts" / "task3"
SCORE_HYBRIDS = ("WeightedHybrid", "TunedWeightedHybrid", "SwitchingHybrid")
DEFAULT_MODELS = ["EASE", "WeightedHybrid", "FeatureCombination",
                  "TunedWeightedHybrid", "SwitchingHybrid", "Random", "Pop"]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validation_users(valid, seed):
    """Separate users prevent hybrid fitting from seeing reranker-tuning labels."""
    users = np.flatnonzero(valid.any(axis=1))
    users = users[users != 0]
    if len(users) < 2:
        raise ValueError("Need at least two validation users")
    users = np.random.default_rng(seed).permutation(users)
    midpoint = len(users) // 2
    return np.sort(users[:midpoint]), np.sort(users[midpoint:])


def selected_checkpoint(name, record):
    """Keep Task 1 files intact; recover missing/outdated checkpoints locally."""
    local = ARTIFACTS / "checkpoints" / f"{name}.pth"
    original = PROJECT_ROOT / record["checkpoint"]
    for path in (local, original):
        if path.exists():
            config = torch.load(path, map_location="cpu", weights_only=False)["config"]
            if all(config[key] == value for key, value in record["parameters"].items()):
                return path
    print(f"Training {name} with the selected Task 1 settings", flush=True)
    result = train_model(MODEL_SPECS[name], {
        **record["parameters"], "show_progress": False,
        "checkpoint_dir": str(ARTIFACTS / "checkpoints"),
    })
    if not result["checkpoint"]:
        raise RuntimeError(f"Training {name} did not produce a checkpoint")
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_ROOT / result["checkpoint"], local)
    return local


def fit_hybrid(score_matrices, members, train, valid, fit_users, ratio, seed):
    """Refit frozen Task 1 choices without reranker-tuning users' labels."""
    fit_valid = np.zeros_like(valid)
    fit_valid[fit_users] = valid[fit_users]
    users, items, targets = regression_pairs(train, fit_valid, ratio, seed)
    normalized = [normalize_scores(score_matrices[name], train) for name in members]
    fitted = fit_weights(np.column_stack([scores[users, items] for scores in normalized]), targets)
    hybrid = np.full(train.shape, fitted["intercept"], dtype=np.float32)
    for weight, scores in zip(fitted["weights"], normalized):
        hybrid += weight * scores
    record = {
        "weights": dict(zip(members, fitted["weights"].tolist())),
        "intercept": fitted["intercept"], "training_mse": fitted["training_mse"],
        "fit_pairs": len(targets), "negative_ratio": ratio,
    }
    return hybrid, record


def prepare(models, seed, include_components=False):
    selected = json.loads((PROJECT_ROOT / "results/best_models.json").read_text())
    choice_path = PROJECT_ROOT / "results/hybrids/seed2020/selection.json"
    choices = (json.loads(choice_path.read_text())
               if any(name in models for name in ("TunedWeightedHybrid", "SwitchingHybrid")) else {})
    switch_choice = choices.get("switching", {})
    switch_members = ([switch_choice["endpoint"]] if "endpoint" in switch_choice else
                      ["FeatureCombination", "SLIMElastic"])
    names = list(dict.fromkeys([
        *[name for name in models if name not in SCORE_HYBRIDS],
        *(INDIVIDUAL_MODEL_NAMES if "WeightedHybrid" in models else ()),
        *(choices["weighted_models"] if "TunedWeightedHybrid" in models else ()),
        *(switch_members if "SwitchingHybrid" in models else ()),
    ]))
    missing = [name for name in names if name not in selected]
    if missing:
        raise ValueError(f"Run Task 1 tuning first; no selected settings for {missing}")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    score_matrices, components = {}, {}
    signature = None
    scoring_hash = file_hash(PROJECT_ROOT / "task1/train_hybrid.py")
    for name in names:
        path = selected_checkpoint(name, selected[name])
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(path)
        current = dataset_signature(dataset, (train_data, valid_data, test_data))
        if signature is not None and current != signature:
            raise ValueError(f"{name} has a different split or token mapping")
        if signature is None:
            signature = current
            train, valid, test = map(_interaction_matrix, (train_data, valid_data, test_data))
            genres = _genre_features(dataset)
            user_tokens = np.asarray(dataset.field2id_token[dataset.uid_field], dtype=str)
            item_tokens = np.asarray(dataset.field2id_token[dataset.iid_field], dtype=str)
        fingerprint = file_hash(path)
        cache_key = f"{fingerprint}:{signature}:{scoring_hash}:{seed}:16"
        cache = ARTIFACTS / f"{name}-scores.npz"
        scores = None
        if cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                if str(data["key"]) == cache_key:
                    scores = data["scores"].copy()
        if scores is None:
            print(f"Scoring {name}", flush=True)
            init_seed(seed, True)
            scores = collect_scores(model, dataset, 16)
            np.savez_compressed(cache, key=cache_key, scores=scores)
        if scores.shape != train.shape or not np.isfinite(scores).all():
            raise ValueError(f"Invalid score matrix for {name}")
        score_matrices[name] = scores
        components[name] = {
            "checkpoint": path.relative_to(PROJECT_ROOT).as_posix(),
            "checkpoint_sha256": fingerprint,
            "parameters": selected[name]["parameters"],
            "score_sha256": hashlib.sha256(scores.tobytes()).hexdigest(),
        }
        del model
    fit_users, tune_users = validation_users(valid, seed)
    hybrids = {}
    for name in ("WeightedHybrid", "TunedWeightedHybrid"):
        if name not in models:
            continue
        members = INDIVIDUAL_MODEL_NAMES if name == "WeightedHybrid" else choices["weighted_models"]
        ratio = 5 if name == "WeightedHybrid" else choices["negative_ratio"]
        score_matrices[name], hybrids[name] = fit_hybrid(
            score_matrices, members, train, valid, fit_users, ratio, seed)
    if "SwitchingHybrid" in models:
        if "endpoint" in switch_choice:
            score_matrices["SwitchingHybrid"] = score_matrices[switch_choice["endpoint"]].copy()
            hybrids["SwitchingHybrid"] = {"endpoint": switch_choice["endpoint"]}
        else:
            activity = train.sum(axis=1)
            threshold = float(np.quantile(activity[fit_users], switch_choice["quantile"]))
            score_matrices["SwitchingHybrid"] = switch_scores(
                score_matrices["FeatureCombination"], score_matrices["SLIMElastic"],
                activity, threshold, switch_choice["reverse"])
            hybrids["SwitchingHybrid"] = {"quantile": switch_choice["quantile"],
                                          "threshold": threshold, "reverse": switch_choice["reverse"]}
    for name, record in hybrids.items():
        np.savez_compressed(ARTIFACTS / f"{name}.npz", scores=score_matrices[name],
                            user_field=dataset.uid_field, item_field=dataset.iid_field,
                            user_tokens=user_tokens, item_tokens=item_tokens,
                            fit_users=fit_users, tune_users=tune_users,
                            split_signature=signature,
                            **({"weights": list(record["weights"].values()), "intercept": record["intercept"]}
                               if "weights" in record else {}))
    manifest = {
        "split_signature": signature, "seed": seed, "score_batch_size": 16,
        "python": sys.version.split()[0],
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "torch", "scipy", "pandas", "recbole")},
        "components": components, "hybrid": hybrids.get("WeightedHybrid"), "hybrids": hybrids,
        "task1_hybrid_selection": choices,
        "recbole_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT / "third_party/RecBole_DSAIT4335",
            text=True).strip(),
        "hybrid_fit_users": len(fit_users), "reranker_tune_users": len(tune_users),
        "validation_user_split": "seeded permutation, first half fits hybrid, second half tunes rerankers",
        "limitation": "Base hyperparameters, early stopping and Task 1 hybrid structure/ratio/switch "
                      "selection used the original full validation split. "
                      "The reranker tuning scores are development results, not independent estimates. "
                      "No test labels are used for fitting or selection.",
    }
    output_names = list(dict.fromkeys([*names, *models])) if include_components else models
    return {name: score_matrices[name] for name in output_names}, train, valid, test, genres, tune_users, manifest
