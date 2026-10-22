"""Tune weighted/feature/switching hybrids and run focused contribution analysis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys

import numpy as np
import torch
import yaml

from task1.experiment import load_data_and_model, resolve_project_path, train_model
from task1.hybrid_selection import (
    blend, crossfit_weights, refit_weights, switch_scores, user_folds,
)
from task1.io_utils import write_csv, write_json
from task1.model_registry import (
    INDIVIDUAL_MODEL_NAMES, MODEL_SPECS, PRIMARY_MODEL_NAMES, PROJECT_ROOT,
)
from task1.train_hybrid import collect_scores, dataset_signature
from task1.weighted_hybrid import normalize_scores
from task2.evaluate import _genre_features
from task2.metrics import accuracy_metrics, beyond_accuracy_metrics
from task2.recommendations import collect_test_rankings, rank_scores, rankings_for_users


COMPACT = ("SLIMElastic", "EASE", "NeuMF", "ItemKNN")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2020,
                        help="Other seeds reuse the seed-2020 choices without retuning")
    args = parser.parse_args()
    seed = args.seed
    if seed == 2020:
        subprocess.run([sys.executable, "-m", "task1.tune_models", "--models",
                        "FeatureCombination"], cwd=PROJECT_ROOT, check=True)

    selected = json.loads((PROJECT_ROOT / "results/best_models.json").read_text())
    if "FeatureCombination" not in selected:
        parser.error("Run the seed-2020 experiment first to select FeatureCombination")
    folder = PROJECT_ROOT / "results/hybrids" / f"seed{seed}"
    local = PROJECT_ROOT / "artifacts/hybrid" / f"seed{seed}"
    local.mkdir(parents=True, exist_ok=True)
    chosen_file = PROJECT_ROOT / "results/hybrids/seed2020/selection.json"
    choices = {} if seed == 2020 else json.loads(chosen_file.read_text())
    component_scores = {}
    signature = None
    dataset = train = valid = test = features = activity = None
    baseline_rankings = None

    def component(name, overrides=None):
        nonlocal signature, dataset, train, valid, test, features, activity, baseline_rankings
        parameters = {**selected[name]["parameters"], **(overrides or {})}
        key = hashlib.sha256(json.dumps([name, parameters, seed], sort_keys=True).encode()).hexdigest()[:16]
        record_file = local / f"{key}.json"
        if record_file.exists():
            record = json.loads(record_file.read_text())
            checkpoint = resolve_project_path(record["checkpoint"])
        elif seed == 2020 and not overrides:
            checkpoint = resolve_project_path(selected[name]["checkpoint"])
        else:
            checkpoint = local / f"{key}.pth"
        if not checkpoint.exists() and seed != 2020:
            # Reuse matching runs from the earlier multi-seed comparison.
            spec = MODEL_SPECS[name]
            required = {**yaml.safe_load(spec.config_file.read_text()), **parameters, "seed": seed}
            previous = sorted((PROJECT_ROOT / "artifacts/checkpoints").glob(
                f"*-{spec.recbole_model}-*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
            for candidate in previous:
                config = torch.load(candidate, map_location="cpu", weights_only=False)["config"]
                if all(config[k] == v for k, v in required.items()):
                    checkpoint = candidate
                    break
        if not checkpoint.exists():
            print(f"Training {name} seed={seed} overrides={overrides}", flush=True)
            result = train_model(MODEL_SPECS[name],
                                 {**parameters, "seed": seed, "show_progress": False})
            source = resolve_project_path(result["checkpoint"])
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if source != checkpoint:
                shutil.copy2(source, checkpoint)
        config, model, current_dataset, tr, va, te = load_data_and_model(checkpoint)
        if name == "Random":
            # The course Random shares draws within a batch: benchmark it using
            # the same test batches as Task 2, not the hybrid collection batches.
            rng_state = torch.get_rng_state()
            baseline_rankings = collect_test_rankings(model, te, current_dataset, 10)
            torch.set_rng_state(rng_state)
        current_signature = dataset_signature(current_dataset, (tr, va, te))
        if signature is not None and signature != current_signature:
            raise ValueError(f"Split/token mismatch for {name}")
        if signature is None:
            signature = current_signature
            dataset = current_dataset
            train, valid, test = [loader._dataset.inter_matrix(form="csr").toarray() > 0
                                  for loader in (tr, va, te)]
            features = _genre_features(dataset)
            activity = train.sum(axis=1)
        fingerprint = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        cache_file = local / f"scores-{key}.npz"
        scores = None
        if cache_file.exists():
            with np.load(cache_file, allow_pickle=False) as cache:
                if str(cache["fingerprint"]) == fingerprint and str(cache["signature"]) == signature:
                    scores = cache["scores"].copy()
        if scores is None:
            scores = normalize_scores(collect_scores(model, dataset, 16), train)
            np.savez_compressed(cache_file, scores=scores, fingerprint=fingerprint, signature=signature)
        write_json(record_file, {"checkpoint": checkpoint.relative_to(PROJECT_ROOT).as_posix(),
                                 "parameters": parameters})
        return scores

    def rankings(scores, split):
        seen = train if split == "valid" else train | valid
        return rank_scores(scores.copy(), seen, 10)

    def measure(scores, split="test", ranked_override=None):
        truth = valid if split == "valid" else test
        users = np.flatnonzero(truth[:, 1:].any(axis=1))
        users = users[users != 0]
        ranked = rankings_for_users(
            rankings(scores, split) if ranked_override is None else ranked_override, users)
        relevant = [set(np.flatnonzero(truth[u])) - {0} for u in users]
        result = accuracy_metrics(ranked, relevant, 10)
        if split == "test":
            result.update(beyond_accuracy_metrics(ranked, train.sum(axis=0), features, 10))
        return result

    names = (list(INDIVIDUAL_MODEL_NAMES) if seed == 2020 else
             list(dict.fromkeys([*PRIMARY_MODEL_NAMES, *choices["weighted_models"]])))
    for name in (*names, "FeatureCombination"):
        print(f"Loading {name} seed={seed}", flush=True)
        component_scores[name] = component(name)
    all_scores = np.stack([component_scores[n] for n in names])
    valid_users = np.flatnonzero(valid[:, 1:].any(axis=1))
    folds = user_folds(valid_users, len(train), seed)
    trials = []
    if seed == 2020:
        for group, members in (("full", names), ("compact", list(COMPACT))):
            subset = all_scores[[names.index(n) for n in members]]
            for ratio in (1, 5, 10):
                oof = crossfit_weights(subset, train, valid, ratio, seed, folds)
                score = measure(oof, "valid")["MRR@10"]
                trials.append({"kind": "weighted", "components": group,
                               "negative_ratio": ratio, "validation_mrr@10": score})
        winner = max(trials, key=lambda r: (r["validation_mrr@10"],
                                           r["components"] == "compact"))
        choices["weighted_models"] = names if winner["components"] == "full" else list(COMPACT)
        choices["negative_ratio"] = winner["negative_ratio"]
        activity_values = activity[valid_users]
        for quantile in (0.25, 0.5, 0.6, 0.75):
            threshold = float(np.quantile(activity_values, quantile))
            for reverse in (False, True):
                scores = switch_scores(component_scores["FeatureCombination"],
                                       component_scores["SLIMElastic"], activity, threshold, reverse)
                trials.append({"kind": "switching", "quantile": quantile,
                               "threshold": threshold, "reverse": reverse,
                               "validation_mrr@10": measure(scores, "valid")["MRR@10"]})
        for endpoint in ("FeatureCombination", "SLIMElastic"):
            trials.append({"kind": "switching", "endpoint": endpoint,
                           "validation_mrr@10": measure(component_scores[endpoint], "valid")["MRR@10"]})
        best_switch = max((r for r in trials if r["kind"] == "switching"),
                          key=lambda r: r["validation_mrr@10"])
        choices["switching"] = best_switch
        choices["protocol"] = "Weighted: 3-fold validation-user cross-fit; others: validation MRR@10. Test never selects settings."
        choices["limitation"] = "Base hyperparameters/early stopping already use validation; this is not fully nested validation."
        write_json(chosen_file, choices)
        write_csv(folder / "tuning.csv", trials)

    members = choices["weighted_models"]
    subset = all_scores[[names.index(n) for n in members]]
    fitted = refit_weights(subset, train, valid, choices["negative_ratio"], seed)
    weighted = blend(subset, fitted["weights"])
    switch_choice = choices["switching"]
    if "endpoint" in switch_choice:
        switching = component_scores[switch_choice["endpoint"]]
    else:
        threshold = float(np.quantile(activity[valid_users], switch_choice["quantile"]))
        switching = switch_scores(component_scores["FeatureCombination"],
                                  component_scores["SLIMElastic"], activity, threshold,
                                  switch_choice["reverse"])
    comparison = {**component_scores, "TunedWeightedHybrid": weighted,
                  "SwitchingHybrid": switching}
    write_csv(folder / "comparison.csv", [
        {"model": name, **measure(scores, ranked_override=baseline_rankings if name == "Random" else None)}
        for name, scores in comparison.items()
    ])
    for name, scores in (("TunedWeightedHybrid", weighted), ("SwitchingHybrid", switching)):
        np.savez_compressed(local / f"{name}.npz", scores=scores, split_signature=signature,
                            user_field=dataset.uid_field, item_field=dataset.iid_field,
                            user_tokens=np.asarray(dataset.field2id_token[dataset.uid_field], dtype=str),
                            item_tokens=np.asarray(dataset.field2id_token[dataset.iid_field], dtype=str))
    write_json(folder / "protocol.json", {"seed": seed, "split_signature": signature,
                "selection": choices, "weights": dict(zip(members, fitted["weights"].tolist())),
                "intercept": fitted["intercept"], "training_mse": fitted["training_mse"]})

    # Coefficient size and actual score contribution answer different questions.
    users = np.flatnonzero(test[:, 1:].any(axis=1))
    top = rankings(weighted, "test")[users]
    contributions, ablations, overlap = [], [], []
    for i, name in enumerate(members):
        weight = float(fitted["weights"][i])
        contribution = weight * subset[i, users[:, None], top]
        contributions.append({"model": name, "weight": weight,
                              "top10_contribution_mean": float(contribution.mean()),
                              "top10_contribution_std": float(contribution.std())})
        if weight <= 1e-6:
            continue
        keep = np.arange(len(members)) != i
        fixed = weighted - weight * subset[i]
        refit = refit_weights(subset[keep], train, valid, choices["negative_ratio"], seed)
        ablations.extend([
            {"removed": name, "mode": "fixed_weights", **measure(fixed)},
            {"removed": name, "mode": "refit_weights", **measure(blend(subset[keep], refit["weights"]))},
        ])
    active = [n for n, w in zip(members, fitted["weights"]) if w > 1e-6]
    for i, first in enumerate(active):
        first_top = rankings(component_scores[first], "test")[users]
        first_rr = np.array([next((1 / r for r, item in enumerate(row, 1) if test[u, item]), 0)
                             for u, row in zip(users, first_top)])
        for second in active[i + 1:]:
            second_top = rankings(component_scores[second], "test")[users]
            second_rr = np.array([next((1 / r for r, item in enumerate(row, 1) if test[u, item]), 0)
                                  for u, row in zip(users, second_top)])
            overlap.append({"first": first, "second": second,
                            "top10_jaccard": float(np.mean([len(set(a)&set(b))/len(set(a)|set(b))
                                                            for a,b in zip(first_top,second_top)])),
                            "first_wins": int((first_rr > second_rr).sum()),
                            "second_wins": int((second_rr > first_rr).sum()),
                            "ties": int((first_rr == second_rr).sum())})
    write_csv(folder / "coefficients.csv", contributions)
    write_csv(folder / "component_ablation.csv", ablations)
    write_csv(folder / "overlap.csv", overlap)
    if seed == 2020:
        feature_ablation = []
        for label, overrides in (
            ("interaction_only", {"use_genres": False, "use_text": False, "use_popularity": False}),
            ("no_genres", {"use_genres": False}),
            ("no_text", {"use_text": False}),
            ("no_popularity", {"use_popularity": False}),
        ):
            scores = component("FeatureCombination", overrides)
            feature_ablation.append({"variant": label, **measure(scores)})
        write_csv(folder / "feature_ablation.csv", feature_ablation)
    # One compact table keeps seed robustness visible without another pipeline.
    grouped = {}
    for path in sorted((PROJECT_ROOT / "results/hybrids").glob("seed*/comparison.csv")):
        with path.open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                grouped.setdefault(row["model"], []).append(row)
    summary = []
    for name, rows in grouped.items():
        row = {"model": name, "n_seeds": len(rows)}
        for metric in rows[0]:
            if metric == "model":
                continue
            values = [float(r[metric]) for r in rows]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append(row)
    write_csv(PROJECT_ROOT / "results/hybrids/summary.csv", summary)
    print(f"Completed hybrid experiments: {folder.relative_to(PROJECT_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
