"""Tune rerankers on validation users, then compare selected settings on test."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from task1.io_utils import write_csv, write_json
from task1.model_registry import MODEL_SPECS, PROJECT_ROOT
from task3.metrics import evaluate_lists
from task3.prepare import ARTIFACTS, DEFAULT_MODELS, SCORE_HYBRIDS, prepare
from task3.rerankers import METHODS, build_profiles, candidates, rerank


OBJECTIVES = {
    "diversity": ("IntraListDiversity", 1),
    "calibration": ("Miscalibration", -1),
    "user_fairness": ("UPD", -1),
    "item_fairness": ("ExposureGap", -1),
}


def select_setting(rows, method, k, max_loss):
    """Best objective within a relative NDCG loss budget; zero is a fallback."""
    baseline = next(row for row in rows if row["strength"] == 0)
    eligible = [row for row in rows if row[f"NDCG@{k}"] >= (1 - max_loss) * baseline[f"NDCG@{k}"]]
    metric, direction = OBJECTIVES[method]
    return max(eligible, key=lambda row: (direction * row[f"{metric}@{k}"], -row["strength"]))


def make_lists(candidate_rows, scores, users, method, strength, k, profiles):
    return [rerank(candidate_rows[user], scores[user], method, strength, k, profiles, user)
            for user in users]


def run(models, strengths, k=10, candidate_count=100, max_loss=.05, seed=2020, output=None):
    if not models or len(set(models)) != len(models) or any(m not in [*MODEL_SPECS, *SCORE_HYBRIDS] for m in models):
        raise ValueError("Choose distinct supported models")
    if k < 1 or candidate_count < k or not 0 <= max_loss <= 1:
        raise ValueError("Need 0 < K <= candidates and a loss budget between 0 and 1")
    if not strengths or any(not np.isfinite(s) or not 0 <= s <= 1 for s in strengths):
        raise ValueError("Strengths must be finite and between 0 and 1")
    strengths = sorted(set([0., *strengths]))
    output = Path(output) if output else PROJECT_ROOT / "results/task3"
    scores_by_model, train, valid, test, genres, tune_users, manifest = prepare(models, seed)
    profiles = build_profiles(train, genres)
    test_users = np.flatnonzero(test.any(axis=1))
    test_users = test_users[test_users != 0]
    if not len(test_users):
        raise ValueError("No test users")
    validation_rows, test_rows, per_user, group_rows, selections = [], [], [], [], []
    # Finish every selection before any test metric is calculated.
    for model_name, scores in scores_by_model.items():
        print(f"Tuning rerankers: {model_name}", flush=True)
        candidate_rows = candidates(scores, train, candidate_count)
        np.savez_compressed(ARTIFACTS / f"{model_name}-validation-candidates.npz",
                            items=candidate_rows, users=tune_users)
        for method in METHODS:
            trials = []
            for strength in strengths:
                lists = make_lists(candidate_rows, scores, tune_users, method, strength, k, profiles)
                metrics, _ = evaluate_lists(lists, tune_users, valid, profiles, k)
                trials.append({"model": model_name, "method": method, "strength": strength, **metrics})
            chosen = select_setting(trials, method, k, max_loss)
            validation_rows.extend(trials)
            selections.append({"model": model_name, "method": method, "strength": chosen["strength"]})
    write_csv(output / "validation.csv", validation_rows)
    write_json(output / "selected.json", selections)
    manifest.update({
        "status": "test evaluation pending",
        "models": models, "topk": k, "candidates": candidate_count, "strengths": strengths,
        "max_relative_validation_ndcg_loss": max_loss, "test_users": len(test_users),
        "selection": "Best method-specific validation objective within NDCG budget; ties prefer smaller strength",
        "candidate_order": "descending score, ties by ascending internal item ID",
        "score_scaling": "per-user candidate min-max; mean relevance in greedy list objective",
        "history_mask": "train for validation; train plus all validation interactions for test",
        "profiles": "training interactions only; equal weight per interaction, fractional genres per item",
        "popularity_groups": "head = top ceil(20% of catalog) by training count; remaining items = tail",
        "item_fairness_target": profiles["target_exposure"].tolist(),
        "fairness": "User fairness targets popularity preference, not equal accuracy; item fairness targets "
                    "catalog-proportional head/tail exposure, not demographic or merit-based fairness",
        "metrics": {
            "Miscalibration": "KL(p || 0.99*q + 0.01*p), natural log, uniform list weights; lower is better",
            "UPD": "total variation of training-history and recommended head/tail distributions; lower is better",
            "ExposureGap": "total variation of aggregate discounted head/tail exposure and catalog proportions",
            "ExposureGini": "Gini of item exposure, including never-recommended items; lower is better",
            "ActivityGRU": "absolute gap in mean NDCG between <= training median activity and > median activity",
        },
        "code_sha256": {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted((PROJECT_ROOT / "task3").glob("*.py"))},
    })
    write_json(output / "protocol.json", manifest)
    for model_name, scores in scores_by_model.items():
        print(f"Test evaluation: {model_name}", flush=True)
        candidate_rows = candidates(scores, train | valid, candidate_count)
        np.savez_compressed(ARTIFACTS / f"{model_name}-test-candidates.npz", items=candidate_rows, users=test_users)
        settings = [{"model": model_name, "method": "none", "strength": 0.}]
        settings += [row for row in selections if row["model"] == model_name]
        rankings = {}
        for setting in settings:
            method = setting["method"]
            lists = make_lists(candidate_rows, scores, test_users,
                               "diversity" if method == "none" else method,
                               setting["strength"], k, profiles)
            metrics, users = evaluate_lists(lists, test_users, test, profiles, k)
            if method == "none":
                baseline = metrics
            test_rows.append({**setting, **metrics, **{
                f"delta_{metric}": value - baseline[metric] for metric, value in metrics.items()
            }})
            per_user.extend({**setting, **row} for row in users)
            for group in ("low", "high"):
                members = [row for row in users if row["activity_group"] == group]
                if members:
                    group_rows.append({**setting, "activity_group": group, "users": len(members), **{
                        name: float(np.mean([row[name] for row in members]))
                        for name in (f"NDCG@{k}", f"Miscalibration@{k}", f"UPD@{k}")
                    }})
            padded = np.full((len(test_users), k), -1, dtype=np.int64)
            for index, row in enumerate(lists):
                padded[index, :len(row)] = row
            rankings[method] = padded
        np.savez_compressed(ARTIFACTS / f"{model_name}-test-rankings.npz", users=test_users, **rankings)
    write_csv(output / "test.csv", test_rows)
    write_csv(ARTIFACTS / "per_user.csv", per_user)
    write_csv(output / "groups.csv", group_rows)
    manifest["status"] = "complete"
    write_json(output / "protocol.json", manifest)
    print(f"Saved results to {output}", flush=True)
    return test_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=[*MODEL_SPECS, *SCORE_HYBRIDS])
    parser.add_argument("--strengths", type=float, nargs="+", default=[0, .1, .25, .5, .75, 1])
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--max-ndcg-loss", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=2020, help="Validation-user partition and score RNG seed")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args.models, args.strengths, args.topk, args.candidates, args.max_ndcg_loss, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
