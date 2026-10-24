"""Compare reranking individual hybrid components with reranking the hybrid."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from task1.io_utils import write_csv, write_json
from task1.model_registry import PROJECT_ROOT
from task3.evaluate import select_setting
from task3.metrics import evaluate_lists
from task3.prepare import ARTIFACTS, prepare
from task3.rerankers import METHODS, build_profiles, candidates, rerank


HYBRIDS = ("WeightedHybrid", "TunedWeightedHybrid")
STRATEGIES = ("rerank_components_then_aggregate", "aggregate_then_rerank")
DEFAULT_STRENGTHS = [0, .1, .25, .5, .75, 1]


def rank_aggregate(items, component_scores, members, weights, method, strength,
                   k, profiles, user):
    """Aggregate reranked component lists with the hybrid's fitted weights."""
    items = np.asarray(items, dtype=int)
    items = items[items > 0]
    if len(np.unique(items)) != len(items):
        raise ValueError("Candidate pool must contain unique items")
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(members),) or not np.isfinite(weights).all():
        raise ValueError("Need one finite hybrid weight per component")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Hybrid weights must be nonnegative and sum to one")
    if not len(items):
        return []

    combined = np.zeros(len(items), dtype=float)
    for member, weight in zip(members, weights):
        scores = np.asarray(component_scores[member])[user]
        order = items[np.lexsort((items, -scores[items]))]
        ranked = rerank(order, scores, method, strength, k, profiles, user)
        if ranked:
            rank_scores = (len(ranked) - np.arange(len(ranked))) / len(ranked)
            positions = {item: index for index, item in enumerate(items)}
            for rank, item in enumerate(ranked):
                combined[positions[item]] += weight * rank_scores[rank]

    return items[np.lexsort((items, -combined))[:k]].tolist()


def _lists(strategy, candidate_rows, score_matrices, members, weights,
           users, method, strength, k, profiles):
    result = []
    for user in users:
        items = candidate_rows[user]
        if strategy == "rerank_components_then_aggregate":
            ranked = rank_aggregate(
                items, score_matrices, members, weights, method, strength,
                k, profiles, user,
            )
        elif strategy == "aggregate_then_rerank":
            ranked = rerank(items, score_matrices["__hybrid__"][user], method,
                            strength, k, profiles, user)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        result.append(ranked)
    return result


def run(hybrids=HYBRIDS, strengths=DEFAULT_STRENGTHS, k=10,
        candidate_count=100, max_loss=.05, seed=2020, output=None):
    if not hybrids or len(set(hybrids)) != len(hybrids) or any(name not in HYBRIDS for name in hybrids):
        raise ValueError(f"Choose distinct supported weighted hybrids: {HYBRIDS}")
    if k < 1 or candidate_count < k or not 0 <= max_loss <= 1:
        raise ValueError("Need 0 < K <= candidates and a loss budget between 0 and 1")
    if not strengths or any(not np.isfinite(value) or not 0 <= value <= 1 for value in strengths):
        raise ValueError("Strengths must be finite and between 0 and 1")
    strengths = sorted(set([0., *strengths]))
    output = Path(output) if output else PROJECT_ROOT / "results/task3_3"

    score_matrices, train, valid, test, genres, tune_users, manifest = prepare(
        hybrids, seed, include_components=True,
    )
    profiles = build_profiles(train, genres)
    test_users = np.flatnonzero(test.any(axis=1))
    test_users = test_users[test_users != 0]
    if not len(test_users):
        raise ValueError("No test users")

    validation_rows, selections, test_rows = [], [], []
    for hybrid in hybrids:
        hybrid_record = manifest["hybrids"][hybrid]
        if "weights" not in hybrid_record:
            raise ValueError(f"{hybrid} does not use weighted score aggregation")
        members = list(hybrid_record["weights"])
        weights = np.asarray([hybrid_record["weights"][name] for name in members], dtype=float)
        if any(name not in score_matrices for name in members):
            raise ValueError(f"Missing component scores required by {hybrid}")
        hybrid_scores = score_matrices[hybrid]
        component_scores = {name: score_matrices[name] for name in members}
        candidate_rows = candidates(hybrid_scores, train, candidate_count)
        np.savez_compressed(
            ARTIFACTS / f"{hybrid}-task3_3-validation-candidates.npz",
            items=candidate_rows, users=tune_users,
        )
        strategy_scores = {**component_scores, "__hybrid__": hybrid_scores}

        for strategy in STRATEGIES:
            for method in METHODS:
                trials = []
                for strength in strengths:
                    lists = _lists(
                        strategy, candidate_rows, strategy_scores, members, weights,
                        tune_users, method, strength, k, profiles,
                    )
                    metrics, _ = evaluate_lists(lists, tune_users, valid, profiles, k)
                    trials.append({
                        "hybrid": hybrid, "strategy": strategy, "method": method,
                        "strength": strength, **metrics,
                    })
                chosen = select_setting(trials, method, k, max_loss)
                validation_rows.extend(trials)
                selections.append({
                    "hybrid": hybrid, "strategy": strategy, "method": method,
                    "strength": chosen["strength"],
                })

        candidate_rows = candidates(hybrid_scores, train | valid, candidate_count)
        np.savez_compressed(
            ARTIFACTS / f"{hybrid}-task3_3-test-candidates.npz",
            items=candidate_rows, users=test_users,
        )
        ranking_data = {"users": test_users}
        for strategy in STRATEGIES:
            selected = [row for row in selections
                        if row["hybrid"] == hybrid and row["strategy"] == strategy
                        and row["strength"] > 0]
            for method in METHODS:
                settings = [{"strength": 0.}]
                settings.extend(row for row in selected if row["method"] == method)
                baseline = None
                for setting in settings:
                    lists = _lists(
                        strategy, candidate_rows, strategy_scores, members, weights,
                        test_users, method, setting["strength"], k, profiles,
                    )
                    metrics, _ = evaluate_lists(lists, test_users, test, profiles, k)
                    if setting["strength"] == 0:
                        baseline = metrics
                    row = {
                        "hybrid": hybrid, "strategy": strategy, "method": method,
                        "strength": setting["strength"], **metrics,
                    }
                    row.update({
                        f"delta_{name}": value - baseline[name]
                        for name, value in metrics.items()
                    })
                    test_rows.append(row)
                    padded = np.full((len(test_users), k), -1, dtype=np.int64)
                    for index, values in enumerate(lists):
                        padded[index, :len(values)] = values
                    ranking_data[f"{strategy}__{method}__{setting['strength']}"] = padded
        np.savez_compressed(ARTIFACTS / f"{hybrid}-task3_3-test-rankings.npz", **ranking_data)

    write_csv(output / "validation.csv", validation_rows)
    write_json(output / "selected.json", selections)
    write_csv(output / "test.csv", test_rows)
    manifest.update({
        "status": "complete",
        "task": "3.3",
        "hybrids": hybrids,
        "strategies": {
            "rerank_components_then_aggregate": (
                "Rerank each component on the common hybrid-derived candidate pool; "
                "convert its top-K list to normalized rank scores; aggregate with the "
                "hybrid's fitted component weights."
            ),
            "aggregate_then_rerank": (
                "Apply the existing weighted score hybrid, then rerank that hybrid's "
                "ranking on the same candidate pool."
            ),
        },
        "topk": k,
        "candidates": candidate_count,
        "strengths": strengths,
        "max_relative_validation_ndcg_loss": max_loss,
        "candidate_pool": "Top candidates from the unreranked hybrid; same pool for both strategies",
        "selection": "Per strategy and method, best validation objective within the NDCG budget",
        "rank_aggregation": (
            "Weighted sum of per-component normalized rank scores "
            "(top-K scores decrease linearly from 1 to 1/K); uses fitted hybrid weights."
        ),
        "test_users": len(test_users),
        "code_sha256": {
            str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((PROJECT_ROOT / "task3").glob("*.py"))
        },
    })
    write_json(output / "protocol.json", manifest)
    print(f"Saved Task 3.3 results to {output}", flush=True)
    return test_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrids", nargs="+", default=["TunedWeightedHybrid"],
                        choices=HYBRIDS)
    parser.add_argument("--strengths", type=float, nargs="+", default=DEFAULT_STRENGTHS)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--max-ndcg-loss", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args.hybrids, args.strengths, args.topk, args.candidates,
        args.max_ndcg_loss, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
