"""Analyze held-out Task 3.3 reranker effects across user and item groups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task1.io_utils import write_csv, write_json
from task1.model_registry import PROJECT_ROOT
from task2.metrics import accuracy_metrics
from task3.evaluate_hybrid_reranking import STRATEGIES
from task3.prepare import ARTIFACTS, prepare
from task3.rerankers import build_profiles, miscalibration, popularity_deviation


USER_GROUPS = ("low", "high")
ITEM_GROUPS = ("head", "tail")
ACCURACY_METRICS = ("Precision", "Recall", "NDCG", "MRR", "Hit")
OBJECTIVE_METRIC = {
    "diversity": "IntraListDiversity",
    "calibration": "Miscalibration",
    "user_fairness": "UPD",
    "item_fairness": "ExposureGap",
}
STRATEGY_LABELS = {
    "rerank_components_then_aggregate": "Rerank → Aggregate",
    "aggregate_then_rerank": "Aggregate → Rerank",
}
GROUP_COLORS = {"low": "#2878B5", "high": "#E1812C",
                "head": "#2878B5", "tail": "#E1812C"}
BOOTSTRAP_REPLICATES = 4000
CONFIDENCE_LEVEL = .95
matplotlib.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _ranking_key(strategy, method, strength):
    return f"{strategy}__{method}__{float(strength)}"


def _mean_ci(values, rng):
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError("Cannot calculate a group metric without observations")
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[indices].mean(axis=1)
    alpha = 1 - CONFIDENCE_LEVEL
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _paired_mean_delta_ci(baseline, selected, rng):
    differences = np.asarray(selected, dtype=float) - np.asarray(baseline, dtype=float)
    return _mean_ci(differences, rng)


def _paired_statistic_delta_ci(baseline, selected, statistic, rng):
    baseline, selected = np.asarray(baseline), np.asarray(selected)
    if baseline.shape != selected.shape or not len(baseline):
        raise ValueError("Paired group data must have aligned nonempty rows")
    draws = rng.integers(
        0, len(baseline), size=(BOOTSTRAP_REPLICATES, len(baseline)),
    )
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index, sample in enumerate(draws):
        deltas[index] = statistic(selected[sample]) - statistic(baseline[sample])
    alpha = 1 - CONFIDENCE_LEVEL
    return (
        float(np.quantile(deltas, alpha / 2)),
        float(np.quantile(deltas, 1 - alpha / 2)),
    )


def _per_user_metrics(lists, users, test, profiles, k):
    relevant = [set(np.flatnonzero(test[user])) - {0} for user in users]
    rows = []
    genre_features = np.asarray(profiles["genres"], dtype=float)
    norms = np.linalg.norm(genre_features, axis=1)
    normalized_genres = np.divide(
        genre_features, norms[:, None], out=np.zeros_like(genre_features),
        where=norms[:, None] > 0,
    )
    for user, ranked, truth in zip(users, lists, relevant):
        row = [int(item) for item in ranked[:k] if int(item) > 0]
        accuracy = accuracy_metrics([row], [truth], k)
        if len(row) > 1:
            vectors = normalized_genres[row]
            distances = 1 - vectors @ vectors.T
            diversity = float(np.mean(distances[np.triu_indices(len(row), k=1)]))
        else:
            diversity = 0.
        genre_q = profiles["genre_dist"][row].mean(axis=0) if row else np.zeros_like(
            profiles["user_genres"][user],
        )
        pop_q = profiles["groups"][row].mean(axis=0) if row else np.zeros(2)
        discounts = 1 / np.log2(np.arange(len(row)) + 2)
        group_exposure = np.sum(
            profiles["groups"][row] * discounts[:, None], axis=0,
        ) if row else np.zeros(2)
        metrics = {
            name.removesuffix(f"@{k}"): value for name, value in accuracy.items()
        }
        metrics.update({
            "IntraListDiversity": diversity,
            "Miscalibration": float(miscalibration(profiles["user_genres"][user], genre_q)),
            "UPD": float(popularity_deviation(profiles["user_popularity"][user], pop_q)),
        })
        rows.append({
            "user_id": int(user),
            "metrics": metrics,
            "group_exposure": group_exposure,
        })
    return rows


def _exposure_gap(exposures, target):
    total = np.asarray(exposures, dtype=float).sum(axis=0)
    if total.sum() == 0:
        return 0.
    return float(popularity_deviation(target, total / total.sum()))


def _user_group_analysis(hybrid, selection, users, baseline_lists, rankings,
                         test, profiles, k, seed):
    strategy, method, strength = (
        selection["strategy"], selection["method"], float(selection["strength"]),
    )
    selected_lists = rankings[_ranking_key(strategy, method, strength)]
    baseline_users = _per_user_metrics(baseline_lists, users, test, profiles, k)
    selected_users = _per_user_metrics(selected_lists, users, test, profiles, k)
    median_activity = float(np.median(profiles["activity"][1:]))
    user_groups = {
        "low": np.asarray([profiles["activity"][user] <= median_activity for user in users]),
        "high": np.asarray([profiles["activity"][user] > median_activity for user in users]),
    }
    metrics = [*ACCURACY_METRICS]
    if method != "item_fairness":
        metrics.append(OBJECTIVE_METRIC[method])
    rows = []
    for group, mask in user_groups.items():
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        rng = np.random.default_rng(seed)
        base = [baseline_users[index]["metrics"] for index in indices]
        chosen = [selected_users[index]["metrics"] for index in indices]
        for metric in metrics:
            baseline_values = np.asarray([row[metric] for row in base], dtype=float)
            selected_values = np.asarray([row[metric] for row in chosen], dtype=float)
            low, high = _paired_mean_delta_ci(baseline_values, selected_values, rng)
            rows.append({
                "hybrid": hybrid, "strategy": strategy, "method": method,
                "strength": strength, "user_group": group, "users": len(indices),
                "comparison_baseline": "original_unreranked_hybrid",
                "group_definition": f"training interactions <= median ({median_activity:g})"
                if group == "low" else f"training interactions > median ({median_activity:g})",
                "metric": metric, "baseline": float(baseline_values.mean()),
                "reranked": float(selected_values.mean()),
                "delta": float(selected_values.mean() - baseline_values.mean()),
                "ci95_low": low, "ci95_high": high,
                "ci_method": "paired user bootstrap percentile interval",
            })

        if method == "item_fairness":
            baseline_exposure = np.asarray(
                [baseline_users[index]["group_exposure"] for index in indices],
            )
            selected_exposure = np.asarray(
                [selected_users[index]["group_exposure"] for index in indices],
            )
            target = profiles["target_exposure"]
            stat = lambda rows: _exposure_gap(rows, target)
            baseline_gap, selected_gap = stat(baseline_exposure), stat(selected_exposure)
            low, high = _paired_statistic_delta_ci(
                baseline_exposure, selected_exposure, stat, rng,
            )
            rows.append({
                "hybrid": hybrid, "strategy": strategy, "method": method,
                "strength": strength, "user_group": group, "users": len(indices),
                "comparison_baseline": "original_unreranked_hybrid",
                "group_definition": f"training interactions <= median ({median_activity:g})"
                if group == "low" else f"training interactions > median ({median_activity:g})",
                "metric": "ExposureGap", "baseline": baseline_gap,
                "reranked": selected_gap, "delta": selected_gap - baseline_gap,
                "ci95_low": low, "ci95_high": high,
                "ci_method": "paired user bootstrap percentile interval",
            })
    return rows


def _per_user_item_counts(lists, users, test, item_masks, k):
    rows = []
    for user, ranking in zip(users, lists):
        truth = set(np.flatnonzero(test[user])) - {0}
        values = {}
        for group in ITEM_GROUPS:
            recommendations = [item for item in ranking[:k] if item > 0 and item_masks[group][item]]
            relevant = [item for item in truth if item_masks[group][item]]
            hits = sum(item in truth for item in recommendations)
            values[group] = {
                "recommendations": len(recommendations),
                "relevant": len(relevant),
                "hits": hits,
                "hit_rate": float(hits > 0),
                "exposure": sum(
                    1 / np.log2(rank + 2)
                    for rank, item in enumerate(ranking[:k])
                    if item > 0 and item_masks[group][item]
                ),
            }
        rows.append(values)
    return rows


def _item_statistic(rows, group, metric):
    counts = [row[group] for row in rows]
    if metric == "recall":
        numerator = sum(row["hits"] for row in counts)
        denominator = sum(row["relevant"] for row in counts)
        return numerator / denominator if denominator else 0.
    if metric == "precision":
        numerator = sum(row["hits"] for row in counts)
        denominator = sum(row["recommendations"] for row in counts)
        return numerator / denominator if denominator else 0.
    if metric == "hit_rate":
        return float(np.mean([row["hit_rate"] for row in counts]))
    if metric == "exposure_share":
        exposure = np.asarray([
            [row[item_group]["exposure"] for item_group in ITEM_GROUPS]
            for row in rows
        ])
        total = exposure.sum()
        if total == 0:
            return 0.
        return float(exposure[:, ITEM_GROUPS.index(group)].sum() / total)
    raise ValueError(f"Unknown item group metric: {metric}")


def _item_group_analysis(hybrid, selection, users, baseline_lists, rankings, test, profiles, k,
                         catalog_sizes, seed):
    strategy, method, strength = (
        selection["strategy"], selection["method"], float(selection["strength"]),
    )
    selected_lists = rankings[_ranking_key(strategy, method, strength)]
    item_masks = {
        "head": profiles["groups"][:, 0] > 0,
        "tail": profiles["groups"][:, 1] > 0,
    }
    baseline = _per_user_item_counts(baseline_lists, users, test, item_masks, k)
    reranked = _per_user_item_counts(selected_lists, users, test, item_masks, k)
    metrics = ("exposure_share", "recall", "precision", "hit_rate")
    rows = []
    for group in ITEM_GROUPS:
        for metric in metrics:
            base_value = _item_statistic(baseline, group, metric)
            selected_value = _item_statistic(reranked, group, metric)
            rng = np.random.default_rng(seed)
            draws = rng.integers(0, len(users), size=(BOOTSTRAP_REPLICATES, len(users)))
            deltas = np.empty(BOOTSTRAP_REPLICATES)
            for index, sample in enumerate(draws):
                sample_base = [baseline[user_index] for user_index in sample]
                sample_selected = [reranked[user_index] for user_index in sample]
                deltas[index] = (
                    _item_statistic(sample_selected, group, metric)
                    - _item_statistic(sample_base, group, metric)
                )
            alpha = 1 - CONFIDENCE_LEVEL
            rows.append({
                "hybrid": hybrid, "strategy": strategy, "method": method,
                "strength": strength, "item_group": group,
                "comparison_baseline": "original_unreranked_hybrid",
                "catalog_items": catalog_sizes[group], "users": len(users),
                "test_relevant_items": sum(item[group]["relevant"] for item in baseline),
                "metric": metric, "baseline": base_value,
                "reranked": selected_value, "delta": selected_value - base_value,
                "ci95_low": float(np.quantile(deltas, alpha / 2)),
                "ci95_high": float(np.quantile(deltas, 1 - alpha / 2)),
                "ci_method": "paired user bootstrap percentile interval",
            })
    return rows


def _bar_delta(ax, rows, groups, show_ci=True):
    values = [next(row for row in rows if row["group"] == group) for group in groups]
    deltas = [row["delta"] for row in values]
    labels = [group.title() for group in groups]
    colors = [GROUP_COLORS[group] for group in groups]
    yerr = None
    if show_ci:
        yerr = np.asarray([
            [max(0, row["delta"] - row["ci95_low"]) for row in values],
            [max(0, row["ci95_high"] - row["delta"]) for row in values],
        ])
    bars = ax.bar(
        np.arange(len(groups)), deltas, color=colors, width=.65,
        yerr=yerr, capsize=3 if show_ci else 0,
        error_kw={"linewidth": 1},
    )
    ax.set_xticks(np.arange(len(groups)), labels)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.bar_label(bars, fmt="%+.3f", padding=3, fontsize=9)
    ax.grid(axis="y", alpha=.2)


def _plot_user_metrics(directory, rows, hybrids, k, user_sizes, median_activity, include_svg):
    for hybrid in hybrids:
        subset = [row for row in rows if row["hybrid"] == hybrid]
        for metric, filename, caption, title in (
            ("accuracy", "user-group-accuracy",
             "Figure 2 — Impact of Reranking on Recommendation Accuracy Across User Activity Groups",
             f"Change in NDCG@{k}"),
            (None, "user-group-objectives",
             "Figure 3 — Impact of Reranking on Quality Objectives Across User Activity Groups",
             "Change in intended reranking objective"),
        ):
            fig, axes = plt.subplots(
                2, 4, figsize=(16, 9), layout="constrained",
                sharey=True if metric else "col",
            )
            for strategy_index, strategy in enumerate(STRATEGIES):
                for method_index, method in enumerate(OBJECTIVE_METRIC):
                    ax = axes[strategy_index, method_index]
                    selected_metric = "NDCG" if metric else OBJECTIVE_METRIC[method]
                    group_rows = [
                        {
                            **row,
                            "group": row["user_group"],
                        }
                        for row in subset
                        if row["strategy"] == strategy and row["method"] == method
                        and row["metric"] == selected_metric
                    ]
                    if not group_rows:
                        ax.set_visible(False)
                        continue
                    _bar_delta(ax, group_rows, USER_GROUPS)
                    if strategy_index == 0:
                        if metric:
                            panel_title = method.replace("_", " ").title()
                        else:
                            objective_metric = OBJECTIVE_METRIC[method]
                            direction = "higher is better" if method == "diversity" else "lower is better"
                            panel_title = (
                                f"{method.replace('_', ' ').title()}\n"
                                f"{objective_metric} ({direction})"
                            )
                        ax.set_title(panel_title, fontsize=11)
                    if method_index == 0:
                        ax.set_ylabel(
                            f"{STRATEGY_LABELS[strategy]}\n{title}\n"
                            f"Δ {selected_metric} (95% paired bootstrap CI)",
                            fontsize=10,
                        )
                    else:
                        ax.set_ylabel("")
                    ax.tick_params(labelsize=9)
            fig.suptitle(
                f"{caption}\n{hybrid} | {title}\n"
                f"Training activity ≤{median_activity:g}: low n={user_sizes['low']:,}; "
                f"> {median_activity:g}: high n={user_sizes['high']:,}; "
                + ("higher is better" if metric else "objective direction is stated per panel"),
                fontsize=14,
            )
            fig.savefig(directory / f"Task3.4-{hybrid}-{filename}.png", dpi=200)
            if include_svg:
                fig.savefig(directory / f"Task3.4-{hybrid}-{filename}.svg")
            plt.close(fig)


def _plot_item_exposure(directory, rows, hybrids, catalog_sizes, include_svg):
    metric = "exposure_share"
    for hybrid in hybrids:
        subset = [row for row in rows if row["hybrid"] == hybrid and row["metric"] == metric]
        if not subset:
            continue
        fig, axes = plt.subplots(2, 4, figsize=(16, 9), layout="constrained", sharey=True)
        for strategy_index, strategy in enumerate(STRATEGIES):
            for method_index, method in enumerate(OBJECTIVE_METRIC):
                ax = axes[strategy_index, method_index]
                group_rows = [
                    {**row, "group": row["item_group"]}
                    for row in subset
                    if row["strategy"] == strategy and row["method"] == method
                ]
                if not group_rows:
                    ax.set_visible(False)
                    continue
                _bar_delta(ax, group_rows, ITEM_GROUPS)
                if strategy_index == 0:
                    ax.set_title(method.replace("_", " ").title(), fontsize=12)
                if method_index == 0:
                    ax.set_ylabel(
                        f"{STRATEGY_LABELS[strategy]}\n"
                        "Δ discounted exposure share\n"
                        "(95% paired bootstrap CI)",
                        fontsize=10,
                    )
                ax.tick_params(labelsize=9)
        fig.suptitle(
            "Figure 4 — Impact of Reranking on Recommendation Exposure Across Item Popularity Groups\n"
            f"{hybrid} | Change in discounted exposure share (positive = more exposure; "
            "not evidence of improved relevance)\n"
            f"Training-popularity head n={catalog_sizes['head']:,} items, "
            f"tail n={catalog_sizes['tail']:,}",
            fontsize=14,
        )
        fig.savefig(directory / f"Task3.4-{hybrid}-item-group-exposure.png", dpi=200)
        if include_svg:
            fig.savefig(directory / f"Task3.4-{hybrid}-item-group-exposure.svg")
        plt.close(fig)


def _write_group_protocol(directory, hybrids, k, users, user_sizes,
                          catalog_sizes, median_activity, test_relevant):
    write_json(directory / "group_analysis.json", {
        "task": "3.4",
        "hybrids": hybrids,
        "evaluation": "Held-out test only; validation-selected strengths are reused without group-specific retuning",
        "baseline": "Original hybrid ranking: top K from the hybrid-derived test candidate pool, before reranking",
        "user_groups": {
            "definition": "Low = training interactions <= median over nonpadding users; high = > median",
            "training_activity_median": median_activity,
            "sizes": user_sizes,
        },
        "item_groups": {
            "definition": "Head = top ceil(20%) of catalog by training interaction count; tail = remaining items",
            "catalog_item_counts": catalog_sizes,
            "test_relevant_item_counts": test_relevant,
        },
        "user_group_metrics": (
            "Macro per-user Precision, Recall, NDCG, MRR, Hit plus the objective targeted by that reranker; "
            "item fairness also reports group-level head/tail exposure gap."
        ),
        "item_group_metrics": (
            "Discounted exposure share, group-specific recall (hits/relevant test items), precision "
            "(hits/recommendations), and hit rate (fraction of test users with a hit in the group). "
            "Exposure is distribution, not recommendation quality."
        ),
        "uncertainty": {
            "method": "paired user bootstrap percentile confidence interval",
            "confidence_level": CONFIDENCE_LEVEL,
            "replicates": BOOTSTRAP_REPLICATES,
            "interpretation": "Intervals describe uncertainty in the observed test-user sample; no significance claim is made.",
        },
        "topk": k,
        "test_users": len(users),
    })


def run(directory=None, hybrids=None, include_svg=False):
    directory = Path(directory) if directory else PROJECT_ROOT / "results/task3_3"
    protocol = _read_json(directory / "protocol.json")
    selections = _read_json(directory / "selected.json")
    hybrids = list(hybrids or protocol["hybrids"])
    unknown = set(hybrids) - set(protocol["hybrids"])
    if unknown:
        raise ValueError(f"No Task 3.3 results for requested hybrid(s): {sorted(unknown)}")

    _, train, _, test, genres, _, _ = prepare(hybrids, int(protocol["seed"]))
    profiles = build_profiles(train, genres)
    k = int(protocol["topk"])
    all_users = np.flatnonzero(test.any(axis=1))
    all_users = all_users[all_users != 0]
    user_rows, item_rows = [], []
    median_activity = float(np.median(profiles["activity"][1:]))
    user_sizes = {
        group: int(sum(
            profiles["activity"][user] <= median_activity if group == "low"
            else profiles["activity"][user] > median_activity
            for user in all_users
        ))
        for group in USER_GROUPS
    }
    catalog_sizes = {
        "head": int(np.sum(profiles["groups"][1:, 0] > 0)),
        "tail": int(np.sum(profiles["groups"][1:, 1] > 0)),
    }
    test_relevant = {
        group: int(sum(
            1
            for user in all_users
            for item in np.flatnonzero(test[user])
            if profiles["groups"][item, 0 if group == "head" else 1] > 0
        ))
        for group in ITEM_GROUPS
    }

    for hybrid in hybrids:
        ranking_path = ARTIFACTS / f"{hybrid}-task3_3-test-rankings.npz"
        candidate_path = ARTIFACTS / f"{hybrid}-task3_3-test-candidates.npz"
        with np.load(ranking_path, allow_pickle=False) as archive:
            users = archive["users"].astype(int)
            rankings = {name: archive[name] for name in archive.files if name != "users"}
        with np.load(candidate_path, allow_pickle=False) as archive:
            candidate_users = archive["users"].astype(int)
            candidate_items = archive["items"]
        if not np.array_equal(users, all_users):
            raise ValueError(f"Saved rankings for {hybrid} do not match current held-out test users")
        if (
            not np.array_equal(candidate_users, users)
            or candidate_items.ndim != 2
            or candidate_items.shape[0] <= int(users.max())
        ):
            raise ValueError(f"Saved candidate pool for {hybrid} does not match held-out test users")
        baseline_lists = candidate_items[users, :k]
        hybrid_selections = [row for row in selections if row["hybrid"] == hybrid]
        if not hybrid_selections:
            raise ValueError(f"No validation-selected settings found for {hybrid}")
        for selection_index, selection in enumerate(hybrid_selections):
            seed = int(protocol["seed"]) + selection_index
            user_rows.extend(_user_group_analysis(
                hybrid, selection, users, baseline_lists, rankings,
                test, profiles, k, seed,
            ))
            item_rows.extend(_item_group_analysis(
                hybrid, selection, users, baseline_lists, rankings, test, profiles, k,
                catalog_sizes, seed,
            ))

    write_csv(directory / "user_groups.csv", user_rows)
    write_csv(directory / "item_groups.csv", item_rows)
    _write_group_protocol(
        directory, hybrids, k, all_users, user_sizes, catalog_sizes,
        median_activity, test_relevant,
    )
    _plot_user_metrics(
        directory, user_rows, hybrids, k, user_sizes, median_activity, include_svg,
    )
    _plot_item_exposure(directory, item_rows, hybrids, catalog_sizes, include_svg)
    print(f"Saved Task 3.4 group analysis to {directory}", flush=True)
    return user_rows, item_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=PROJECT_ROOT / "results/task3_3")
    parser.add_argument("--hybrids", nargs="+")
    parser.add_argument("--svg", action="store_true", help="Also export SVG versions")
    args = parser.parse_args()
    run(args.directory, args.hybrids, args.svg)
