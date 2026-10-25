"""Tasks 2.5/2.6: training-defined groups and a validation-selected user gate."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import numpy as np

from task1.io_utils import write_csv, write_json
from task2.metrics import accuracy_metrics, beyond_accuracy_metrics


def define_groups(train):
    """Use training only; padding belongs to no group, ties stay together."""
    train = np.asarray(train, dtype=bool)
    activity, popularity = train.sum(axis=1), train.sum(axis=0)
    active = np.flatnonzero(activity > 0)
    active = active[active != 0]
    if not len(active):
        raise ValueError("Need nonpadding users with training history")
    median = float(np.median(activity[active]))
    catalog = np.arange(1, train.shape[1])
    order = catalog[np.lexsort((catalog, -popularity[1:]))]
    head = np.zeros(train.shape[1], dtype=bool)
    head[order[:max(1, int(np.ceil(.2 * len(catalog))))]] = True
    affinity = np.divide(train[:, head].sum(axis=1), activity,
                         out=np.zeros(len(activity)), where=activity > 0)
    affinity_median = float(np.median(affinity[active]))
    eligible = np.zeros(len(activity), dtype=bool)
    eligible[active] = True
    user_groups = {
        "activity_low": eligible & (activity <= median),
        "activity_high": eligible & (activity > median),
        "head_affinity_low": eligible & (affinity <= affinity_median),
        "head_affinity_high": eligible & (affinity > affinity_median),
    }
    tail = ~head
    tail[0] = False
    cold = popularity == 0
    cold[0] = False
    # Four disjoint cells allow a gate to combine activity and preference.
    cells = (activity > median).astype(int) * 2 + (affinity > affinity_median)
    cells[~eligible] = -1
    return user_groups, {"head": head, "tail": tail, "cold": cold}, cells, {
        "activity_median": median, "head_affinity_median": affinity_median,
        "head_items": int(head.sum()), "tail_items": int(tail.sum()),
        "cold_items": int(cold.sum()),
        "head_rule": "top ceil(20% of catalog) training counts; ties by internal ID",
        "cold_rule": "zero training interactions; subset of tail, not a third disjoint group",
    }


def lists_for(rankings, users):
    return [[int(i) for i in rankings[u] if i > 0] for u in users]


def user_values(rankings, users, truth, popularity, genres, k):
    """Keep per-user values for paired uncertainty and inspectable raw output."""
    rows = []
    for user, rec in zip(users, lists_for(rankings, users)):
        relevant = set(np.flatnonzero(truth[user])) - {0}
        row = accuracy_metrics([rec], [relevant], k)
        row.update(beyond_accuracy_metrics([rec], popularity, genres, k))
        row.pop(f"Coverage@{k}")  # coverage is a group statistic, not a user mean
        rows.append(row)
    return rows


def paired_ci(differences, seed=2020, replicates=2000):
    differences = np.asarray(differences, dtype=float)
    if not len(differences):
        return None, None
    rng = np.random.default_rng(seed)
    means = np.concatenate([
        rng.choice(differences, (min(100, replicates - start), len(differences))).mean(axis=1)
        for start in range(0, replicates, 100)])
    return tuple(map(float, np.quantile(means, [.025, .975])))


def group_tables(rankings, train, truth, genres, k, baseline, seed=2020):
    groups, item_groups, _, definitions = define_groups(train)
    users = np.flatnonzero(truth.any(axis=1))
    users = users[users != 0]
    popularity = train.sum(axis=0)
    values = {name: user_values(r, users, truth, popularity, genres, k)
              for name, r in rankings.items()}
    raw, user_rows, item_rows, comparisons = [], [], [], []
    groups = {"all": np.arange(train.shape[0]) > 0, **groups}
    for name, ranking in rankings.items():
        for user, metrics in zip(users, values[name]):
            raw.append({"model": name, "user_id": int(user), **metrics})
        for group, mask in groups.items():
            positions = np.flatnonzero(mask[users])
            if not len(positions):
                continue
            selected_users = users[positions]
            recs = lists_for(ranking, selected_users)
            metrics = {key: float(np.mean([values[name][p][key] for p in positions]))
                       for key in values[name][0]}
            metrics[f"Coverage@{k}"] = len({i for row in recs for i in row}) / (train.shape[1] - 1)
            user_rows.append({"model": name, "group": group, "users": len(positions), **metrics})
            for reference in dict.fromkeys(["Pop", baseline]):
                if reference not in values or reference == name:
                    continue
                for metric in (f"MRR@{k}", f"NDCG@{k}", f"Novelty@{k}", f"IntraListDiversity@{k}"):
                    delta = np.array([values[name][p][metric] - values[reference][p][metric]
                                      for p in positions])
                    low, high = paired_ci(delta, seed)
                    comparisons.append({"model": name, "reference": reference, "group": group,
                                        "metric": metric, "delta": float(delta.mean()),
                                        "ci95_low": low, "ci95_high": high, "users": len(positions)})
        recs = lists_for(ranking, users)
        total_exposure = sum(sum(1 / np.log2(r + 2) for r in range(len(row))) for row in recs)
        for group, mask in item_groups.items():
            hits = relevant_count = recommended_count = 0
            exposure = 0.
            recalls, ndcgs = [], []
            unique = set()
            for user, row in zip(users, recs):
                relevant = set(np.flatnonzero(truth[user] & mask))
                selected = [i for i in row if mask[i]]
                found = sum(i in relevant for i in row)
                hits += found
                relevant_count += len(relevant)
                recommended_count += len(selected)
                unique.update(selected)
                exposure += sum(1 / np.log2(r + 2) for r, i in enumerate(row) if mask[i])
                if relevant:
                    recalls.append(found / len(relevant))
                    dcg = sum(1 / np.log2(r + 2) for r, i in enumerate(row) if i in relevant)
                    ideal = sum(1 / np.log2(r + 2) for r in range(min(k, len(relevant))))
                    ndcgs.append(dcg / ideal)
            item_rows.append({"model": name, "group": group, "items": int(mask.sum()),
                              "users_with_relevant": len(recalls), "relevant_interactions": relevant_count,
                              "hits": hits, "recommended_slots": recommended_count,
                              "micro_recall": hits / relevant_count if relevant_count else None,
                              "macro_recall_eligible": float(np.mean(recalls)) if recalls else None,
                              "ndcg_original_ranks_eligible": float(np.mean(ndcgs)) if ndcgs else None,
                              "slot_precision": hits / recommended_count if recommended_count else None,
                              "exposure_share": exposure / total_exposure if total_exposure else 0.,
                              "catalog_share": float(mask[1:].mean()),
                              "within_group_coverage": len(unique) / mask.sum() if mask.any() else None,
                              "novelty_recommended": float(np.mean(-np.log2((popularity[list(unique)] + 1) /
                                  (popularity[1:].sum() + len(popularity) - 1)))) if unique else None})
    return user_rows, item_rows, comparisons, raw, definitions


def fit_gate(validation_rankings, valid, fit_users, cells, k, min_users=30, margin=.005):
    """Fit a small user gate without test labels; endpoint fallback avoids forced mixing."""
    names = sorted(validation_rankings)
    if not names or not len(fit_users):
        raise ValueError("Need candidates and validation users")
    utilities = {}
    for name in names:
        utilities[name] = np.array([
            accuracy_metrics([lists_for(validation_rankings[name], [u])[0]],
                             [set(np.flatnonzero(valid[u])) - {0}], k)[f"MRR@{k}"]
            for u in fit_users])
    global_best = max(names, key=lambda n: utilities[n].mean())
    choices, rows = {}, []
    for cell in sorted(set(cells[fit_users])):
        selected = cells[fit_users] == cell
        best = max(names, key=lambda n: utilities[n][selected].mean())
        gain = float((utilities[best][selected] - utilities[global_best][selected]).mean())
        chosen = best if selected.sum() >= min_users and gain > margin else global_best
        choices[int(cell)] = chosen
        for name in names:
            rows.append({"cell": int(cell), "model": name, "users": int(selected.sum()),
                         "validation_mrr": float(utilities[name][selected].mean()),
                         "selected": name == chosen})
    return global_best, choices, rows


def apply_gate(rankings, cells, global_best, choices):
    result = rankings[global_best].copy()
    for cell, name in choices.items():
        result[cells == cell] = rankings[name][cells == cell]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/task2_groups"))
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1,
                        help="CPU tensor threads; one avoids overhead on MovieLens 100K")
    args = parser.parse_args()
    if args.topk < 1:
        parser.error("topk must be positive")
    # Lazy import keeps the statistical helpers testable without RecBole.
    from task1.model_registry import MODEL_SPECS
    from task2.recommendations import rank_scores
    from task3.prepare import prepare, SCORE_HYBRIDS
    import torch
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)

    models = args.models or [*MODEL_SPECS, *SCORE_HYBRIDS]
    scores, train, valid, test, genres, tune_users, manifest = prepare(models, 2020)
    user_masks, item_masks, cells, definitions = define_groups(train)
    valid_rankings = {n: rank_scores(s, train, args.topk) for n, s in scores.items()}
    # Base settings were selected earlier; gate fitting has no access to test labels.
    candidates = {n: r for n, r in valid_rankings.items() if n not in ("Random", "Pop")}
    if not candidates:
        candidates = valid_rankings
    baseline, choices, selection = fit_gate(candidates, valid, tune_users, cells, args.topk)
    # Cross-fit the gate choices to reveal fitting optimism on development users.
    folds = np.array_split(np.random.default_rng(2020).permutation(tune_users), 3)
    crossfit = candidates[baseline].copy()
    fold_records = []
    for fold, held in enumerate(folds):
        fitting = np.setdiff1d(tune_users, held)
        fallback, local_choices, _ = fit_gate(candidates, valid, fitting, cells, args.topk)
        crossfit[held] = apply_gate(candidates, cells, fallback, local_choices)[held]
        fold_records.append({"fold": fold, "fit_users": len(fitting), "held_users": len(held),
                             "fallback": fallback, "choices": local_choices})
    dev_truth = [set(np.flatnonzero(valid[u])) - {0} for u in tune_users]
    development = {n: accuracy_metrics(lists_for(r, tune_users), dev_truth, args.topk)
                   for n, r in {**candidates, "GroupAwareHybrid_crossfit": crossfit}.items()}
    rankings = {n: rank_scores(s, train | valid, args.topk) for n, s in scores.items()}
    rankings["GroupAwareHybrid"] = apply_gate(rankings, cells, baseline, choices)
    user_rows, item_rows, comparisons, raw, _ = group_tables(
        rankings, train, test, genres, args.topk, baseline)
    directory = args.output_dir
    activity = train.sum(axis=1)
    write_csv(directory / "user_membership.csv", [
        {"user_id": u, "training_interactions": int(activity[u]), "gate_cell": int(cells[u]),
         "head_affinity": float(train[u, item_masks['head']].sum() / activity[u]) if activity[u] else None,
         **{name: bool(mask[u]) for name, mask in user_masks.items()}}
        for u in range(1, train.shape[0])])
    write_csv(directory / "item_membership.csv", [
        {"item_id": i, "training_interactions": int(train[:, i].sum()),
         **{name: bool(mask[i]) for name, mask in item_masks.items()}}
        for i in range(1, train.shape[1])])
    for filename, rows in (("user_groups.csv", user_rows), ("item_groups.csv", item_rows),
                           ("paired_deltas.csv", comparisons), ("per_user.csv", raw),
                           ("gate_selection.csv", selection)):
        write_csv(directory / filename, rows)
    write_json(directory / "protocol.json", {
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "repository_base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "cpu_tensor_threads": args.threads,
        "source": manifest, "groups": definitions, "topk": args.topk,
        "baseline_selected_on_validation": baseline, "gate_choices": choices,
        "gate_min_users": 30, "gate_min_mrr_gain": .005,
        "gate_cells": "2 * (activity > training median) + (head affinity > training median)",
        "gate_validation_users": list(map(int, tune_users)), "gate_crossfit": fold_records,
        "development_accuracy": development,
        "evaluation": "test; train and validation history masked; full catalog; no rerankers",
        "uncertainty": "2000 paired user bootstrap samples, seed 2020; descriptive 95% intervals; no multiple-comparison correction",
        "item_metrics": "Relevant-item group filtering preserves original list ranks. Recall excludes users without relevant group items. Slot precision divides by group recommendation slots; undefined ratios are blank. Novelty is averaged over unique recommended group items.",
        "limitations": ["One split/seed; observational groups, not causal or demographic fairness claims.",
                        "Group coverage depends on user-group size; interpret alongside counts.",
                        "Base tuning and hybrid structure used full validation; gate cross-fit is not nested model selection.",
                        "Weighted hybrids are refitted on the reserved half of validation users via task3.prepare; values can differ from Task 2.2.",
                        "Random uses seeded full-score batches; ties/random batching can differ from earlier tables.",
                        "Training-inactive users are excluded from user subgroup tables; they remain in all-user evaluation."]})
    plot_results(directory, user_rows, item_rows, args.topk)
    write_findings(directory, user_rows, item_rows, baseline, choices, development, args.topk)
    print(f"Saved Tasks 2.5/2.6 to {directory}", flush=True)


def plot_results(directory, users, items, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(row["model"] for row in users))
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    for ax, metric in zip(axes[:2], (f"MRR@{k}", f"Novelty@{k}")):
        for offset, group, color in ((-.18, "activity_low", "#2878B5"), (.18, "activity_high", "#E1812C")):
            lookup = {r["model"]: r[metric] for r in users if r["group"] == group}
            ax.barh(np.arange(len(names)) + offset, [lookup.get(n, 0) for n in names],
                    height=.35, label=group, color=color)
        ax.set_yticks(np.arange(len(names)), names)
        ax.set_xlabel(metric)
        ax.legend(fontsize=8)
    for offset, group, color in ((-.18, "head", "#2878B5"), (.18, "tail", "#E1812C")):
        lookup = {r["model"]: r["micro_recall"] for r in items if r["group"] == group}
        axes[2].barh(np.arange(len(names)) + offset, [lookup.get(n) or 0 for n in names],
                     height=.35, label=group, color=color)
    axes[2].set_yticks(np.arange(len(names)), names)
    axes[2].set_xlabel(f"Item-group micro recall@{k}")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "group_comparison.png", dpi=180)
    plt.close(fig)


def write_findings(directory, users, items, baseline, choices, development, k):
    lines = ["# Task 2.5: Group performance", "",
             "Groups use training interactions only. All models rank the same unseen full catalog; "
             "the CSVs include all five accuracy measures, novelty, diversity, popularity and coverage. "
             "Paired bootstrap intervals compare aligned users against Pop and the validation-selected reference. "
             "Item metrics retain original top-K positions and report relevance and exposure separately.", ""]
    for group in ("activity_low", "activity_high", "head_affinity_low", "head_affinity_high"):
        rows = [r for r in users if r["group"] == group]
        if not rows:
            continue
        best = max(rows, key=lambda r: r[f"MRR@{k}"])
        novel = max(rows, key=lambda r: r[f"Novelty@{k}"])
        lines.append(f"- {group} ({best['users']} users): {best['model']} has the highest observed MRR "
                     f"({best[f'MRR@{k}']:.4f}); {novel['model']} has the highest novelty "
                     f"({novel[f'Novelty@{k}']:.2f} bits). Test winners are descriptive, not selection rules.")
    reference_users = {r['group']: r for r in users if r['model'] == baseline}
    reference_items = {r['group']: r for r in items if r['model'] == baseline}
    low, high = reference_users.get('activity_low'), reference_users.get('activity_high')
    if low and high:
        lines += ["", f"For {baseline}, low/high-activity MRR is {low[f'MRR@{k}']:.4f}/"
                  f"{high[f'MRR@{k}']:.4f}, but recall is {low[f'Recall@{k}']:.4f}/"
                  f"{high[f'Recall@{k}']:.4f}. The reversal is consistent with more relevant "
                  "opportunities but larger recall denominators for active users; MRR alone "
                  "would conceal their lower fraction of recovered relevant items."]
    head, tail = reference_items['head'], reference_items['tail']
    if head['micro_recall'] is not None and tail['micro_recall'] is not None:
        lines += ["", f"{baseline} assigns {100*head['exposure_share']:.2f}% of discounted exposure "
                  f"to the head ({100*head['catalog_share']:.1f}% of items). Head/tail micro recall is "
                  f"{head['micro_recall']:.4f}/{tail['micro_recall']:.4f}. The collaborative ensemble "
                  "therefore retains a strong popularity concentration: combining several interaction "
                  "models does not by itself recover the tail. Tail slot precision can still be high "
                  "because its denominator contains very few recommendations; inspect counts before "
                  "interpreting that ratio as broad tail success."]
    lines += ["", "# Task 2.6: Group-aware hybrid", "",
              "Hypothesis: activity and preference for head items identify users who benefit from different "
              "components. A four-cell switch selects complete model rankings using validation MRR, with a "
              "global-model fallback for cells with fewer than 30 users or gains at most 0.005. "
              "It includes single-model endpoints so a lack of complementarity need not force a hybrid.", "",
              f"Validation-selected global reference: **{baseline}**. Cell choices: `{choices}`.", "",
              f"Cross-fitted gate development MRR: {development['GroupAwareHybrid_crossfit'][f'MRR@{k}']:.4f}; "
              f"reference development MRR: {development[baseline][f'MRR@{k}']:.4f}. "
              "The global reference uses all gate development users, so this is not a fully nested comparison.", ""]
    all_rows = {r["model"]: r for r in users if r["group"] == "all"}
    a, b = all_rows["GroupAwareHybrid"], all_rows[baseline]
    lines += ["| Test metric | Reference | Group-aware hybrid | Change |",
              "|---|---:|---:|---:|"]
    for metric in [f"MRR@{k}", f"NDCG@{k}", f"Novelty@{k}", f"IntraListDiversity@{k}", f"Coverage@{k}"]:
        lines.append(f"| {metric} | {b[metric]:.4f} | {a[metric]:.4f} | {a[metric]-b[metric]:+.4f} |")
    lines += ["", "Superiority requires consistent accuracy gains without unacceptable losses in other dimensions; "
              "a positive point estimate alone does not establish it. If the gate collapses to one endpoint, "
              "these groups do not support the proposed specialization. Inspect paired_deltas.csv for uncertainty.", "",
              "Mechanism checks: compare activity-group recall with MRR because users with larger held-out histories "
              "have more chances for a hit but a larger recall denominator. Compare head/tail recall with exposure "
              "and catalog share to distinguish accurate retrieval from simply recommending popular items. "
              "Higher novelty is not evidence of relevance; Random is the control for that distinction. "
              "Observed group differences motivate hypotheses, not causal explanations.", "",
              "Limits: one seed, small or empty cold-item strata, reused base-model validation, unequal group sizes, "
              "and implicit relevance from observed ratings. No demographic fairness claim is supported. "
              "The gate cross-fit addresses gate-fitting optimism only. Repeat frozen choices on fresh splits "
              "before claiming a novel superior hybrid.", ""]
    if a[f'MRR@{k}'] <= b[f'MRR@{k}']:
        lines += [f"In this run the gate loses {b[f'MRR@{k}']-a[f'MRR@{k}']:.4f} test MRR "
                  "while slightly changing beyond-accuracy values. Its cross-fit result also fails to "
                  "support adopting the rule. Retain the validation-selected reference rather than "
                  "promoting this candidate. The next testable hypothesis is a tail-focused mixture "
                  "with genuinely different content evidence and a validation accuracy constraint; "
                  "activity/preference switching among largely collaborative models did not resolve "
                  "the observed tail failure. This is future work, not a demonstrated superior model.", ""]
    (directory / "findings.md").write_text("\n".join(lines), encoding="utf-8")
    report = ["# Task 2.5", "",
              "Training-only groups separate users by median activity and median head-item preference. "
              "Head items are the top 20% of the catalog by training count. All models use the same "
              "full-catalog test protocol. The table contrasts accuracy, novelty and popularity exposure. "
              "Higher activity can raise MRR through more relevant opportunities while reducing recall "
              "through a larger denominator. Head/tail recall must therefore be interpreted alongside "
              "exposure, rather than treating popularity concentration as retrieval success. Random "
              "provides a control for novelty without personalization. These are observational groups, "
              "not causal or demographic fairness estimates. Coverage depends on group size; cold-item "
              "and small-group conclusions require caution. Full model/group metrics, support counts "
              "and paired uncertainty intervals accompany this table.", "",
              "| Model | Activity | Users | MRR | Recall | Novelty (bits) | Diversity |",
              "|---|---|---:|---:|---:|---:|---:|"]
    highlights = list(dict.fromkeys(["Random", "Pop", baseline, "FeatureCombination", "GroupAwareHybrid"]))
    for name in highlights:
        for row in users:
            if row['model'] == name and row['group'] in ('activity_low', 'activity_high'):
                report.append(f"| {name} | {row['group'].removeprefix('activity_')} | {row['users']} | "
                              f"{row[f'MRR@{k}']:.3f} | {row[f'Recall@{k}']:.3f} | "
                              f"{row[f'Novelty@{k}']:.2f} | {row[f'IntraListDiversity@{k}']:.3f} |")
    report += ["", "| Model | Item group | Relevant interactions | Recall (micro) | Exposure share |",
               "|---|---|---:|---:|---:|"]
    for row in items:
        if row['model'] in (baseline, 'Pop', 'GroupAwareHybrid') and row['group'] in ('head', 'tail'):
            recall = row['micro_recall']
            report.append(f"| {row['model']} | {row['group']} | {row['relevant_interactions']} | "
                          f"{recall:.3f} | {row['exposure_share']:.3f} |" if recall is not None else
                          f"| {row['model']} | {row['group']} | 0 | undefined | {row['exposure_share']:.3f} |")
    if low and high and head['micro_recall'] is not None and tail['micro_recall'] is not None:
        report.insert(3, f"For {baseline}, low/high activity MRR is {low[f'MRR@{k}']:.3f}/"
                      f"{high[f'MRR@{k}']:.3f}, but recall is {low[f'Recall@{k}']:.3f}/"
                      f"{high[f'Recall@{k}']:.3f}. Head exposure is {100*head['exposure_share']:.1f}%; "
                      f"head/tail micro recall is {head['micro_recall']:.3f}/{tail['micro_recall']:.3f}. "
                      "The ensemble retains popularity concentration despite combining components.")
    delta = a[f'MRR@{k}'] - b[f'MRR@{k}']
    verdict = ("The switch improves the test MRR point estimate, but this alone does not establish superiority."
               if delta > 0 else "The switch does not improve test MRR over its validation-selected reference.")
    report += ["", "# Task 2.6", "",
               "Hypothesis: activity and popularity preference identify complementary model strengths. "
               "A four-cell switch selects full rankings by validation MRR; cells with fewer than 30 "
               "validation users or gains no greater than 0.005 retain the global reference. "
               f"The reference is {baseline}; final cell choices are {choices}. "
               "The gate is cross-fitted on reserved validation users before fitting its final rule. "
               "Test labels enter neither grouping nor selection. " + verdict + " "
               "Novelty, diversity and coverage changes qualify any accuracy gain. If multiple cells "
               "choose the same model, these signals provide limited evidence for specialization. "
               "Base hyperparameters and the pre-existing hybrid structure reused full validation, "
               "so gate cross-fitting is not fully nested evaluation. One seed and exploratory group "
               "comparisons limit generalization. Freeze the rule and replicate on fresh splits before "
               "claiming a superior hybrid.", "",
               "| Metric | Reference | Group-aware hybrid | Change |",
               "|---|---:|---:|---:|"]
    for metric in [f'MRR@{k}', f'NDCG@{k}', f'Novelty@{k}', f'IntraListDiversity@{k}', f'Coverage@{k}']:
        report.append(f"| {metric} | {b[metric]:.4f} | {a[metric]:.4f} | {a[metric]-b[metric]:+.4f} |")
    (directory / "report_sections.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
