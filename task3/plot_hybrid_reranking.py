"""Plot Task 3.3 validation trade-offs and held-out strategy comparisons."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task1.io_utils import write_csv
from task1.model_registry import PROJECT_ROOT
from task3.evaluate_hybrid_reranking import STRATEGIES
from task3.evaluate import OBJECTIVES


matplotlib.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

STRATEGY_LABELS = {
    "rerank_components_then_aggregate": "Rerank → Aggregate",
    "aggregate_then_rerank": "Aggregate → Rerank",
}
OBJECTIVE_METRICS = {
    "diversity": "IntraListDiversity",
    "calibration": "Miscalibration",
    "user_fairness": "UPD",
    "item_fairness": "ExposureGap",
}
COLORS = {
    "rerank_components_then_aggregate": "#2878B5",
    "aggregate_then_rerank": "#E1812C",
}


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _number(row, key):
    return float(row[key])


def _strength_label(value):
    return f"{float(value):g}"


def _find_row(rows, strategy, method, strength):
    return next(
        row for row in rows
        if row["strategy"] == strategy and row["method"] == method
        and _number(row, "strength") == float(strength)
    )


def _test_table_rows(test_rows, selections, hybrid, k):
    baseline = _find_row(test_rows, "aggregate_then_rerank", "diversity", 0)
    metric_names = [
        key for key in baseline
        if key.endswith(f"@{k}") and not key.startswith("delta_")
    ]
    table_rows = [{
        "hybrid": hybrid,
        "strategy": "original_hybrid_no_reranking",
        "method": "baseline",
        "strength": 0.,
        **{name: baseline[name] for name in metric_names},
        **{f"delta_{name}": 0. for name in metric_names},
    }]
    for selection in selections:
        if selection["hybrid"] != hybrid:
            continue
        row = _find_row(
            test_rows, selection["strategy"], selection["method"], selection["strength"],
        )
        table_rows.append({
            **{name: row[name] for name in row if name != "hybrid"},
            "hybrid": hybrid,
        })
    return table_rows


def _write_test_table(directory, hybrid, rows, k, include_svg):
    metric_names = [
        key for key in rows[0]
        if key.endswith(f"@{k}") and not key.startswith("delta_")
    ]
    csv_fields = ["hybrid", "strategy", "method", "strength"]
    for metric in metric_names:
        csv_fields.extend((metric, f"delta_{metric}"))
    write_csv(directory / f"{hybrid}-selected-test.csv", [
        {field: row.get(field, "") for field in csv_fields} for row in rows
    ])

    displayed_metrics = (
        f"NDCG@{k}",
        f"delta_NDCG@{k}",
        *(f"{metric}@{k}" for metric in OBJECTIVE_METRICS.values()),
    )
    metric_headers = {
        f"NDCG@{k}": f"NDCG@{k}",
        f"delta_NDCG@{k}": f"Delta\nNDCG@{k}",
        f"IntraListDiversity@{k}": "IntraList\nDiversity",
        f"Miscalibration@{k}": "Miscalibration",
        f"UPD@{k}": "UPD\n(user fairness)",
        f"ExposureGap@{k}": "ExposureGap\n(item fairness)",
    }
    headers = [
        "Approach", "Objective", "Strength",
        *(metric_headers[metric] for metric in displayed_metrics),
    ]
    cells = []
    for row in rows:
        if row["method"] == "baseline":
            approach, method, strength = "Original hybrid", "—", "—"
        else:
            approach = STRATEGY_LABELS[row["strategy"]]
            method, strength = row["method"].replace("_", " ").title(), _strength_label(row["strength"])
        cells.append([
            approach, method, strength,
            *[
                (f"{float(row[metric]):.3f}" if row.get(metric, "") != "" else "—")
                for metric in displayed_metrics
            ],
        ])

    fig_height = max(3.2, .43 * (len(cells) + 1))
    fig, ax = plt.subplots(figsize=(17, fig_height), layout="constrained")
    ax.axis("off")
    table = ax.table(
        cellText=cells, colLabels=headers, loc="center", cellLoc="center",
        colColours=["#E8EEF5"] * len(headers),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    widths = (.19, .12, .07, .09, .09, .11, .11, .10, .12)
    for column, width in enumerate(widths):
        for row_index in range(len(cells) + 1):
            table[(row_index, column)].set_width(width)
    ax.set_title(
        f"Table 1 — Held-out Test Performance of Hybrid Reranking Strategies\n{hybrid}",
        pad=16,
    )
    fig.savefig(directory / f"Task3.3-{hybrid}-selected-test-table.png",
                dpi=200, bbox_inches="tight")
    if include_svg:
        fig.savefig(directory / f"Task3.3-{hybrid}-selected-test-table.svg",
                    bbox_inches="tight")
    plt.close(fig)


def plot(directory, include_svg=False):
    directory = Path(directory)
    validation = _read_csv(directory / "validation.csv")
    test = _read_csv(directory / "test.csv")
    selected = json.loads((directory / "selected.json").read_text(encoding="utf-8"))
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    hybrids = protocol.get("hybrids", sorted({row["hybrid"] for row in validation}))
    k = protocol["topk"]

    for hybrid in hybrids:
        chosen = {
            (row["strategy"], row["method"]): float(row["strength"])
            for row in selected if row["hybrid"] == hybrid
        }
        val_rows = [row for row in validation if row["hybrid"] == hybrid]
        test_rows = [row for row in test if row["hybrid"] == hybrid]
        if not val_rows or not test_rows:
            raise ValueError(f"Missing validation or test rows for {hybrid}")
        baseline_ndcg = _number(
            _find_row(val_rows, "aggregate_then_rerank", "diversity", 0),
            f"NDCG@{k}",
        )

        fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
        for ax, (method, (metric, direction)) in zip(axes.flat, OBJECTIVES.items()):
            for strategy in STRATEGIES:
                rows = sorted(
                    (row for row in val_rows
                     if row["strategy"] == strategy and row["method"] == method),
                    key=lambda row: _number(row, "strength"),
                )
                if not rows:
                    continue
                x = [_number(row, f"NDCG@{k}") for row in rows]
                y = [_number(row, f"{metric}@{k}") for row in rows]
                color = COLORS[strategy]
                ax.plot(x, y, "o-", color=color, markersize=4,
                        label=STRATEGY_LABELS[strategy])
                selected_strength = chosen[(strategy, method)]
                selected_row = _find_row(rows, strategy, method, selected_strength)
                ax.scatter(
                    [_number(selected_row, f"NDCG@{k}")],
                    [_number(selected_row, f"{metric}@{k}")],
                    marker="*", s=230, color=color, edgecolor="black",
                    linewidth=.7, zorder=4,
                )
                horizontal_offset = -42 if strategy == STRATEGIES[0] else 12
                vertical_offset = 14 if strategy == STRATEGIES[0] else -24
                ax.annotate(
                    f"s={_strength_label(selected_strength)}",
                    (_number(selected_row, f"NDCG@{k}"),
                     _number(selected_row, f"{metric}@{k}")),
                    xytext=(horizontal_offset, vertical_offset),
                    textcoords="offset points", fontsize=8, color=color,
                    bbox={"boxstyle": "round,pad=.2", "fc": "white", "ec": color, "alpha": .9},
                    zorder=5,
                )
            ax.axvline(
                baseline_ndcg, color="#333333", linestyle="--", linewidth=1.3,
                label=f"Original hybrid baseline ({baseline_ndcg:.3f})",
            )
            direction_label = "higher is better" if direction == 1 else "lower is better"
            ax.set_title(
                f"{method.replace('_', ' ').title()}\n{metric}@{k} ({direction_label})",
            )
            ax.set_xlabel(f"Validation NDCG@{k} (higher is better)")
            ax.set_ylabel(f"{metric}@{k} ({'higher' if direction == 1 else 'lower'} is better)")
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        fig.suptitle(
            f"Figure 1 — Accuracy–Quality Trade-offs of Hybrid Reranking Strategies\n"
            f"{hybrid} | Stars mark validation-selected strengths",
            fontsize=13,
        )
        fig.savefig(directory / f"Task3.3-{hybrid}-validation-tradeoffs.png", dpi=200)
        if include_svg:
            fig.savefig(directory / f"Task3.3-{hybrid}-validation-tradeoffs.svg")
        plt.close(fig)

        _write_test_table(
            directory, hybrid,
            _test_table_rows(test_rows, selected, hybrid, k), k, include_svg,
        )

    print(f"Saved Task 3.3 plots to {directory}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=PROJECT_ROOT / "results/task3_3")
    parser.add_argument("--svg", action="store_true", help="Also export SVG versions")
    args = parser.parse_args()
    plot(args.directory, args.svg)
