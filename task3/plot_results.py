"""Plot saved validation results without rerunning any experiments."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from task1.model_registry import PROJECT_ROOT
from task3.evaluate import OBJECTIVES


def plot(directory, models):
    directory = Path(directory)
    rows = list(csv.DictReader((directory / "validation.csv").open()))
    selected = json.loads((directory / "selected.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    k = protocol["topk"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for ax, (method, (metric, direction)) in zip(axes.flat, OBJECTIVES.items()):
        for model in models:
            trials = [row for row in rows if row["model"] == model and row["method"] == method]
            if not trials:
                continue
            chosen = next(row["strength"] for row in selected if row["model"] == model and row["method"] == method)
            x = [float(row[f"NDCG@{k}"]) for row in trials]
            y = [float(row[f"{metric}@{k}"]) for row in trials]
            line, = ax.plot(x, y, "o-", markersize=4, label=model)
            index = next(i for i, row in enumerate(trials) if float(row["strength"]) == chosen)
            ax.scatter([x[index]], [y[index]], s=100, facecolors="none", edgecolors=line.get_color())
        ax.set_title(method.replace("_", " ").capitalize())
        ax.set_xlabel(f"Validation nDCG@{k} (higher is better)")
        ax.set_ylabel(f"{metric}@{k} ({'higher' if direction == 1 else 'lower'} is better)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Reranking trade-offs\nRings mark settings selected within the validation accuracy budget", fontsize=13)
    fig.savefig(directory / "tradeoffs.svg")
    fig.savefig(directory / "tradeoffs.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=PROJECT_ROOT / "results/task3")
    parser.add_argument("--models", nargs="+", default=["EASE", "FeatureCombination", "TunedWeightedHybrid", "SwitchingHybrid"])
    args = parser.parse_args()
    plot(args.directory, args.models)
