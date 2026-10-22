"""Reuse Task 2 metrics and add calibration and fairness measures."""
import numpy as np

from task2.metrics import accuracy_metrics, beyond_accuracy_metrics
from task3.rerankers import miscalibration, popularity_deviation


def gini(values):
    """Exposure inequality, including items that were never recommended."""
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values) or values.sum() == 0:
        return 0.
    n = len(values)
    return float(2 * np.dot(np.arange(1, n + 1), values) / (n * values.sum()) - (n + 1) / n)


def evaluate_lists(lists, users, truth, profiles, k):
    """Macro user metrics; exposure uses 1/log2(rank+1) attention weights."""
    relevant = [set(np.flatnonzero(truth[user])) - {0} for user in users]
    metrics = accuracy_metrics(lists, relevant, k)
    metrics.update(beyond_accuracy_metrics(lists, profiles["popularity"], profiles["genres"], k))
    exposure = np.zeros(len(profiles["popularity"]))
    user_rows = []
    median_activity = np.median(profiles["activity"][1:])
    for user, row, target in zip(users, lists, relevant):
        row = row[:k]
        if not row:
            raise ValueError("Task 3 requires at least one eligible candidate per evaluated user")
        discounts = 1 / np.log2(np.arange(len(row)) + 2)
        np.add.at(exposure, row, discounts)
        genre_q = profiles["genre_dist"][row].mean(axis=0)
        pop_q = profiles["groups"][row].mean(axis=0)
        record = {
            "user_id": int(user),
            "activity_group": "low" if profiles["activity"][user] <= median_activity else "high",
            "history_head_share": float(profiles["user_popularity"][user, 0]),
            **accuracy_metrics([row], [target], k),
            f"Miscalibration@{k}": float(miscalibration(profiles["user_genres"][user], genre_q)),
            f"UPD@{k}": float(popularity_deviation(profiles["user_popularity"][user], pop_q)),
        }
        user_rows.append(record)
    for name in ("Miscalibration", "UPD"):
        metrics[f"{name}@{k}"] = float(np.mean([r[f"{name}@{k}"] for r in user_rows]))
    group_means = [
        np.mean([r[f"NDCG@{k}"] for r in user_rows if r["activity_group"] == group])
        for group in ("low", "high") if any(r["activity_group"] == group for r in user_rows)
    ]
    metrics[f"ActivityGRU@{k}"] = float(max(group_means) - min(group_means))
    group_exposure = exposure @ profiles["groups"]
    metrics[f"ExposureGap@{k}"] = float(popularity_deviation(
        profiles["target_exposure"], group_exposure / group_exposure.sum(),
    ))
    metrics[f"ExposureGini@{k}"] = gini(exposure[1:])
    return metrics, user_rows
