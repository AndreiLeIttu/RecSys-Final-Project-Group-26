"""Small greedy rerankers. Inputs are scored candidates, not test labels."""
from __future__ import annotations

import numpy as np


METHODS = ("diversity", "calibration", "user_fairness", "item_fairness")


def distributions(features):
    """Give each item one unit of mass, shared equally between its genres."""
    features = np.asarray(features, dtype=float)
    totals = features.sum(axis=-1, keepdims=True)
    return np.divide(features, totals, out=np.zeros_like(features), where=totals > 0)


def miscalibration(p, q):
    """KL(p || 0.99 q + 0.01 p), in nats (Calibration lecture, slide 12)."""
    p, q = np.broadcast_arrays(np.asarray(p, dtype=float), np.asarray(q, dtype=float))
    smooth = .99 * q + .01 * p
    ratio = np.divide(p, smooth, out=np.ones_like(p), where=p > 0)
    return np.sum(p * np.log(ratio), axis=-1)


def popularity_deviation(p, q):
    """Total variation between head/tail distributions, between zero and one."""
    return .5 * np.abs(np.asarray(p) - np.asarray(q)).sum(axis=-1)


def build_profiles(train, genres):
    """All profiles, popularity groups and targets come from training only."""
    popularity = train.sum(axis=0)
    count = len(popularity) - 1
    if count < 2 or np.any(genres[1:].sum(axis=1) == 0):
        raise ValueError("Need at least two items and genre metadata for every item")
    # Head = top 20% of catalog items; ties use ascending internal item ID.
    order = np.lexsort((np.arange(1, count + 1), -popularity[1:])) + 1
    head_count = max(1, int(np.ceil(.2 * count)))
    groups = np.zeros((count + 1, 2))
    groups[1:, 1] = 1
    groups[order[:head_count]] = [1, 0]
    genre_dist = distributions(genres)
    return {
        "genres": genres,
        "genre_dist": genre_dist,
        "user_genres": distributions(train @ genre_dist),
        "groups": groups,
        "user_popularity": distributions(train @ groups),
        "target_exposure": groups[1:].mean(axis=0),
        "popularity": popularity,
        "activity": train.sum(axis=1),
    }


def candidates(scores, seen, size):
    """Stable relevance order: descending score, then ascending item ID."""
    scores = np.asarray(scores)
    if scores.shape != seen.shape or scores.ndim != 2 or size < 1:
        raise ValueError("Need aligned score/history matrices and positive size")
    if not np.isfinite(scores).all():
        raise ValueError("Candidate scores must be finite")
    masked = np.where(seen, -np.inf, scores)
    masked[:, 0] = -np.inf
    order = np.argsort(-masked, axis=1, kind="stable")[:, :size]
    values = np.take_along_axis(masked, order, axis=1)
    return np.where(np.isfinite(values), order, -1)


def rerank(items, scores, method, strength, k, profiles, user):
    """Greedily trade mean normalized relevance for a list-level objective.

    Item fairness targets rank-discounted head/tail exposure proportional to
    catalog size. User fairness targets the user's own head/tail preferences.
    Neither method uses evaluation labels or promises equal user accuracy.
    """
    if method not in METHODS or not 0 <= strength <= 1 or k < 1:
        raise ValueError("Unknown method, invalid strength, or nonpositive K")
    items = np.asarray(items, dtype=int)
    items = items[items > 0]
    if len(np.unique(items)) != len(items):
        raise ValueError("Candidates must be unique")
    if not len(items) or strength == 0:
        return items[:k].tolist()
    relevance = np.asarray(scores)[items].astype(float)
    span = np.ptp(relevance)
    relevance = (relevance - relevance.min()) / span if span else np.zeros_like(relevance)
    available = np.ones(len(items), dtype=bool)
    chosen = []
    relevance_sum = 0.
    genre_sum = np.zeros(profiles["genre_dist"].shape[1])
    group_sum = np.zeros(2)
    exposure_sum = np.zeros(2)
    discount_sum = 0.
    pair_sum = 0.
    distances = None
    if method == "diversity":
        vectors = profiles["genres"][items].astype(float)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        distances = np.clip(1 - vectors @ vectors.T, 0, 1)
    for position in range(min(k, len(items))):
        length = position + 1
        discount = 1 / np.log2(length + 1)
        if method == "diversity":
            additions = distances[:, chosen].sum(axis=1) if chosen else np.zeros(len(items))
            benefit = (pair_sum + additions) / (length * (length - 1) / 2) if position else additions
        elif method == "calibration":
            q = (genre_sum + profiles["genre_dist"][items]) / length
            benefit = -miscalibration(profiles["user_genres"][user], q)
        elif method == "user_fairness":
            q = (group_sum + profiles["groups"][items]) / length
            benefit = -popularity_deviation(profiles["user_popularity"][user], q)
        else:
            q = (exposure_sum + discount * profiles["groups"][items]) / (discount_sum + discount)
            benefit = -popularity_deviation(profiles["target_exposure"], q)
        objective = (1 - strength) * (relevance_sum + relevance) / length + strength * benefit
        index = int(np.argmax(np.where(available, objective, -np.inf)))
        if method == "diversity":
            pair_sum += float(additions[index])
        chosen.append(index)
        available[index] = False
        relevance_sum += relevance[index]
        genre_sum += profiles["genre_dist"][items[index]]
        group_sum += profiles["groups"][items[index]]
        exposure_sum += discount * profiles["groups"][items[index]]
        discount_sum += discount
    return items[chosen].tolist()
