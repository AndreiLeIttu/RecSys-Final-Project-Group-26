"""Independent accuracy and beyond-accuracy metrics for ranked recommendations."""
from __future__ import annotations

from collections.abc import Sequence
from typing import AbstractSet

import numpy as np


RecommendationLists = Sequence[Sequence[int]]
GroundTruth = Sequence[AbstractSet[int]]


def _validate_inputs(
    recommendations: RecommendationLists,
    relevant_items: GroundTruth,
    k: int,
) -> tuple[list[list[int]], list[set[int]]]:
    if k < 1:
        raise ValueError("k must be positive")
    if len(recommendations) != len(relevant_items):
        raise ValueError("recommendations and relevant_items must have equal lengths")
    ranked = [[int(item) for item in row[:k]] for row in recommendations]
    relevant = [set(map(int, items)) for items in relevant_items]
    return ranked, relevant


def precision_at_k(
    recommendations: RecommendationLists, relevant_items: GroundTruth, k: int,
) -> float:
    """Macro Precision@K; divide by K even when fewer than K items are eligible."""
    ranked, relevant = _validate_inputs(recommendations, relevant_items, k)
    if not ranked:
        return 0.0
    return float(np.mean([
        sum(item in truth for item in row) / k
        for row, truth in zip(ranked, relevant)
    ]))


def recall_at_k(
    recommendations: RecommendationLists, relevant_items: GroundTruth, k: int,
) -> float:
    """Macro Recall@K; a user with no relevant items contributes zero."""
    ranked, relevant = _validate_inputs(recommendations, relevant_items, k)
    if not ranked:
        return 0.0
    values = [
        sum(item in truth for item in row) / len(truth) if truth else 0.0
        for row, truth in zip(ranked, relevant)
    ]
    return float(np.mean(values))


def hit_rate_at_k(
    recommendations: RecommendationLists, relevant_items: GroundTruth, k: int,
) -> float:
    """Macro Hit Rate@K: fraction of users with at least one relevant result."""
    ranked, relevant = _validate_inputs(recommendations, relevant_items, k)
    if not ranked:
        return 0.0
    return float(np.mean([
        any(item in truth for item in row)
        for row, truth in zip(ranked, relevant)
    ]))


def mrr_at_k(
    recommendations: RecommendationLists, relevant_items: GroundTruth, k: int,
) -> float:
    """Macro reciprocal rank of each user's first relevant result in the top K."""
    ranked, relevant = _validate_inputs(recommendations, relevant_items, k)
    if not ranked:
        return 0.0
    values = []
    for row, truth in zip(ranked, relevant):
        rank = next((index for index, item in enumerate(row, start=1) if item in truth), None)
        values.append(1 / rank if rank is not None else 0.0)
    return float(np.mean(values))


def ndcg_at_k(
    recommendations: RecommendationLists, relevant_items: GroundTruth, k: int,
) -> float:
    """Macro binary nDCG@K with ideal DCG based on min(K, relevant-item count)."""
    ranked, relevant = _validate_inputs(recommendations, relevant_items, k)
    if not ranked:
        return 0.0
    values = []
    for row, truth in zip(ranked, relevant):
        dcg = sum(
            1 / np.log2(rank + 1)
            for rank, item in enumerate(row, start=1)
            if item in truth
        )
        ideal_length = min(k, len(truth))
        ideal_dcg = sum(1 / np.log2(rank + 1) for rank in range(1, ideal_length + 1))
        values.append(float(dcg / ideal_dcg) if ideal_dcg else 0.0)
    return float(np.mean(values))


def accuracy_metrics(
    recommendations: RecommendationLists, relevant_items: GroundTruth, k: int,
) -> dict[str, float]:
    """Return the five Task 1 accuracy metrics, calculated independently."""
    return {
        f"Precision@{k}": precision_at_k(recommendations, relevant_items, k),
        f"Recall@{k}": recall_at_k(recommendations, relevant_items, k),
        f"NDCG@{k}": ndcg_at_k(recommendations, relevant_items, k),
        f"MRR@{k}": mrr_at_k(recommendations, relevant_items, k),
        f"Hit@{k}": hit_rate_at_k(recommendations, relevant_items, k),
    }


def _top_items(recommendations: RecommendationLists, k: int) -> list[list[int]]:
    if k < 1:
        raise ValueError("k must be positive")
    return [[int(item) for item in row[:k]] for row in recommendations]


def catalog_coverage_at_k(
    recommendations: RecommendationLists, catalog_size: int, k: int,
) -> float:
    """Unique nonpadding recommended items divided by the nonpadding catalog size."""
    if catalog_size < 1:
        raise ValueError("catalog_size must be positive")
    rows = _top_items(recommendations, k)
    items = {item for row in rows for item in row}
    if any(item < 1 or item > catalog_size for item in items):
        raise ValueError("recommendations must contain nonpadding catalog item IDs")
    return len(items) / catalog_size


def novelty_at_k(
    recommendations: RecommendationLists,
    item_popularity: np.ndarray,
    k: int,
    smoothing: float = 1.0,
) -> float:
    """Mean self-information in bits using Laplace-smoothed training popularity.

    For item i, p(i) = (train_count(i) + smoothing) /
    (sum_j train_count(j) + smoothing * catalog_size), and novelty is
    -log2(p(i)). Item 0 is padding and excluded from the catalog.
    """
    popularity = np.asarray(item_popularity, dtype=np.float64)
    if popularity.ndim != 1 or len(popularity) < 2:
        raise ValueError("item_popularity must include padding and at least one item")
    if not np.isfinite(popularity).all() or np.any(popularity < 0):
        raise ValueError("item_popularity must contain finite nonnegative counts")
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")
    rows = _top_items(recommendations, k)
    items = [item for row in rows for item in row]
    if not items:
        return 0.0
    catalog_counts = popularity[1:]
    probabilities = (catalog_counts + smoothing) / (
        catalog_counts.sum() + smoothing * len(catalog_counts)
    )
    if any(item < 1 or item >= len(popularity) for item in items):
        raise ValueError("recommendations must contain nonpadding catalog item IDs")
    return float(np.mean([-np.log2(probabilities[item - 1]) for item in items]))


def average_recommendation_popularity_at_k(
    recommendations: RecommendationLists, item_popularity: np.ndarray, k: int,
) -> float:
    """Mean training interaction count of recommended items, pooled over users."""
    popularity = np.asarray(item_popularity, dtype=np.float64)
    if popularity.ndim != 1 or len(popularity) < 2:
        raise ValueError("item_popularity must include padding and at least one item")
    if not np.isfinite(popularity).all() or np.any(popularity < 0):
        raise ValueError("item_popularity must contain finite nonnegative counts")
    items = [item for row in _top_items(recommendations, k) for item in row]
    if not items:
        return 0.0
    if any(item < 1 or item >= len(popularity) for item in items):
        raise ValueError("recommendations must contain nonpadding catalog item IDs")
    return float(np.mean(popularity[items]))


def intra_list_diversity_at_k(
    recommendations: RecommendationLists, item_features: np.ndarray, k: int,
) -> float:
    """Macro mean pairwise genre cosine distance within each user's top K.

    Lists with fewer than two items contribute zero. A zero feature vector has
    cosine similarity zero to every vector, so its pairwise distance is one.
    """
    features = np.asarray(item_features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("item_features must have rows for padding and catalog items")
    if not np.isfinite(features).all():
        raise ValueError("item_features must be finite")
    rows = _top_items(recommendations, k)
    norms = np.linalg.norm(features, axis=1)
    normalized = np.divide(
        features,
        norms[:, None],
        out=np.zeros_like(features),
        where=norms[:, None] > 0,
    )
    values = []
    for row in rows:
        if any(item < 1 or item >= len(features) for item in row):
            raise ValueError("recommendations must contain nonpadding catalog item IDs")
        if len(row) < 2:
            values.append(0.0)
            continue
        vectors = normalized[row]
        similarities = vectors @ vectors.T
        pairwise = 1 - similarities[np.triu_indices(len(row), k=1)]
        values.append(float(np.mean(pairwise)))
    return float(np.mean(values)) if values else 0.0


def beyond_accuracy_metrics(
    recommendations: RecommendationLists,
    item_popularity: np.ndarray,
    item_features: np.ndarray,
    k: int,
) -> dict[str, float]:
    """Return global coverage and macro/pool-averaged catalog properties."""
    popularity = np.asarray(item_popularity)
    if popularity.ndim != 1:
        raise ValueError("item_popularity must be one-dimensional")
    return {
        f"Coverage@{k}": catalog_coverage_at_k(recommendations, len(popularity) - 1, k),
        f"IntraListDiversity@{k}": intra_list_diversity_at_k(
            recommendations, item_features, k,
        ),
        f"Novelty@{k}": novelty_at_k(recommendations, popularity, k),
        f"AveragePopularity@{k}": average_recommendation_popularity_at_k(
            recommendations, popularity, k,
        ),
    }