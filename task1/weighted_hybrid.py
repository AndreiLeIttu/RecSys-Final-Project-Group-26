"""Score normalization and constrained linear regression for Task 1.3."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def normalize_scores(scores: np.ndarray, train_matrix: np.ndarray) -> np.ndarray:
    """Min-max scale each user's unseen-item scores; constant rows become zero."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != train_matrix.shape or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite and match the training matrix shape")
    eligible = ~train_matrix.astype(bool)
    eligible[:, 0] = False  # RecBole's padding item is never a candidate.
    lower = np.min(np.where(eligible, scores, np.inf), axis=1, keepdims=True)
    upper = np.max(np.where(eligible, scores, -np.inf), axis=1, keepdims=True)
    span = upper - lower
    usable = np.isfinite(span) & (span > 0)
    normalized = np.divide(
        scores - np.where(np.isfinite(lower), lower, 0),
        span,
        out=np.zeros_like(scores),
        where=usable,
    )
    return np.where(eligible, normalized, 0).astype(np.float32)


def regression_pairs(
    train_matrix: np.ndarray,
    valid_matrix: np.ndarray,
    negative_ratio: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validation positives and sampled unobserved pairs; never inspect test labels."""
    if negative_ratio < 1:
        raise ValueError("negative_ratio must be at least 1")
    rng = np.random.default_rng(seed)
    users, items, targets = [], [], []
    for user in range(1, train_matrix.shape[0]):
        positive = np.flatnonzero(valid_matrix[user])
        positive = positive[positive != 0]
        if not len(positive):
            continue
        eligible = ~(train_matrix[user].astype(bool) | valid_matrix[user].astype(bool))
        eligible[0] = False
        candidates = np.flatnonzero(eligible)
        negative = rng.choice(
            candidates, size=min(len(candidates), negative_ratio * len(positive)),
            replace=False,
        )
        user_items = np.concatenate([positive, negative])
        users.extend([user] * len(user_items))
        items.extend(user_items)
        targets.extend([1.0] * len(positive) + [0.0] * len(negative))
    if not targets or min(targets) == max(targets):
        raise ValueError("Regression requires both validation positives and negatives")
    return np.asarray(users), np.asarray(items), np.asarray(targets)


def fit_weights(features: np.ndarray, targets: np.ndarray) -> dict:
    """Fit y ~ intercept + Xw, subject to w >= 0 and sum(w) = 1."""
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if (
        features.ndim != 2 or not features.shape[1]
        or targets.shape != (features.shape[0],) or not len(targets)
        or not np.isfinite(features).all() or not np.isfinite(targets).all()
    ):
        raise ValueError("Expected finite nonempty features and aligned targets")
    # Centering eliminates the unconstrained intercept from the optimization.
    mean_x, mean_y = features.mean(axis=0), targets.mean()
    centered_x, centered_y = features - mean_x, targets - mean_y
    gram = centered_x.T @ centered_x / len(targets)
    rhs = centered_x.T @ centered_y / len(targets)
    count = features.shape[1]
    result = minimize(
        lambda w: float(w @ gram @ w - 2 * rhs @ w),
        np.full(count, 1 / count),
        jac=lambda w: 2 * (gram @ w - rhs),
        method="SLSQP",
        bounds=[(0, 1)] * count,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                     "jac": lambda w: np.ones_like(w)},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Hybrid regression failed: {result.message}")
    weights = np.clip(result.x, 0, 1)
    weights /= weights.sum()
    intercept = float(mean_y - mean_x @ weights)
    mse = float(np.mean((features @ weights + intercept - targets) ** 2))
    return {"weights": weights, "intercept": intercept, "training_mse": mse}
