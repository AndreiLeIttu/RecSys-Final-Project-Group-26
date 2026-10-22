"""Small selection helpers shared by hybrid tuning and its ablations."""
import numpy as np

from task1.weighted_hybrid import fit_weights, regression_pairs


def user_folds(users, user_count, seed, count=3):
    """Every validation user is held out from regression fitting once."""
    users = np.asarray(users, dtype=int)
    if len(users) < count:
        raise ValueError("Not enough validation users for cross-fitting")
    shuffled = np.random.default_rng(seed).permutation(users)
    folds = np.full(user_count, -1, dtype=int)
    folds[shuffled] = np.arange(len(shuffled)) % count
    return folds


def blend(scores, weights):
    return np.einsum("m,mui->ui", weights, scores, optimize=True)


def crossfit_weights(scores, train, valid, ratio, seed, folds):
    """Return out-of-fold validation scores; no held-out user's labels enter fit."""
    users, items, targets = regression_pairs(train, valid, ratio, seed)
    features = scores[:, users, items].T
    predictions = np.zeros(train.shape, dtype=np.float32)
    for fold in sorted(set(folds[folds >= 0])):
        fitting = (folds[users] >= 0) & (folds[users] != fold)
        fit = fit_weights(features[fitting], targets[fitting])
        held_out = folds == fold
        predictions[held_out] = blend(scores[:, held_out], fit["weights"])
    return predictions


def refit_weights(scores, train, valid, ratio, seed):
    users, items, targets = regression_pairs(train, valid, ratio, seed)
    return fit_weights(scores[:, users, items].T, targets)


def switch_scores(first, second, activity, threshold, reverse=False):
    """Select an entire ranked-score row using training activity only."""
    choose_first = np.asarray(activity) <= threshold
    if reverse:
        choose_first = ~choose_first
    return np.where(choose_first[:, None], first, second)
