"""Generate full-catalog recommendation rankings with explicit history masking."""
from __future__ import annotations

import numpy as np
import torch


def rank_scores(scores: np.ndarray, seen_items: np.ndarray, k: int) -> np.ndarray:
    """Return each user's ordered top-K item IDs, padded with -1 if needed.

    The seen mask should combine training and validation interactions for test
    evaluation. Item 0 is always excluded, regardless of the mask value.
    Ties use PyTorch top-k ordering, matching the Task 1 full-sort evaluator.
    """
    scores = np.asarray(scores, dtype=np.float64)
    seen_items = np.asarray(seen_items, dtype=bool)
    if scores.ndim != 2 or scores.shape != seen_items.shape:
        raise ValueError("scores and seen_items must be equally shaped 2D matrices")
    if scores.shape[1] < 2:
        raise ValueError("scores must include padding and at least one catalog item")
    if k < 1:
        raise ValueError("k must be positive")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")

    score_tensor = torch.as_tensor(scores, dtype=torch.float64)
    seen_tensor = torch.as_tensor(seen_items, dtype=torch.bool)
    score_tensor.masked_fill_(seen_tensor, -torch.inf)
    score_tensor[:, 0] = -torch.inf
    values, items = torch.topk(score_tensor, k, dim=-1)
    rankings = np.full((scores.shape[0], k), -1, dtype=np.int64)
    item_rows = items.numpy()
    valid_rows = torch.isfinite(values).numpy()
    for user_id in range(scores.shape[0]):
        selected = item_rows[user_id][valid_rows[user_id]]
        rankings[user_id, :len(selected)] = selected
    return rankings


def _store_batch_topk(
    rankings: np.ndarray,
    user_ids: np.ndarray,
    scores: torch.Tensor,
    history_index,
    k: int,
) -> None:
    scores = scores.view(len(user_ids), -1)
    scores[:, 0] = -torch.inf
    if history_index is not None:
        history_users, history_items = history_index
        scores[history_users.to(scores.device), history_items.to(scores.device)] = -torch.inf
    values, items = torch.topk(scores, k, dim=-1)
    items = items.cpu().numpy()
    valid = torch.isfinite(values).cpu().numpy()
    for row, user_id in enumerate(user_ids):
        selected = items[row][valid[row]]
        rankings[int(user_id), :len(selected)] = selected


@torch.no_grad()
def rank_score_matrix(scores: np.ndarray, test_data, dataset, k: int) -> np.ndarray:
    """Rank cached full-catalog scores in RecBole's test-loader batches."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (dataset.user_num, dataset.item_num):
        raise ValueError("scores do not match the dataset user/item dimensions")
    if k < 1 or k >= dataset.item_num:
        raise ValueError("k must be positive and smaller than the item catalog")
    rankings = np.full((dataset.user_num, k), -1, dtype=np.int64)
    for interaction, history_index, _, _ in test_data:
        user_ids = interaction[dataset.uid_field].detach().cpu().numpy()
        user_scores = torch.as_tensor(scores[user_ids], dtype=torch.float64)
        _store_batch_topk(rankings, user_ids, user_scores, history_index, k)
    return rankings


@torch.no_grad()
def collect_test_rankings(model, test_data, dataset, k: int) -> np.ndarray:
    """Rank test users using RecBole's full-sort batches and history masks.

    This uses the same prediction batching and ``torch.topk`` ordering as the
    Task 1 evaluator, but calculates no RecBole evaluation metrics.
    """
    if k < 1 or k >= dataset.item_num:
        raise ValueError("k must be positive and smaller than the item catalog")
    model.eval()
    rankings = np.full((dataset.user_num, k), -1, dtype=np.int64)
    item_ids = torch.arange(dataset.item_num, device=model.device)
    for interaction, history_index, _, _ in test_data:
        user_ids = interaction[dataset.uid_field].detach().cpu().numpy()
        interaction = interaction.to(model.device)
        try:
            scores = model.full_sort_predict(interaction)
        except NotImplementedError:
            user_count = len(interaction)
            tiled = interaction.repeat_interleave(dataset.item_num)
            tiled.update({dataset.iid_field: item_ids.repeat(user_count)})
            scores = model.predict(tiled)
        _store_batch_topk(rankings, user_ids, scores, history_index, k)
    return rankings


def rankings_for_users(
    rankings: np.ndarray, user_ids: list[int] | np.ndarray,
) -> list[list[int]]:
    """Convert a padded ranking matrix to lists for the selected users."""
    return [
        [int(item) for item in rankings[int(user_id)] if item >= 0]
        for user_id in user_ids
    ]