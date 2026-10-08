"""Verify checkpoint reload, scoring consistency, and training-only features.

Run after the validation-only smoke check:
python -m task1.feature_combination.check
"""
import numpy as np
import torch

from task1.experiment import load_data_and_model, model_class
from task1.model_registry import PROJECT_ROOT


def main():
    checkpoint = max((PROJECT_ROOT / "artifacts/checkpoints").glob(
        "*FeatureCombination*.pth"), key=lambda path: path.stat().st_mtime)
    config, model, dataset, train, _, _ = load_data_and_model(checkpoint)
    model.eval()
    matrix = train.dataset.inter_matrix(form="csr")
    counts = np.asarray(matrix.sum(axis=1)).reshape(-1, 1)
    expected = np.asarray(matrix @ model.item_genres.numpy()) / np.maximum(counts, 1)
    np.testing.assert_allclose(model.user_genres.numpy(), expected, atol=1e-6)
    expected = np.log1p(np.asarray(matrix.sum(axis=0)).reshape(-1, 1))
    expected /= max(float(expected.max()), 1)
    np.testing.assert_allclose(model.popularity.numpy(), expected, atol=1e-6)
    users = torch.tensor([1, 2])
    interaction = {dataset.uid_field: users}
    with torch.no_grad():
        full = model.full_sort_predict(interaction).view(2, dataset.item_num)
        direct = model(users, torch.tensor([10, 20]))
    torch.testing.assert_close(full[torch.arange(2), torch.tensor([10, 20])], direct)
    assert torch.isfinite(full).all()
    # Reconstructed training features and saved buffers should give identical scores.
    _, reloaded, *_ = load_data_and_model(checkpoint)
    reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded.full_sort_predict(interaction), full.flatten())
    # Every toggle combination supports scoring and a differentiable pairwise loss.
    for genres, text, popularity in [(False, False, False), (False, True, True),
                                      (True, False, True), (True, True, False)]:
        config["use_genres"], config["use_text"], config["use_popularity"] = genres, text, popularity
        variant = model_class("FeatureCombination")(config, train.dataset)
        loss = variant.calculate_loss({dataset.uid_field: users,
                                       dataset.iid_field: torch.tensor([10, 20]),
                                       variant.NEG_ITEM_ID: torch.tensor([30, 40])})
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None for p in variant.parameters())
    print("Passed: training-only profiles/popularity, full-catalog scoring, checkpoint reload, feature toggles and gradients.")


if __name__ == "__main__":
    main()
