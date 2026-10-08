"""Build the Task 1.3 hybrid from the selected Task 1.2 recommenders."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil

import numpy as np
import torch

from task1.experiment import load_data_and_model, resolve_project_path, train_model
from task1.io_utils import write_csv, write_json
from task1.model_registry import PROJECT_ROOT, resolve_models
from task1.weighted_hybrid import fit_weights, normalize_scores, regression_pairs

from recbole.data.interaction import Interaction
from recbole.trainer import Trainer
from recbole.utils import init_seed


class HybridEvaluator(Trainer):
    """Use RecBole evaluation without constructing a neural training optimizer."""

    def _build_optimizer(self, **kwargs):
        return None


class WeightedHybrid(torch.nn.Module):
    """Frozen full-catalog hybrid scores, with RecBole's prediction interface.

    Store scores to reproduce inference even for the stochastic Random baseline.
    RecBole applies the appropriate validation/test history masks at evaluation.
    """

    def __init__(self, scores: np.ndarray, user_field: str, item_field: str):
        super().__init__()
        self.register_buffer("scores", torch.as_tensor(scores, dtype=torch.float32))
        self.user_field, self.item_field = user_field, item_field
        self.other_parameter_name = []

    def full_sort_predict(self, interaction):
        return self.scores[interaction[self.user_field]].flatten()

    def predict(self, interaction):
        return self.scores[interaction[self.user_field], interaction[self.item_field]]


def load_hybrid(path="artifacts/hybrid/WeightedHybrid.npz") -> WeightedHybrid:
    """Restore scores; user/item IDs use the saved RecBole token mappings."""
    with np.load(resolve_project_path(path), allow_pickle=False) as artifact:
        return WeightedHybrid(
            artifact["scores"], str(artifact["user_field"]), str(artifact["item_field"]),
        )


def dataset_signature(dataset, loaders) -> str:
    """Reject components with different token mappings or train/valid/test splits."""
    digest = hashlib.sha256()
    for field in (dataset.uid_field, dataset.iid_field):
        digest.update(json.dumps(dataset.field2id_token[field].tolist()).encode())
    for loader in loaders:
        interactions = loader._dataset.inter_feat
        for field in (dataset.uid_field, dataset.iid_field):
            digest.update(interactions[field].cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def collect_scores(model, dataset, batch_size: int) -> np.ndarray:
    """Use full-sort scores consistently, including FISM's full-sort score scale."""
    model.eval()
    scores = np.zeros((dataset.user_num, dataset.item_num), dtype=np.float32)
    for start in range(1, dataset.user_num, batch_size):
        users = torch.arange(start, min(start + batch_size, dataset.user_num))
        interaction = Interaction({dataset.uid_field: users}).to(model.device)
        try:
            output = model.full_sort_predict(interaction)
        except NotImplementedError:
            interaction = Interaction({
                dataset.uid_field: users.repeat_interleave(dataset.item_num),
                dataset.iid_field: torch.arange(dataset.item_num).repeat(len(users)),
            }).to(model.device)
            output = model.predict(interaction)
        scores[users.numpy()] = output.detach().cpu().numpy().reshape(
            len(users), dataset.item_num,
        )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn a weighted hybrid by regression.")
    parser.add_argument("--models", nargs="+", help="Default: all Task 1.2 models")
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--user-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.negative_ratio < 1 or args.user_batch_size < 1:
        parser.error("negative-ratio and user-batch-size must be positive")
    specs = resolve_models(args.models)
    if len(specs) < 2 or len({spec.name for spec in specs}) != len(specs):
        parser.error("Choose at least two distinct recommenders")
    selected_path = PROJECT_ROOT / "results" / "best_models.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    missing = [spec.name for spec in specs if spec.name not in selected]
    if missing:
        parser.error(f"Run task1.tune_models first; no selected settings for {missing}")

    normalized_scores, checkpoints = [], {}
    signature = None
    for spec in specs:
        path = PROJECT_ROOT / "artifacts" / "best" / f"{spec.name}.pth"
        if not path.exists():
            print(f"Training missing {spec.name} checkpoint with selected settings", flush=True)
            result = train_model(spec, {**selected[spec.name]["parameters"],
                                       "show_progress": False})
            if not result["checkpoint"]:
                raise RuntimeError(f"No checkpoint produced for {spec.name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolve_project_path(result["checkpoint"]), path)
        print(f"Collecting {spec.name} scores", flush=True)
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(path)
        current_signature = dataset_signature(dataset, (train_data, valid_data, test_data))
        if signature is not None and current_signature != signature:
            raise ValueError(f"{spec.name} uses different splits or user/item IDs")
        if signature is None:
            signature = current_signature
            reference_config, reference_dataset = config, dataset
            reference_valid, reference_test = valid_data, test_data
            train_matrix = train_data._dataset.inter_matrix(form="csr").toarray() > 0
            valid_matrix = valid_data._dataset.inter_matrix(form="csr").toarray() > 0
        init_seed(config["seed"], config["reproducibility"])
        normalized_scores.append(normalize_scores(
            collect_scores(model, dataset, args.user_batch_size), train_matrix,
        ))
        checkpoints[spec.name] = path.relative_to(PROJECT_ROOT).as_posix()
        del model

    users, items, targets = regression_pairs(
        train_matrix, valid_matrix, args.negative_ratio, reference_config["seed"],
    )
    features = np.column_stack([scores[users, items] for scores in normalized_scores])
    fitted = fit_weights(features, targets)
    hybrid_scores = np.full(train_matrix.shape, fitted["intercept"], dtype=np.float32)
    for weight, scores in zip(fitted["weights"], normalized_scores):
        hybrid_scores += weight * scores
    hybrid = WeightedHybrid(hybrid_scores, reference_dataset.uid_field,
                            reference_dataset.iid_field)
    trainer = HybridEvaluator(reference_config, hybrid)
    # Validation is a fit diagnostic, not an unbiased estimate or tuning result.
    valid_result = trainer.evaluate(reference_valid, load_best_model=False,
                                    show_progress=False)
    test_result = trainer.evaluate(reference_test, load_best_model=False,
                                   show_progress=False)
    artifact_dir = PROJECT_ROOT / "artifacts" / "hybrid"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_dir / "WeightedHybrid.npz", scores=hybrid_scores,
        weights=fitted["weights"], intercept=fitted["intercept"],
        models=np.asarray([spec.name for spec in specs]),
        user_field=reference_dataset.uid_field, item_field=reference_dataset.iid_field,
        user_tokens=np.asarray(reference_dataset.field2id_token[reference_dataset.uid_field], dtype=str),
        item_tokens=np.asarray(reference_dataset.field2id_token[reference_dataset.iid_field], dtype=str),
    )
    record = {
        "model": "WeightedHybrid", "dataset": "ml-100k",
        "seed": int(reference_config["seed"]), "split_signature": signature,
        "regression": "least squares with intercept, nonnegative weights summing to one",
        "normalization": "per-user min-max over nonpadding items unseen in training",
        "negative_ratio": args.negative_ratio, "user_batch_size": args.user_batch_size,
        "fit_pairs": len(targets), "fit_positives": int(targets.sum()),
        "fit_negatives": int(len(targets) - targets.sum()),
        "weights": dict(zip([spec.name for spec in specs], fitted["weights"].tolist())),
        "intercept": fitted["intercept"], "training_mse": fitted["training_mse"],
        "base_checkpoints": checkpoints,
        "artifact": "artifacts/hybrid/WeightedHybrid.npz",
        "validation_fit_result": {k: float(v) for k, v in valid_result.items()},
        "test_result": {k: float(v) for k, v in test_result.items()},
    }
    write_json(PROJECT_ROOT / "results" / "weighted_hybrid.json", record)
    write_csv(PROJECT_ROOT / "results" / "hybrid_coefficients.csv", [
        {"model": name, "weight": weight} for name, weight in record["weights"].items()
    ])
    write_csv(PROJECT_ROOT / "results" / "weighted_hybrid.csv", [{
        "model": record["model"], **record["test_result"],
    }])
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
