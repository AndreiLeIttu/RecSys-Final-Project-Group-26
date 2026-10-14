"""Evaluate selected Task 1 checkpoints using independent Task 2 metrics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from task1.experiment import load_data_and_model, resolve_project_path, train_model
from task1.model_registry import MODEL_SPECS, PROJECT_ROOT
from task1.train_hybrid import dataset_signature, load_hybrid
from task2.metrics import accuracy_metrics, beyond_accuracy_metrics
from task2.recommendations import collect_test_rankings, rank_score_matrix, rankings_for_users


DEFAULT_MODELS = [*MODEL_SPECS, "WeightedHybrid"]
CACHE_VERSION = "2"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_checkpoint(
    name: str,
    selected: dict[str, Any],
    recover_missing: bool,
) -> tuple[Path | None, str | None]:
    record = selected.get(name)
    checkpoint = resolve_project_path(record["checkpoint"]) if record and record.get("checkpoint") else (
        PROJECT_ROOT / "artifacts" / "best" / f"{name}.pth"
    )
    if checkpoint.exists():
        return checkpoint, None
    if not recover_missing:
        return None, "checkpoint is missing and recovery was disabled"
    if name not in MODEL_SPECS or not record or not record.get("parameters") is not None:
        return None, "no selected Task 1 configuration is available for recovery"

    try:
        result = train_model(
            MODEL_SPECS[name], {**record.get("parameters", {}), "show_progress": False},
        )
        if not result.get("checkpoint"):
            return None, "checkpoint recovery training produced no checkpoint"
        source = resolve_project_path(result["checkpoint"])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != checkpoint.resolve():
            shutil.copy2(source, checkpoint)
        return checkpoint, None
    except Exception as error:
        return None, f"checkpoint recovery failed: {type(error).__name__}: {error}"


def _interaction_matrix(loader) -> np.ndarray:
    return loader._dataset.inter_matrix(form="csr").toarray().astype(bool)


def _genre_features(dataset) -> np.ndarray:
    metadata_path = PROJECT_ROOT / "data" / "movie_metadata.tsv"
    genres_by_token: dict[str, set[str]] = {}
    with metadata_path.open(encoding="utf-8", newline="") as source:
        for row in csv.reader(source, delimiter="\t"):
            if len(row) >= 3:
                genres_by_token[row[0]] = {
                    genre.strip() for genre in row[2].split(",") if genre.strip()
                }

    tokens = dataset.field2id_token[dataset.iid_field]
    genres = sorted({
        genre for token in tokens[1:]
        for genre in genres_by_token.get(str(token), set())
    })
    if not genres:
        raise ValueError(f"No movie genres were loaded from {metadata_path}")
    genre_indices = {genre: index for index, genre in enumerate(genres)}
    features = np.zeros((dataset.item_num, len(genres)), dtype=np.float32)
    for item_id, token in enumerate(tokens[1:], start=1):
        for genre in genres_by_token.get(str(token), set()):
            features[item_id, genre_indices[genre]] = 1
    return features


def _load_cached_rankings(
    cache_path: Path,
    signature: str,
    fingerprint: str,
    k: int,
) -> np.ndarray | None:
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            valid = (
                str(cache["cache_version"].item()) == CACHE_VERSION
                and str(cache["split_signature"].item()) == signature
                and str(cache["model_fingerprint"].item()) == fingerprint
                and int(cache["topk"].item()) == k
            )
            return cache["rankings"].copy() if valid else None
    except (OSError, KeyError, ValueError):
        return None


def _save_cached_rankings(
    cache_path: Path,
    rankings: np.ndarray,
    signature: str,
    fingerprint: str,
    k: int,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        rankings=rankings,
        cache_version=CACHE_VERSION,
        split_signature=signature,
        model_fingerprint=fingerprint,
        topk=k,
    )


def _load_reference(
    requested: list[str],
    selected: dict[str, Any],
    recover_missing: bool,
):
    candidates = [name for name in requested if name in MODEL_SPECS]
    candidates.extend(name for name in MODEL_SPECS if name not in candidates)
    issues = []
    for name in candidates:
        checkpoint, reason = _ensure_checkpoint(name, selected, recover_missing)
        if checkpoint is None:
            issues.append(f"{name}: {reason}")
            continue
        try:
            loaded = load_data_and_model(checkpoint)
            config, model, dataset, train_data, valid_data, test_data = loaded
            signature = dataset_signature(dataset, (train_data, valid_data, test_data))
            matrices = (
                _interaction_matrix(train_data),
                _interaction_matrix(valid_data),
                _interaction_matrix(test_data),
            )
            return checkpoint, signature, loaded, matrices, issues
        except Exception as error:
            issues.append(f"{name}: could not load reference checkpoint ({error})")
    raise RuntimeError("No usable Task 1 checkpoint found. " + "; ".join(issues))


def _hybrid_scores(
    dataset,
    reference_signature: str,
    scores_file: Path,
) -> np.ndarray:
    hybrid_result = _read_json(PROJECT_ROOT / "results" / "weighted_hybrid.json")
    if hybrid_result.get("split_signature") != reference_signature:
        raise ValueError("WeightedHybrid split signature does not match Task 1 data")
    with np.load(scores_file, allow_pickle=False) as artifact:
        expected_users = np.asarray(
            dataset.field2id_token[dataset.uid_field], dtype=str,
        )
        expected_items = np.asarray(
            dataset.field2id_token[dataset.iid_field], dtype=str,
        )
        if (
            not np.array_equal(artifact["user_tokens"], expected_users)
            or not np.array_equal(artifact["item_tokens"], expected_items)
        ):
            raise ValueError("WeightedHybrid token mappings do not match Task 1 data")
    return load_hybrid(scores_file).scores.detach().cpu().numpy()


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_models(
    models: list[str] | None = None,
    topk: int = 10,
    recover_missing: bool = True,
) -> dict[str, Any]:
    """Run independent test ranking evaluation and write comparison artifacts."""
    if topk < 1:
        raise ValueError("topk must be positive")
    requested = models or DEFAULT_MODELS
    unknown = sorted(set(requested) - set(DEFAULT_MODELS))
    if unknown:
        raise ValueError(f"Unknown models {unknown}; choose from {DEFAULT_MODELS}")

    selected = _read_json(PROJECT_ROOT / "results" / "best_models.json")
    reference_path, signature, reference, matrices, skipped = _load_reference(
        requested, selected, recover_missing,
    )
    _, reference_model, dataset, _, _, test_data = reference
    train_matrix, _, test_matrix = matrices
    popularity = train_matrix.sum(axis=0, dtype=np.int64)
    features = _genre_features(dataset)
    user_ids = np.flatnonzero(test_matrix[:, 1:].any(axis=1))
    user_ids = user_ids[user_ids != 0]
    ground_truth = [set(np.flatnonzero(test_matrix[user])) - {0} for user in user_ids]
    results: list[dict[str, Any]] = []
    cache_dir = PROJECT_ROOT / "artifacts" / "task2" / "recommendations"

    for name in requested:
        score_path: Path | None = None
        model = None
        fingerprint: str
        try:
            if name == "WeightedHybrid":
                score_path = PROJECT_ROOT / "artifacts" / "hybrid" / "WeightedHybrid.npz"
                if not score_path.exists():
                    skipped.append(f"{name}: artifact is missing")
                    continue
                fingerprint = _file_hash(score_path)
                scores = None
            else:
                checkpoint, reason = _ensure_checkpoint(name, selected, recover_missing)
                if checkpoint is None:
                    skipped.append(f"{name}: {reason}")
                    continue
                score_path = checkpoint
                fingerprint = _file_hash(checkpoint)
                scores = None

            cache_path = cache_dir / f"{name}-top{topk}.npz"
            rankings = _load_cached_rankings(cache_path, signature, fingerprint, topk)
            if rankings is None:
                if name == "WeightedHybrid":
                    scores = _hybrid_scores(dataset, signature, score_path)
                    rankings = rank_score_matrix(scores, test_data, dataset, topk)
                else:
                    if checkpoint.resolve() == reference_path.resolve():
                        model = reference_model
                    else:
                        config, model, model_dataset, train_data, valid_data, model_test_data = (
                            load_data_and_model(checkpoint)
                        )
                        model_signature = dataset_signature(
                            model_dataset, (train_data, valid_data, model_test_data),
                        )
                        if model_signature != signature:
                            raise ValueError("checkpoint uses a different split or token mapping")
                    rankings = collect_test_rankings(model, test_data, dataset, topk)
                _save_cached_rankings(cache_path, rankings, signature, fingerprint, topk)

            user_recommendations = rankings_for_users(rankings, user_ids)
            row = {"model": name}
            row.update(accuracy_metrics(user_recommendations, ground_truth, topk))
            row.update(beyond_accuracy_metrics(
                user_recommendations, popularity, features, topk,
            ))
            results.append(row)
            print(f"Evaluated {name}", flush=True)
        except Exception as error:
            skipped.append(f"{name}: {type(error).__name__}: {error}")
            print(f"Skipped {name}: {type(error).__name__}: {error}", flush=True)
        finally:
            if model is not None and model is not reference_model:
                del model

    if not results:
        raise RuntimeError("No models were evaluated. " + "; ".join(skipped))

    output_dir = PROJECT_ROOT / "results" / "task2"
    accuracy_fields = ["model", *[f"{metric}@{topk}" for metric in (
        "Precision", "Recall", "NDCG", "MRR", "Hit",
    )]]
    beyond_fields = [f"Coverage@{topk}", f"IntraListDiversity@{topk}",
                     f"Novelty@{topk}", f"AveragePopularity@{topk}"]
    all_fields = [*accuracy_fields, *beyond_fields]
    _write_csv(output_dir / "all_metrics.csv", results, all_fields)
    summary = {
        "protocol": "MovieLens 100K, seed 2020, user-grouped random-order 80/10/10 split",
        "evaluation_split": "test",
        "topk": topk,
        "selected_models": requested,
        "evaluated_models": [row["model"] for row in results],
        "skipped_models": skipped,
        "reference_checkpoint": reference_path.relative_to(PROJECT_ROOT).as_posix(),
        "split_signature": signature,
        "aggregation": "macro average over users with at least one test interaction; coverage is catalog-wide",
        "beyond_accuracy_definitions": {
            "Coverage": "unique top-K recommendations divided by nonpadding catalog size",
            "IntraListDiversity": "macro mean pairwise genre cosine distance per user's list; singleton lists score 0",
            "Novelty": "mean -log2((training item count + 1)/(training interactions + catalog size)) in bits",
            "AveragePopularity": "mean raw training interaction count per recommended item",
        },
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS,
                        help="Default: all Task 1 models plus WeightedHybrid if available")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--no-recover-missing", action="store_true",
                        help="Skip absent checkpoints instead of training with saved best parameters")
    args = parser.parse_args()
    if args.topk < 1:
        parser.error("topk must be positive")
    evaluate_models(args.models, args.topk, not args.no_recover_missing)


if __name__ == "__main__":
    main()