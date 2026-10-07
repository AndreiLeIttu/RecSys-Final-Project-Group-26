from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import traceback
from typing import Any

import yaml

from task1.experiment import evaluate_checkpoint, resolve_project_path, train_model
from task1.io_utils import append_jsonl, flatten_result, read_jsonl, write_csv, write_json
from task1.model_registry import PROJECT_ROOT, ModelSpec, resolve_models


RESULTS_DIR = PROJECT_ROOT / "results"
SEARCH_FILE = PROJECT_ROOT / "task1" / "search_spaces.yaml"
TRIALS_FILE = RESULTS_DIR / "tuning_trials.jsonl"
BEST_JSON = RESULTS_DIR / "best_models.json"
BEST_CSV = RESULTS_DIR / "best_models.csv"
BEST_CONFIG_DIR = PROJECT_ROOT / "task1" / "configs" / "best"
BEST_CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "best"


def candidate_key(model: str, parameters: dict[str, Any]) -> str:
    payload = json.dumps([model, parameters], sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def load_candidates(spec: ModelSpec) -> list[dict[str, Any]]:
    with SEARCH_FILE.open(encoding="utf-8") as source:
        spaces = yaml.safe_load(source)
    candidates = spaces.get(spec.name)
    if candidates is None:
        raise KeyError(f"No search candidates defined for {spec.name}")
    return [dict(candidate or {}) for candidate in candidates]


def select_best(model: str, trials: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [
        trial
        for trial in trials
        if trial.get("model") == model
        and trial.get("status") == "ok"
        and trial.get("checkpoint")
        and resolve_project_path(trial["checkpoint"]).exists()
    ]
    if not successful:
        raise RuntimeError(f"No successful tuning trial is available for {model}")
    return max(successful, key=lambda trial: trial["best_valid_score"])


def finalize_best(spec: ModelSpec, best: dict[str, Any]) -> dict[str, Any]:
    source_checkpoint = resolve_project_path(best["checkpoint"])
    BEST_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    destination = BEST_CHECKPOINT_DIR / f"{spec.name}.pth"
    shutil.copy2(source_checkpoint, destination)

    final = dict(best)
    final["checkpoint"] = destination.relative_to(PROJECT_ROOT).as_posix()
    final["test_result"] = evaluate_checkpoint(destination)

    return final


def write_best_config(spec: ModelSpec, best: dict[str, Any]) -> None:
    with spec.config_file.open(encoding="utf-8") as source:
        config_values = yaml.safe_load(source) or {}

    BEST_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_values.update(dataset="ml-100k", **best["parameters"])
    with (BEST_CONFIG_DIR / f"{spec.name}.yaml").open("w", encoding="utf-8") as output:
        yaml.safe_dump(config_values, output, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune individual RecBole models using validation MRR@10."
    )
    parser.add_argument("--models", nargs="*", help="Model names; omit for all models")
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Run only the first N declared candidates per model",
    )
    args = parser.parse_args()

    completed = {
        row.get("candidate_key")
        for row in read_jsonl(TRIALS_FILE)
        if row.get("status") == "ok"
        and row.get("checkpoint")
        and resolve_project_path(row["checkpoint"]).exists()
    }

    selected_specs = resolve_models(args.models)
    for spec in selected_specs:
        candidates = load_candidates(spec)
        if args.max_trials is not None:
            candidates = candidates[: args.max_trials]

        for index, candidate in enumerate(candidates, start=1):
            overrides = {**candidate, "show_progress": False}
            key = candidate_key(spec.name, candidate)
            if key in completed:
                print(f"Skipping completed trial {spec.name} {index}/{len(candidates)}")
                continue

            print(f"Tuning {spec.name} {index}/{len(candidates)}: {candidate}")
            try:
                result = train_model(
                    spec,
                    overrides,
                )
                result.update(
                    {
                        "parameters": candidate,
                        "status": "ok",
                        "candidate_key": key,
                    }
                )
            except Exception as exc:
                result = {
                    "model": spec.name,
                    "recbole_model": spec.recbole_model,
                    "parameters": candidate,
                    "candidate_key": key,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                print(result["error"])
            append_jsonl(TRIALS_FILE, result)

    all_trials = read_jsonl(TRIALS_FILE)
    existing_best = {}
    if BEST_JSON.exists():
        existing_best = json.loads(BEST_JSON.read_text(encoding="utf-8"))

    finalized: dict[str, dict[str, Any]] = {}
    for spec in selected_specs:
        try:
            best = select_best(spec.name, all_trials)
            previous = existing_best.get(spec.name)
            unchanged = (
                previous
                and previous.get("candidate_key") == best.get("candidate_key")
                and previous.get("test_result")
                and previous.get("checkpoint")
                and resolve_project_path(previous["checkpoint"]).exists()
            )
            finalized[spec.name] = previous if unchanged else finalize_best(spec, best)
            write_best_config(spec, best)
        except RuntimeError as exc:
            print(exc)

    existing_best.update(finalized)
    for record in existing_best.values():
        checkpoint = record.get("checkpoint")
        if checkpoint:
            path = resolve_project_path(checkpoint)
            if path.is_relative_to(PROJECT_ROOT):
                record["checkpoint"] = path.relative_to(PROJECT_ROOT).as_posix()
    write_json(BEST_JSON, existing_best)
    write_csv(BEST_CSV, [flatten_result(row) for row in existing_best.values()])
    write_csv(RESULTS_DIR / "tuning_trials.csv", [flatten_result(row) for row in all_trials])

    print(f"Trial history: {TRIALS_FILE}")
    print(f"Selected models: {BEST_JSON}")


if __name__ == "__main__":
    main()

