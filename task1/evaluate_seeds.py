"""Evaluate validation-selected Task 1 models across multiple random seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import traceback
from typing import Any

from task1.experiment import evaluate_checkpoint, resolve_project_path, train_model
from task1.io_utils import append_jsonl, read_jsonl, write_csv, write_json
from task1.model_registry import PROJECT_ROOT, resolve_models


RESULTS_DIR = PROJECT_ROOT / "results"
BEST_JSON = RESULTS_DIR / "best_models.json"
RUNS_JSONL = RESULTS_DIR / "seed_runs.jsonl"
METRICS = ("recall@10", "mrr@10", "ndcg@10", "hit@10", "precision@10")


def run_key(model: str, parameters: dict[str, Any], seed: int) -> str:
    payload = json.dumps([model, parameters, seed], sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def flatten_run(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        "model": record["model"],
        "family": record["family"],
        "role": record["role"],
        "seed": record["seed"],
        "parameters": json.dumps(record["parameters"], sort_keys=True),
        "source": record["source"],
        "duration_seconds": record.get("duration_seconds"),
        "status": record["status"],
        "error": record.get("error"),
    }
    for prefix in ("best_valid_result", "test_result"):
        for metric, value in (record.get(prefix) or {}).items():
            row[f"{prefix}.{metric}"] = value
    return row


def summarize(specs, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        model_runs = [
            run for run in runs if run["model"] == spec.name and run["status"] == "ok"
        ]
        if not model_runs:
            continue
        row = {
            "model": spec.name,
            "family": spec.family,
            "role": spec.role,
            "seeds": ",".join(str(run["seed"]) for run in model_runs),
            "n_seeds": len(model_runs),
            "parameters": json.dumps(model_runs[0]["parameters"], sort_keys=True),
        }
        valid_values = [run["best_valid_result"]["mrr@10"] for run in model_runs]
        row["valid_mrr@10_mean"], row["valid_mrr@10_std"] = mean_std(valid_values)
        for metric in METRICS:
            values = [run["test_result"][metric] for run in model_runs]
            row[f"test_{metric}_mean"], row[f"test_{metric}_std"] = mean_std(values)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain fixed validation-selected settings across seeds."
    )
    parser.add_argument("--models", nargs="*", help="Default: primary Task 1 models")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2020, 2021, 2022])
    args = parser.parse_args()

    specs = resolve_models(args.models)
    selected = json.loads(BEST_JSON.read_text(encoding="utf-8"))
    missing = [spec.name for spec in specs if spec.name not in selected]
    if missing:
        parser.error(f"Run task1.tune_models first; missing selections for {missing}")

    expected_keys = {
        run_key(spec.name, selected[spec.name]["parameters"], seed)
        for spec in specs
        for seed in args.seeds
    }
    existing = {
        row["run_key"]: row
        for row in read_jsonl(RUNS_JSONL)
        if row.get("run_key") in expected_keys and row.get("status") == "ok"
    }

    for spec in specs:
        selection = selected[spec.name]
        parameters = selection["parameters"]
        for seed in args.seeds:
            key = run_key(spec.name, parameters, seed)
            if key in existing:
                print(f"Skipping completed seed run {spec.name} seed={seed}")
                continue

            checkpoint = resolve_project_path(selection["checkpoint"])
            reuse_selected = seed == selection["seed"] and checkpoint.exists()
            print(f"Evaluating {spec.name} seed={seed}", flush=True)
            try:
                if reuse_selected:
                    record = {
                        "best_valid_result": selection["best_valid_result"],
                        "test_result": selection["test_result"],
                        "duration_seconds": selection.get("duration_seconds"),
                        "source": "selected tuning checkpoint",
                    }
                else:
                    trained = train_model(
                        spec,
                        {**parameters, "seed": seed, "show_progress": False},
                    )
                    if not trained.get("checkpoint"):
                        raise RuntimeError("Training did not produce a checkpoint")
                    record = {
                        "best_valid_result": trained["best_valid_result"],
                        "test_result": evaluate_checkpoint(trained["checkpoint"]),
                        "duration_seconds": trained["duration_seconds"],
                        "source": "retrained selected configuration",
                    }
                record.update({
                    "model": spec.name,
                    "family": spec.family,
                    "role": spec.role,
                    "parameters": parameters,
                    "seed": seed,
                    "run_key": key,
                    "status": "ok",
                })
            except Exception as exc:
                record = {
                    "model": spec.name,
                    "family": spec.family,
                    "role": spec.role,
                    "parameters": parameters,
                    "seed": seed,
                    "run_key": key,
                    "source": "retrained selected configuration",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                print(record["error"])
            append_jsonl(RUNS_JSONL, record)

    latest = {
        row["run_key"]: row
        for row in read_jsonl(RUNS_JSONL)
        if row.get("run_key") in expected_keys
    }
    failed = [row for row in latest.values() if row.get("status") != "ok"]
    missing_keys = expected_keys - set(latest)
    if failed or missing_keys:
        failed_runs = [f"{row['model']} seed={row['seed']}" for row in failed]
        raise RuntimeError(
            f"Seed evaluation incomplete: failed={failed_runs}, "
            f"missing={len(missing_keys)}. See {RUNS_JSONL.relative_to(PROJECT_ROOT)}."
        )
    runs = [
        latest[run_key(spec.name, selected[spec.name]["parameters"], seed)]
        for spec in specs
        for seed in args.seeds
        if run_key(spec.name, selected[spec.name]["parameters"], seed) in latest
    ]
    summary = summarize(specs, runs)
    write_json(RESULTS_DIR / "seed_results.json", runs)
    write_csv(RESULTS_DIR / "seed_results.csv", [flatten_run(run) for run in runs])
    write_json(RESULTS_DIR / "model_summary.json", summary)
    write_csv(RESULTS_DIR / "model_summary.csv", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
