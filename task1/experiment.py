from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch

from task1.model_registry import PROJECT_ROOT, RECOBOLE_ROOT, ModelSpec, config_files


if not RECOBOLE_ROOT.exists():
    raise RuntimeError(
        "RecBole submodule is missing. Run: "
        "git submodule update --init --recursive"
    )

sys.path.insert(0, str(RECOBOLE_ROOT))

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_logger, init_seed


DATASET = "ml-100k"


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _reset_logger() -> None:
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


def build_config(
    spec: ModelSpec,
    overrides: dict[str, Any] | None = None,
) -> Config:
    config_dict = {
        "data_path": str(RECOBOLE_ROOT / "dataset"),
        "checkpoint_dir": str(PROJECT_ROOT / "artifacts" / "checkpoints"),
    }
    if overrides:
        config_dict.update(overrides)

    return Config(
        model=spec.recbole_model,
        dataset=DATASET,
        config_file_list=config_files(spec),
        config_dict=config_dict,
    )


def train_model(
    spec: ModelSpec,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one candidate and return its validation result and checkpoint."""
    _reset_logger()
    config = build_config(spec, overrides)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    started = time.perf_counter()
    dataset = create_dataset(config)
    train_data, valid_data, _ = data_preparation(config, dataset)

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        verbose=False,
        saved=True,
        show_progress=config["show_progress"],
    )

    checkpoint = None
    candidate = Path(trainer.saved_model_file).resolve()
    if candidate.exists():
        checkpoint = candidate.relative_to(PROJECT_ROOT).as_posix()

    return {
        "model": spec.name,
        "recbole_model": spec.recbole_model,
        "parameters": dict(overrides or {}),
        "best_valid_score": float(best_valid_score),
        "best_valid_result": {
            key: float(value) for key, value in best_valid_result.items()
        },
        "checkpoint": checkpoint,
        "duration_seconds": time.perf_counter() - started,
        "seed": int(config["seed"]),
        "device": str(config["device"]),
    }


def evaluate_checkpoint(checkpoint: str | Path) -> dict[str, float]:
    """Evaluate exactly one already selected checkpoint on the test set."""
    _reset_logger()
    config, model, _, _, _, test_data = load_data_and_model(str(checkpoint))
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    result = trainer.evaluate(
        test_data,
        load_best_model=False,
        show_progress=False,
    )
    return {key: float(value) for key, value in result.items()}


def load_data_and_model(model_file: str | Path):
    """Load a checkpoint without importing RecBole's optional Ray integration."""
    checkpoint = torch.load(
        resolve_project_path(model_file),
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    config.compatibility_settings()
    config["data_path"] = str(RECOBOLE_ROOT / "dataset" / DATASET)
    config["checkpoint_dir"] = str(PROJECT_ROOT / "artifacts" / "checkpoints")
    config["use_gpu"] = False
    config["device"] = torch.device("cpu")
    _reset_logger()
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))
    return config, model, dataset, train_data, valid_data, test_data

