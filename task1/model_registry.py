from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECOBOLE_ROOT = PROJECT_ROOT / "third_party" / "RecBole_DSAIT4335"
COMMON_CONFIG = PROJECT_ROOT / "task1" / "configs" / "common.yaml"
MODEL_CONFIG_DIR = PROJECT_ROOT / "task1" / "configs" / "models"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    recbole_model: str
    config_file: Path


def _spec(
    name: str,
    recbole_model: str | None = None,
) -> ModelSpec:
    return ModelSpec(
        name=name,
        recbole_model=recbole_model or name,
        config_file=MODEL_CONFIG_DIR / f"{name}.yaml",
    )


MODEL_SPECS = {
    spec.name: spec
    for spec in (
        _spec("Random"),
        _spec("Pop"),
        _spec("ItemKNN"),
        _spec("UserKNN", recbole_model="ItemKNN"),
        _spec("BPR"),
        _spec("FISM"),
        _spec("SLIMElastic"),
        _spec("EASE"),
        _spec("NeuMF"),
        _spec("NGCF"),
        _spec("LightGCN"),
    )
}

def resolve_models(names: list[str] | None) -> list[ModelSpec]:
    if not names:
        return list(MODEL_SPECS.values())

    unknown = sorted(set(names) - set(MODEL_SPECS))
    if unknown:
        choices = ", ".join(MODEL_SPECS)
        raise ValueError(f"Unknown models {unknown}. Available models: {choices}")
    return [MODEL_SPECS[name] for name in names]


def config_files(spec: ModelSpec) -> list[str]:
    return [str(COMMON_CONFIG), str(spec.config_file)]

