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
    family: str
    role: str = "candidate"


def _spec(
    name: str,
    family: str,
    recbole_model: str | None = None,
    role: str = "candidate",
) -> ModelSpec:
    return ModelSpec(
        name=name,
        recbole_model=recbole_model or name,
        config_file=MODEL_CONFIG_DIR / f"{name}.yaml",
        family=family,
        role=role,
    )


MODEL_SPECS = {
    spec.name: spec
    for spec in (
        _spec("Random", "baseline", role="baseline"),
        _spec("Pop", "baseline", role="baseline"),
        _spec("ItemKNN", "neighborhood"),
        _spec("UserKNN", "neighborhood", recbole_model="ItemKNN"),
        _spec("BPR", "matrix factorization"),
        _spec("FISM", "latent item similarity", role="supplementary"),
        _spec("SLIMElastic", "linear item-item"),
        _spec("EASE", "linear item-item"),
        _spec("NeuMF", "neural interaction"),
        _spec("NGCF", "graph", role="supplementary"),
        _spec("LightGCN", "graph"),
        _spec("FeatureCombination", "hybrid", role="hybrid"),
    )
}

PRIMARY_MODEL_NAMES = (
    "Random",
    "Pop",
    "ItemKNN",
    "UserKNN",
    "BPR",
    "SLIMElastic",
    "EASE",
    "NeuMF",
    "LightGCN",
)

INDIVIDUAL_MODEL_NAMES = (
    "Random",
    "Pop",
    "ItemKNN",
    "UserKNN",
    "BPR",
    "FISM",
    "SLIMElastic",
    "EASE",
    "NeuMF",
    "NGCF",
    "LightGCN",
)


def resolve_models(names: list[str] | None) -> list[ModelSpec]:
    if not names:
        return [MODEL_SPECS[name] for name in PRIMARY_MODEL_NAMES]

    unknown = sorted(set(names) - set(MODEL_SPECS))
    if unknown:
        choices = ", ".join(MODEL_SPECS)
        raise ValueError(f"Unknown models {unknown}. Available models: {choices}")
    return [MODEL_SPECS[name] for name in names]


def config_files(spec: ModelSpec) -> list[str]:
    return [str(COMMON_CONFIG), str(spec.config_file)]

