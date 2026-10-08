"""Feature-combination hybrid model and feature preparation."""

# Keep the original class import available for checkpoints saved before the move.
__all__ = ["FeatureCombination"]


def __getattr__(name):
    if name == "FeatureCombination":
        from .model import FeatureCombination
        return FeatureCombination
    raise AttributeError(name)
