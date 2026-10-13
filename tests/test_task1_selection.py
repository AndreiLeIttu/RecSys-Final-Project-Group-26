import unittest

from task1.model_registry import (
    INDIVIDUAL_MODEL_NAMES,
    PRIMARY_MODEL_NAMES,
    resolve_models,
)


class ModelSelectionTests(unittest.TestCase):
    def test_default_models_are_the_declared_primary_set(self):
        self.assertEqual(
            [spec.name for spec in resolve_models(None)],
            list(PRIMARY_MODEL_NAMES),
        )

    def test_supplementary_models_remain_explicitly_available(self):
        specs = resolve_models(["FISM", "NGCF", "FeatureCombination"])
        self.assertEqual(
            [spec.role for spec in specs],
            ["supplementary", "supplementary", "hybrid"],
        )

    def test_hybrid_can_preserve_the_original_individual_model_set(self):
        self.assertEqual(len(INDIVIDUAL_MODEL_NAMES), 11)
        self.assertNotIn("FeatureCombination", INDIVIDUAL_MODEL_NAMES)
        self.assertEqual(
            [spec.name for spec in resolve_models(list(INDIVIDUAL_MODEL_NAMES))],
            list(INDIVIDUAL_MODEL_NAMES),
        )


if __name__ == "__main__":
    unittest.main()
