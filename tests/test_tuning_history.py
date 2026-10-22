import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from task1.io_utils import write_csv, write_csv_history


def trial(model, checkpoint, key):
    return {"model": model, "candidate_key": key, "seed": 2020,
            "checkpoint": checkpoint, "status": "ok", "error": None,
            "best_valid_score": 0.4}


class TuningHistoryTests(unittest.TestCase):
    def test_partial_history_preserves_old_runs_and_updates_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            previous = [trial("EASE", "old-ease.pth", "ease"),
                        trial("FeatureCombination", "old-feature.pth", "feature")]
            write_csv(path, previous)
            new = trial("FeatureCombination", "new-feature.pth", "feature")
            write_csv_history(path, [new])
            new["best_valid_score"] = 0.5
            write_csv_history(path, [new])
            with path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 3)
            self.assertEqual([r["checkpoint"] for r in rows],
                             ["old-ease.pth", "old-feature.pth", "new-feature.pth"])
            self.assertEqual(rows[-1]["best_valid_score"], "0.5")

    def test_empty_local_history_keeps_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            write_csv_history(path, [trial("EASE", "ease.pth", "ease")])
            before = path.read_bytes()
            write_csv_history(path, [])
            self.assertEqual(path.read_bytes(), before)

    def test_partial_tuning_without_jsonl_preserves_other_models(self):
        from task1 import tune_models

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "tuning_trials.csv"
            write_csv(history, [trial("EASE", "archived.pth", "ease")])
            checkpoint = root / "new-feature.pth"
            checkpoint.touch()
            spec = SimpleNamespace(name="FeatureCombination", recbole_model="FeatureCombination")
            record = trial("FeatureCombination", str(checkpoint), "unused")
            with (
                patch.object(sys, "argv", ["tune_models", "--models", "FeatureCombination"]),
                patch.object(tune_models, "RESULTS_DIR", root),
                patch.object(tune_models, "TRIALS_FILE", root / "tuning_trials.jsonl"),
                patch.object(tune_models, "BEST_JSON", root / "best.json"),
                patch.object(tune_models, "BEST_CSV", root / "best.csv"),
                patch.object(tune_models, "resolve_models", return_value=[spec]),
                patch.object(tune_models, "load_candidates", return_value=[{}]),
                patch.object(tune_models, "train_model", return_value=record),
                patch.object(tune_models, "finalize_best", side_effect=lambda _, result: result),
                patch.object(tune_models, "write_best_config"),
                redirect_stdout(io.StringIO()),
            ):
                tune_models.main()
            with history.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual([r["model"] for r in rows], ["EASE", "FeatureCombination"])


if __name__ == "__main__":
    unittest.main()
