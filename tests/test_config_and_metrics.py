import json
import unittest
from pathlib import Path

import numpy as np

from model.config import load_config
from model.metrics import classification_metrics, priority_weighted_score


ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_committed_config_is_valid(self):
        config = load_config(ROOT / "configs/preliminary_experiment.json")
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["data"]["image_size"], 224)

    def test_priority_weighted_score(self):
        score = priority_weighted_score(
            {"high": 0.80, "medium": 0.75, "low": 0.70},
            {"high": 0.5, "medium": 0.3, "low": 0.2},
        )
        self.assertAlmostEqual(score, 0.765)

    def test_merged_dataset_metadata_matches_protocol(self):
        config = load_config(ROOT / "configs/preliminary_experiment.json")
        with (ROOT / "notebooks/cache_224/cache_meta.json").open(encoding="utf-8") as stream:
            cache = json.load(stream)
        with (ROOT / "notebooks/preliminary_splits/splits_meta.json").open(
            encoding="utf-8"
        ) as stream:
            splits = json.load(stream)

        self.assertEqual(splits["seed"], config["seed"])
        self.assertEqual(cache["img_size"], config["data"]["image_size"])
        self.assertEqual(cache["pad_color"], config["data"]["cache_pad_color"])
        self.assertEqual(cache["splits_csv_sha256"], splits["splits_csv_sha256"])
        for group in config["groups"]:
            self.assertEqual(set(cache["groups"][group]["labels"]), set(splits["groups"][group]))
            self.assertEqual(
                sum(cache["groups"][group]["counts"].values()),
                cache["groups"][group]["n_images"],
            )

    def test_metrics_have_expected_shape(self):
        targets = np.array([0, 1, 2, 3, 4])
        probabilities = np.eye(5)
        metrics = classification_metrics(targets, probabilities, ["a", "b", "c", "d", "e"])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["expected_calibration_error"], 0.0)
        self.assertEqual(len(metrics["confusion_matrix"]), 5)


if __name__ == "__main__":
    unittest.main()
