import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.segmentation import (FEATURES, fit_candidate_models, load_artifact,
                              predict_customer, save_artifact, train_run,
                              validate_dataset, dataset_fingerprint)


def sample_data():
    return pd.DataFrame({FEATURES[0]: [20, 22, 25, 100, 105, 110],
                         FEATURES[1]: [20, 22, 18, 85, 90, 88],
                         "CustomerID": range(1, 7), "Age": [20, 21, 22, 55, 56, 57]})


class SegmentationTests(unittest.TestCase):
    def test_cleaning_and_invalid_values_reported(self):
        df = sample_data(); df[FEATURES[0]] = df[FEATURES[0]].astype(object)
        df.loc[0, FEATURES[0]] = np.inf; df.loc[1, FEATURES[0]] = "bad"
        clean, summary = validate_dataset(df, FEATURES)
        self.assertEqual(len(clean), 4)
        self.assertEqual(summary["invalid_numeric_values"], 2)

    def test_missing_feature_and_identifier_rejected(self):
        with self.assertRaises(ValueError): validate_dataset(sample_data(), ["missing"])
        with self.assertRaises(ValueError): validate_dataset(sample_data(), ["CustomerID"])

    def test_training_prediction_and_artifact_round_trip(self):
        run = train_run(sample_data(), FEATURES, min_k=2, max_k=3)
        self.assertEqual(set(run["results"].columns), {"k", "inertia", "silhouette_score"})
        cluster = predict_customer(run, {FEATURES[0]: 21, FEATURES[1]: 21})
        self.assertIn(cluster, range(1, run["selected_k"] + 1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkl"
            save_artifact(run, path)
            loaded = load_artifact(path)
            self.assertEqual(predict_customer(loaded, {FEATURES[0]: 21, FEATURES[1]: 21}), cluster)

    def test_small_dataset_and_invalid_k_range(self):
        with self.assertRaises(ValueError): fit_candidate_models(sample_data().head(2), min_k=2, max_k=2)
        with self.assertRaises(ValueError): fit_candidate_models(sample_data(), min_k=1, max_k=3)

    def test_dataset_fingerprint_includes_schema_and_values(self):
        base = sample_data()
        changed = base.copy(); changed.loc[0, FEATURES[0]] = 21
        reordered = base[[FEATURES[1], FEATURES[0], "CustomerID", "Age"]]
        self.assertNotEqual(dataset_fingerprint(base), dataset_fingerprint(changed))
        self.assertNotEqual(dataset_fingerprint(base), dataset_fingerprint(reordered))

    def test_categorical_feature_order_reused_for_prediction(self):
        df = sample_data()
        df["Channel"] = ["Retail", "Retail", "Retail", "Horeca", "Horeca", "Horeca"]
        ordered_features = ["Channel", FEATURES[1], FEATURES[0]]
        run = train_run(df, ordered_features, min_k=2, max_k=2)
        self.assertEqual(run["features"], ordered_features)
        self.assertEqual(list(run["preprocessor"].feature_names_in_), ordered_features)
        self.assertIn(predict_customer(run, {"Channel": "Retail", FEATURES[1]: 20, FEATURES[0]: 21}), [1, 2])


if __name__ == "__main__":
    unittest.main()
