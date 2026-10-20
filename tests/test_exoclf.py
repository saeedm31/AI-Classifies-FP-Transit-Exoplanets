"""Tests. Run with: python -m unittest"""

import json
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import exoclf


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.stats = exoclf.load_clean_data()

    def test_no_duplicates_or_conflicting_labels_left(self):
        self.assertFalse(self.df.duplicated(exoclf.INPUT_FEATURES).any())

    def test_stats_add_up(self):
        s = self.stats
        self.assertEqual(s["raw_rows"], s["clean_rows"] + s["removed_duplicates"] + s["removed_conflicting"])
        self.assertEqual(sum(s["class_counts"].values()), s["clean_rows"])

    def test_metrics_file_matches_the_data(self):
        with open("metrics.json") as f:
            m = json.load(f)
        self.assertEqual(m["data"]["clean_rows"], self.stats["clean_rows"])

    def test_model_beats_the_majority_baseline(self):
        y = (self.df[exoclf.LABEL] == exoclf.POSITIVE).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            self.df, y, test_size=0.25, stratify=y, random_state=exoclf.RANDOM_STATE)
        model = exoclf.train(X_train)
        p = exoclf.predict_fp_probability(model, X_test[exoclf.INPUT_FEATURES])
        self.assertTrue(np.all((p >= 0) & (p <= 1)))
        self.assertGreater(roc_auc_score(y_test, p), 0.9)
        accuracy = ((p >= 0.5).astype(int) == y_test.to_numpy()).mean()
        self.assertGreater(accuracy, y_test.mean())   # better than always saying "FP"


class FeatureTests(unittest.TestCase):
    def test_transit_shape_is_impact_plus_radius_ratio(self):
        df = pd.DataFrame({"koi_impact": [0.5], "koi_ror": [0.1]})
        self.assertAlmostEqual(exoclf.add_features(df)["transit_shape"].iloc[0], 0.6)

    def test_red_flags(self):
        planet = {"koi_impact": 0.3, "koi_ror": 0.02, "koi_depth": 500}
        binary = {"koi_impact": 0.95, "koi_ror": 0.4, "koi_depth": 50000}
        self.assertEqual(exoclf.red_flags(planet), [])
        self.assertEqual(len(exoclf.red_flags(binary)), 3)


if __name__ == "__main__":
    unittest.main()
