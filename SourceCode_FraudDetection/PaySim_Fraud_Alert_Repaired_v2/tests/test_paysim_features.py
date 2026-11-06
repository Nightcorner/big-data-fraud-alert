from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paysim_features import build_features, CORE_FEATURES, BALANCE_FEATURES


class PaySimFeatureTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "step": [1, 2, 3],
            "type": ["TRANSFER", "CASH_OUT", "PAYMENT"],
            "amount": [100.0, 50.0, 10.0],
            "oldbalanceOrg": [100.0, 100.0, 0.0],
            "oldbalanceDest": [0.0, 25.0, 0.0],
            "isFraud": [1, 0, 0],
            "isFlaggedFraud": [0, 0, 0],
            "newbalanceOrig": [0.0, 50.0, 0.0],
            "newbalanceDest": [100.0, 75.0, 0.0],
        })

    def test_target_and_post_transaction_columns_are_excluded(self):
        X = build_features(self.df, include_balance_features=True)
        forbidden = {"isFraud", "isFlaggedFraud", "newbalanceOrig", "newbalanceDest", "nameOrig", "nameDest"}
        self.assertFalse(forbidden.intersection(X.columns))

    def test_balance_feature_equality_is_computed_from_pre_event_fields(self):
        X = build_features(self.df, include_balance_features=True)
        self.assertEqual(int(X.loc[0, "amount_equals_oldbalanceOrg"]), 1)
        self.assertEqual(int(X.loc[1, "amount_equals_oldbalanceOrg"]), 0)

    def test_core_ablation_does_not_require_balances(self):
        X = build_features(self.df, include_balance_features=False)
        self.assertEqual(list(X.columns), CORE_FEATURES)
        self.assertNotIn("oldbalanceOrg", X.columns)

    def test_feature_matrix_has_no_infinity(self):
        X = build_features(self.df, include_balance_features=True)
        numeric = X.select_dtypes(include=["number"])
        self.assertTrue(np.isfinite(numeric.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
