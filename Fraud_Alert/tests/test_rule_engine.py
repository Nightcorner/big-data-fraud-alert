from __future__ import annotations

import sys
import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.business_rules import evaluate_business_rules


class BusinessRuleEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = pd.read_csv(ROOT / "data" / "synthetic_rule_scenarios.csv", dtype={"transaction_id": str, "user_id": str})
        cls.evaluation, cls.alerts = evaluate_business_rules(cls.fixture)
        cls.actual = {
            str(row.transaction_id): (set() if row.triggered_rule_ids == "NONE" else set(str(row.triggered_rule_ids).split("|")))
            for row in cls.evaluation.itertuples(index=False)
        }
        cls.expected = {
            str(row.transaction_id): (set() if str(row.expected_rules) == "NONE" else set(str(row.expected_rules).split("|")))
            for row in cls.fixture[["transaction_id", "expected_rules"]].itertuples(index=False)
        }

    def test_all_fixture_expectations_match(self):
        mismatches = {
            txid: {"expected": sorted(exp), "actual": sorted(self.actual.get(txid, set()))}
            for txid, exp in self.expected.items()
            if exp != self.actual.get(txid, set())
        }
        self.assertEqual(mismatches, {})

    def test_rule1_no_minimum_amount_and_approved_or_declined(self):
        self.assertEqual(self.actual["A002"], {"R1"})
        self.assertEqual(self.actual["A003"], {"R1"})

    def test_rule1_not_triggered_on_non_cash_or_pending_status(self):
        self.assertEqual(self.actual["A004"], set())
        self.assertEqual(self.actual["A007"], set())

    def test_rule2_strictly_less_than_90_minutes(self):
        self.assertEqual(self.actual["B002"], {"R2"})
        self.assertEqual(self.actual["B004"], set())

    def test_rule2_detects_later_other_country_event_and_references_withdrawal(self):
        self.assertEqual(self.actual["F001"], set())
        self.assertEqual(self.actual["F002"], {"R2"})
        record = self.alerts[(self.alerts["rule_id"] == "R2") & (self.alerts["transaction_id"] == "F001")]
        self.assertEqual(len(record), 1)
        self.assertEqual(record.iloc[0]["detected_by_transaction_id"], "F002")

    def test_rule3_fifth_foreign_cash_withdrawal(self):
        self.assertEqual(self.actual["C001"], {"R1"})
        self.assertEqual(self.actual["C005"], {"R3"})
        row = self.evaluation[self.evaluation["transaction_id"] == "C005"].iloc[0]
        self.assertEqual(int(row["foreign_cash_withdrawal_number_today"]), 5)

    def test_rule3_rupiah_equivalent_threshold_and_no_duplicate_amount_alert(self):
        self.assertEqual(self.actual["D001"], {"R1"})
        self.assertEqual(self.actual["D004"], {"R3"})
        self.assertEqual(self.actual["D005"], {"R3"})
        self.assertEqual(self.actual["D006"], set())
        self.assertEqual(self.actual["E004"], set())
        below = self.evaluation[self.evaluation["transaction_id"] == "E004"].iloc[0]
        self.assertEqual(float(below["cumulative_foreign_cash_withdrawal_amount_idr_today"]), 5999999.0)

    def test_rule3_excludes_domestic_spend_from_foreign_withdrawal_thresholds(self):
        # Four domestic purchases above Rp6m in aggregate followed by a small first
        # foreign withdrawal should send the customer notification, not a Fraud alert.
        self.assertEqual(self.actual["G005"], {"R1"})
        row = self.evaluation[self.evaluation["transaction_id"] == "G005"].iloc[0]
        self.assertEqual(int(row["transaction_number_today"]), 5)
        self.assertEqual(int(row["foreign_cash_withdrawal_number_today"]), 1)
        self.assertEqual(float(row["cumulative_foreign_cash_withdrawal_amount_idr_today"]), 10000.0)

    def test_alert_channels_match_rules(self):
        if self.alerts.empty:
            self.fail("Fixture harus menghasilkan setidaknya satu alert.")
        wrong = self.alerts[
            ((self.alerts["rule_id"] == "R1") & (self.alerts["notification_channel"] != "SMS_CUSTOMER"))
            | ((self.alerts["rule_id"].isin(["R2", "R3"])) & (self.alerts["notification_channel"] != "EMAIL_FRAUD_TEAM"))
        ]
        self.assertTrue(wrong.empty, wrong.to_dict(orient="records"))

    def test_missing_required_rule_fields_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_business_rules(self.fixture.drop(columns=["home_country"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
