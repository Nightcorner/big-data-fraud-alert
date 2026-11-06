import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "business_rules.json"
EXPECTED_SHA256 = "a9fa1fbccaa933f754221afcc5aa301bcd981c8caa1b126cb7b6820509ec5b02"


class RuleFreezeIntegrityTests(unittest.TestCase):
    def test_frozen_proposed_v2_configuration_has_not_changed(self):
        actual = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
        self.assertEqual(actual, EXPECTED_SHA256)


if __name__ == "__main__":
    unittest.main()
