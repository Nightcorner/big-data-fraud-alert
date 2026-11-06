import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "business_rules.json"
EXPECTED_SHA256 = "cfee32266363481083922efd5d25ef1a0d5a2de5192f7cf399c8382da78689d2"


class ConfigurationIntegrityTests(unittest.TestCase):
    def test_business_rule_configuration_integrity(self):
        actual = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
        self.assertEqual(actual, EXPECTED_SHA256)


if __name__ == "__main__":
    unittest.main()
