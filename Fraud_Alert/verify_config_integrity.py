#!/usr/bin/env python3
"""Verify the integrity of the business-rule configuration."""
from hashlib import sha256
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config" / "business_rules.json"
EXPECTED_SHA256 = "cfee32266363481083922efd5d25ef1a0d5a2de5192f7cf399c8382da78689d2"

actual = sha256(CONFIG.read_bytes()).hexdigest()
if actual != EXPECTED_SHA256:
    print("FAIL: business-rule configuration integrity check failed.")
    print(f"Expected SHA-256: {EXPECTED_SHA256}")
    print(f"Actual SHA-256:   {actual}")
    print("Review the business-rule configuration before using this package.")
    sys.exit(1)

print("PASS: business-rule configuration integrity verified.")
print(f"SHA-256: {actual}")
