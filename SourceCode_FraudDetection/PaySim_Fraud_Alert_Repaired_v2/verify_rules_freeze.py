#!/usr/bin/env python3
"""Verify that the frozen proposed-v2 business rules were not modified."""
from hashlib import sha256
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config" / "business_rules.json"
EXPECTED_SHA256 = "a9fa1fbccaa933f754221afcc5aa301bcd981c8caa1b126cb7b6820509ec5b02"

actual = sha256(CONFIG.read_bytes()).hexdigest()
if actual != EXPECTED_SHA256:
    print("FAIL: frozen business rule config has changed.")
    print(f"Expected SHA-256: {EXPECTED_SHA256}")
    print(f"Actual SHA-256:   {actual}")
    print("Create a versioned rule proposal instead of changing the frozen config silently.")
    sys.exit(1)

print("PASS: proposed-v2 business rule config matches the frozen SHA-256.")
print(f"SHA-256: {actual}")
