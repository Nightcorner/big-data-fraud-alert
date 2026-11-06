"""Leakage-aware feature preparation for the synthetic PaySim dataset.

Only attributes available before/at the simulated transaction are used:
step, type, amount, oldbalanceOrg, oldbalanceDest. The post-transaction
balances (newbalanceOrig/newbalanceDest), target isFraud, and existing flag
isFlaggedFraud are deliberately excluded from model inputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "PS_20174392719_1491204439457_log.csv"
TARGET = "isFraud"
REQUIRED_SOURCE_COLUMNS = {
    "step", "type", "amount", "oldbalanceOrg", "oldbalanceDest", TARGET
}
POST_EVENT_OR_LEAKAGE_COLUMNS = {
    "newbalanceOrig", "newbalanceDest", "isFlaggedFraud", TARGET,
    "nameOrig", "nameDest"
}
CATEGORICAL_FEATURES = ["type"]
CORE_NUMERIC_FEATURES = ["step", "amount", "log_amount"]
BALANCE_NUMERIC_FEATURES = [
    "oldbalanceOrg", "oldbalanceDest", "log_oldbalanceOrg", "log_oldbalanceDest",
    "amount_equals_oldbalanceOrg", "amount_to_oldbalanceOrg_ratio",
    "origin_balance_sufficient", "destination_balance_zero",
]
CORE_FEATURES = CORE_NUMERIC_FEATURES + CATEGORICAL_FEATURES
BALANCE_FEATURES = CORE_FEATURES + BALANCE_NUMERIC_FEATURES


def load_paysim(path: Path | str = DATA_PATH, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    """Load PaySim fields needed for an honest point-in-time experiment."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"PaySim CSV tidak ditemukan: {path}. Simpan file PS_20174392719_1491204439457_log.csv di folder data/."
        )
    columns = list(usecols or sorted(REQUIRED_SOURCE_COLUMNS | {"newbalanceOrig", "newbalanceDest", "isFlaggedFraud"}))
    dtype_map = {
        "step": "int16",
        "type": "category",
        "amount": "float32",
        "oldbalanceOrg": "float32",
        "newbalanceOrig": "float32",
        "oldbalanceDest": "float32",
        "newbalanceDest": "float32",
        "isFraud": "int8",
        "isFlaggedFraud": "int8",
    }
    selected_dtype = {k: v for k, v in dtype_map.items() if k in columns}
    df = pd.read_csv(path, usecols=columns, dtype=selected_dtype, low_memory=False)
    missing = sorted(REQUIRED_SOURCE_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Kolom PaySim yang diperlukan tidak ditemukan: {missing}")
    if df[TARGET].isna().any():
        raise ValueError("Kolom target isFraud mengandung nilai kosong.")
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("isFraud harus berisi label biner 0/1.")
    return df


def build_features(df: pd.DataFrame, include_balance_features: bool = True) -> pd.DataFrame:
    """Build only pre-transaction features; never include the target or post-event outputs."""
    needed = {"step", "type", "amount"}
    if include_balance_features:
        needed.update({"oldbalanceOrg", "oldbalanceDest"})
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"Kolom sumber untuk fitur tidak lengkap: {missing}")

    out = pd.DataFrame(index=df.index)
    out["step"] = pd.to_numeric(df["step"], errors="raise").astype("float32")
    amount = pd.to_numeric(df["amount"], errors="raise").astype("float32").clip(lower=0)
    out["amount"] = amount
    out["log_amount"] = np.log1p(amount).astype("float32")
    # Category schema is fixed by PaySim; no target-dependent operation is used.
    out["type"] = pd.Categorical(
        df["type"].astype(str),
        categories=["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
    )

    if include_balance_features:
        orig = pd.to_numeric(df["oldbalanceOrg"], errors="raise").astype("float32").clip(lower=0)
        dest = pd.to_numeric(df["oldbalanceDest"], errors="raise").astype("float32").clip(lower=0)
        out["oldbalanceOrg"] = orig
        out["oldbalanceDest"] = dest
        out["log_oldbalanceOrg"] = np.log1p(orig).astype("float32")
        out["log_oldbalanceDest"] = np.log1p(dest).astype("float32")
        # This equality pattern is a strong PaySim simulator artifact. It is
        # computable from pre-event fields, but must not be generalized to real banks.
        out["amount_equals_oldbalanceOrg"] = np.isclose(
            amount.to_numpy(dtype=np.float32), orig.to_numpy(dtype=np.float32), rtol=1e-5, atol=0.01
        ).astype("int8")
        out["amount_to_oldbalanceOrg_ratio"] = (
            amount / orig.where(orig > 0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e4, 1e4).astype("float32")
        out["origin_balance_sufficient"] = (orig >= amount).astype("int8")
        out["destination_balance_zero"] = (dest == 0).astype("int8")

    # Defensive guard against accidental target/leakage inclusion.
    forbidden = POST_EVENT_OR_LEAKAGE_COLUMNS.intersection(out.columns)
    if forbidden:
        raise AssertionError(f"Target/leakage fields entered the feature matrix: {sorted(forbidden)}")
    return out


def split_by_step(df: pd.DataFrame, train_end_step: int = 520, validation_end_step: int = 600):
    """Strict forward-time split: early steps train, middle validation, latest steps test."""
    step = pd.to_numeric(df["step"], errors="raise")
    train_mask = step <= train_end_step
    val_mask = (step > train_end_step) & (step <= validation_end_step)
    test_mask = step > validation_end_step
    parts = (df.loc[train_mask], df.loc[val_mask], df.loc[test_mask])
    if any(part.empty for part in parts):
        raise ValueError("Pembagian berdasarkan step menghasilkan partisi kosong; periksa dataset atau cutoff.")
    for name, part in zip(("train", "validation", "test"), parts):
        if part[TARGET].nunique() < 2:
            raise ValueError(f"Partisi {name} tidak memiliki kedua kelas; sesuaikan cutoff dengan catatan metodologi.")
    if parts[0]["step"].max() >= parts[1]["step"].min() or parts[1]["step"].max() >= parts[2]["step"].min():
        raise AssertionError("Split kronologis tidak ketat.")
    return parts
