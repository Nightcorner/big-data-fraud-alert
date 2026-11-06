from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "data" / "PS_20174392719_1491204439457_log.csv"
OUT = ROOT / "outputs" / "audit"
OUT.mkdir(parents=True, exist_ok=True)

# Chunked audit: safe for limited memory and does not load the whole CSV at once.
rows = fraud = flagged = flagged_fraud = eq_fraud = eq_nonfraud = 0
missing_counts = {}
step_counts = {}
type_counts = {}
fraud_type_counts = {}
columns = None
chunksize = 250_000
for chunk in pd.read_csv(CSV, chunksize=chunksize):
    if columns is None:
        columns = list(chunk.columns)
        missing_counts = {c: 0 for c in columns}
    rows += len(chunk)
    fraud += int(chunk["isFraud"].sum())
    flagged += int(chunk["isFlaggedFraud"].sum())
    flagged_fraud += int(((chunk["isFlaggedFraud"] == 1) & (chunk["isFraud"] == 1)).sum())
    eq = (chunk["amount"] == chunk["oldbalanceOrg"])
    eq_fraud += int((eq & (chunk["isFraud"] == 1)).sum())
    eq_nonfraud += int((eq & (chunk["isFraud"] == 0)).sum())
    for c in columns:
        missing_counts[c] += int(chunk[c].isna().sum())
    for k, v in chunk["type"].value_counts().items():
        type_counts[str(k)] = type_counts.get(str(k), 0) + int(v)
    for k, v in chunk.loc[chunk.isFraud == 1, "type"].value_counts().items():
        fraud_type_counts[str(k)] = fraud_type_counts.get(str(k), 0) + int(v)
    for k, v in chunk.groupby("step", observed=False)["isFraud"].sum().items():
        step_counts[int(k)] = step_counts.get(int(k), 0) + int(v)

summary = {
    "dataset": CSV.name,
    "rows": rows,
    "columns": columns,
    "fraud_count": fraud,
    "nonfraud_count": rows - fraud,
    "fraud_rate_percent": 100 * fraud / rows if rows else 0,
    "missing_counts": missing_counts,
    "type_counts": type_counts,
    "fraud_by_type": fraud_type_counts,
    "steps_min": min(step_counts) if step_counts else None,
    "steps_max": max(step_counts) if step_counts else None,
    "steps_with_records": len(step_counts),
    "isFlaggedFraud_count": flagged,
    "isFlaggedFraud_and_fraud_count": flagged_fraud,
    "amount_equals_oldbalanceOrg_fraud_count": eq_fraud,
    "amount_equals_oldbalanceOrg_nonfraud_count": eq_nonfraud,
    "leakage_policy": {
        "target_isFraud": "never use as model input",
        "isFlaggedFraud": "excluded: existing detection flag could encode prior rule-based screening",
        "newbalanceOrig/newbalanceDest": "excluded: post-transaction balances may not be available at alert decision time",
        "oldbalanceOrg/oldbalanceDest": "included only in expanded pre-event feature experiment; operational availability must be justified",
    },
}
(OUT / "paysim_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
pd.DataFrame([{"column": k, "missing_count": v} for k, v in missing_counts.items()]).to_csv(OUT / "missing_values.csv", index=False)
pd.DataFrame([{"type": k, "count": v, "fraud_count": fraud_type_counts.get(k, 0)} for k, v in type_counts.items()]).to_csv(OUT / "type_distribution.csv", index=False)
print(json.dumps(summary, indent=2))
print(f"\nAudit tersimpan di {OUT}")
