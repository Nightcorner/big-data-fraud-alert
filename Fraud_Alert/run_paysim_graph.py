"""Small, explicitly sampled PaySim account graph for descriptive graph analytics.

This uses every Nth transaction across the file to control memory. It is not a
full-6.36M-edge scalable graph; for cluster-scale graph processing use GraphFrames
or GraphX in a Spark environment and document that separately.
"""
from __future__ import annotations
import json
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "data" / "PS_20174392719_1491204439457_log.csv"
OUT = ROOT / "outputs" / "graph"
OUT.mkdir(parents=True, exist_ok=True)
STRIDE = 64
CHUNK_SIZE = 250_000
pieces = []
offset = 0
for chunk in pd.read_csv(CSV, usecols=["step", "nameOrig", "nameDest", "amount", "isFraud"], chunksize=CHUNK_SIZE,
                        dtype={"step": "int16", "nameOrig": "string", "nameDest": "string", "amount": "float32", "isFraud": "int8"}):
    idx = np.arange(offset, offset + len(chunk), dtype=np.int64)
    pieces.append(chunk.loc[(idx % STRIDE) == 0].copy())
    offset += len(chunk)
    if offset % 1_000_000 < CHUNK_SIZE:
        print(f"Read {offset:,} rows; sampled {sum(len(x) for x in pieces):,}", flush=True)
sampled = pd.concat(pieces, ignore_index=True)
sampled.to_csv(OUT / "sampled_transaction_edges.csv", index=False)
edge = (sampled.groupby(["nameOrig", "nameDest"], as_index=False)
        .agg(transaction_count=("isFraud", "size"), fraud_count=("isFraud", "sum"),
             amount_sum=("amount", "sum")))
# Fraud amount aggregation aligned to each grouped edge.
fraud_rows = sampled.loc[sampled["isFraud"] == 1].groupby(["nameOrig", "nameDest"], as_index=False).agg(fraud_amount_sum=("amount", "sum"))
edge = edge.merge(fraud_rows, on=["nameOrig", "nameDest"], how="left")
edge["fraud_amount_sum"] = edge["fraud_amount_sum"].fillna(0.0)
edge.to_csv(OUT / "account_edge_aggregates.csv", index=False)
G = nx.DiGraph()
for row in edge.itertuples(index=False):
    G.add_edge(str(row.nameOrig), str(row.nameDest), transaction_count=int(row.transaction_count),
               fraud_count=int(row.fraud_count), amount_sum=float(row.amount_sum), fraud_amount_sum=float(row.fraud_amount_sum),
               weight=int(row.transaction_count))
indegree = sorted(G.in_degree(weight="weight"), key=lambda x: x[1], reverse=True)[:50]
outdegree = sorted(G.out_degree(weight="weight"), key=lambda x: x[1], reverse=True)[:50]
pd.DataFrame(indegree, columns=["account", "weighted_in_degree"]).to_csv(OUT / "top50_accounts_by_in_degree.csv", index=False)
pd.DataFrame(outdegree, columns=["account", "weighted_out_degree"]).to_csv(OUT / "top50_accounts_by_out_degree.csv", index=False)
summary = {
    "dataset": CSV.name, "total_rows_in_source": int(offset), "sampling_method": f"systematic every {STRIDE}th row",
    "sampled_transactions": int(len(sampled)), "sampled_fraud_transactions": int(sampled["isFraud"].sum()),
    "nodes": int(G.number_of_nodes()), "directed_edges": int(G.number_of_edges()),
    "unique_source_accounts_sample": int(sampled["nameOrig"].nunique()),
    "unique_destination_accounts_sample": int(sampled["nameDest"].nunique()),
    "weighted_in_degree_top10": [{"account": str(a), "weight": int(w)} for a,w in indegree[:10]],
    "weighted_out_degree_top10": [{"account": str(a), "weight": int(w)} for a,w in outdegree[:10]],
    "interpretation_limit": "Descriptive graph on systematic sample only; account degree is not proof of fraud and the sample over/under-represents rare patterns depending on row order. For full-scale graph analytics, use Spark GraphFrames/GraphX or another distributed graph engine.",
}
(OUT / "graph_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
