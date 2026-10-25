import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

print("[*] Loading node and edge data...")
nodes_df = pd.read_csv(os.path.join(DATA_DIR, "Nodes_FraudCircle.csv"))
edges_df = pd.read_csv(os.path.join(DATA_DIR, "Edges_FraudCircle.csv"))
print(f"    Nodes: {len(nodes_df)}, Edges: {len(edges_df)}")

# ============================================================
# BUILD GRAPHS
# ============================================================
tx_types = ["TRANSFERRED_TO", "PAID_MERCHANT"]
tx_edges = edges_df[edges_df["edge_type"].isin(tx_types)].copy()

G_tx = nx.DiGraph()
for _, row in tx_edges.iterrows():
    if G_tx.has_edge(row["source"], row["target"]):
        G_tx[row["source"]][row["target"]]["weight"] += row["amount"]
    else:
        G_tx.add_edge(row["source"], row["target"], weight=row["amount"])

# Parse timestamps
edges_df["timestamp_dt"] = pd.to_datetime(edges_df["timestamp"], errors="coerce")
tx_edges["timestamp_dt"] = pd.to_datetime(tx_edges["timestamp"], errors="coerce")
max_ts = edges_df["timestamp_dt"].max()
cutoff_48h = max_ts - timedelta(hours=48)

# Target nodes
target_mask = nodes_df["node_type"].isin(["Account", "Merchant"])
target_df = nodes_df[target_mask].copy().set_index("node_id")

print("[*] Computing vectorized features...")

# ============================================================
# 1. DEGREE & AMOUNT FEATURES (vectorized)
# ============================================================
in_degree = tx_edges.groupby("target").size().rename("in_degree")
out_degree = tx_edges.groupby("source").size().rename("out_degree")
total_in_amount = tx_edges.groupby("target")["amount"].sum().rename("total_in_amount")
total_out_amount = tx_edges.groupby("source")["amount"].sum().rename("total_out_amount")

target_df = target_df.join(in_degree).join(out_degree).join(total_in_amount).join(total_out_amount)
target_df[["in_degree", "out_degree", "total_in_amount", "total_out_amount"]] = \
    target_df[["in_degree", "out_degree", "total_in_amount", "total_out_amount"]].fillna(0)

# ============================================================
# 2. FLOW PASS-THROUGH RATIO
# ============================================================
target_df["flow_pass_through_ratio"] = (
    target_df[["total_in_amount", "total_out_amount"]].min(axis=1) /
    (target_df[["total_in_amount", "total_out_amount"]].max(axis=1) + 1)
)

# ============================================================
# 3. STRUCTURING TX RATIO (48k-49.999k band)
# ============================================================
print("[*] Computing structuring ratio...")
all_tx_by_source = tx_edges.groupby("source").agg(
    total_count=("amount", "size"),
    struct_count=("amount", lambda x: ((x >= 48000) & (x <= 49999)).sum())
)
all_tx_by_target = tx_edges.groupby("target").agg(
    total_count=("amount", "size"),
    struct_count=("amount", lambda x: ((x >= 48000) & (x <= 49999)).sum())
)

# Combine source and target perspectives
struct_combined = pd.DataFrame(index=target_df.index)
for df, role in [(all_tx_by_source, "source"), (all_tx_by_target, "target")]:
    df.index.name = "node_id"
    struct_combined = struct_combined.join(df, rsuffix=f"_{role}")

struct_combined = struct_combined.fillna(0)
cols_total = [c for c in struct_combined.columns if "total_count" in c]
cols_struct = [c for c in struct_combined.columns if "struct_count" in c]
total_all = struct_combined[cols_total].sum(axis=1)
struct_all = struct_combined[cols_struct].sum(axis=1)
target_df["structuring_tx_ratio"] = (struct_all / total_all.replace(0, np.nan)).fillna(0)

# ============================================================
# 4. SHARED DEVICE DEGREE
# ============================================================
print("[*] Computing shared device degree...")
login_edges = edges_df[edges_df["edge_type"] == "LOGS_IN_FROM"]
# For each device, count how many accounts log in
device_account_count = login_edges.groupby("target")["source"].nunique().rename("dev_user_count")
# For each account, find max (device_user_count - 1) across devices
login_with_count = login_edges.merge(device_account_count, left_on="target", right_index=True, how="left")
login_with_count["shared_device_degree"] = login_with_count["dev_user_count"] - 1
shared_dev = login_with_count.groupby("source")["shared_device_degree"].max()
shared_dev.index.name = "node_id"
target_df["shared_device_degree"] = target_df.index.map(shared_dev).fillna(0)

# ============================================================
# 5. SHARED IP DEGREE
# ============================================================
print("[*] Computing shared IP degree...")
ip_edges = edges_df[edges_df["edge_type"] == "SHARES_IP"]
ip_account_count = ip_edges.groupby("target")["source"].nunique().rename("ip_user_count")
ip_with_count = ip_edges.merge(ip_account_count, left_on="target", right_index=True, how="left")
ip_with_count["shared_ip_degree"] = ip_with_count["ip_user_count"] - 1
shared_ip = ip_with_count.groupby("source")["shared_ip_degree"].max()
shared_ip.index.name = "node_id"
target_df["shared_ip_degree"] = target_df.index.map(shared_ip).fillna(0)

# ============================================================
# 6. IN-CYCLE FLAG (SCC-based, O(V+E))
# ============================================================
print("[*] Computing cycle membership via SCC...")
cycle_nodes = set()
for scc in nx.strongly_connected_components(G_tx):
    if len(scc) >= 2:
        cycle_nodes.update(scc)
print(f"    Found {len(cycle_nodes)} nodes in directed cycles")
target_df["in_cycle_flag"] = target_df.index.map(lambda x: 1 if x in cycle_nodes else 0)

# ============================================================
# 7. PAGERANK
# ============================================================
print("[*] Computing PageRank...")
pr = nx.pagerank(G_tx, alpha=0.85)
target_df["pagerank_score"] = target_df.index.map(lambda x: pr.get(x, 0.0))

# ============================================================
# 8. LOCAL CLUSTERING COEFFICIENT
# ============================================================
print("[*] Computing clustering coefficients...")
G_undirected = G_tx.to_undirected()
clustering = nx.clustering(G_undirected)
target_df["local_clustering_coef"] = target_df.index.map(lambda x: clustering.get(x, 0.0))

# ============================================================
# 9. VELOCITY SPIKE 48h
# ============================================================
print("[*] Computing velocity spike...")
# All tx amounts per node (as source or target)
tx_as_source = tx_edges[["source", "amount", "timestamp_dt"]].rename(columns={"source": "node_id"})
tx_as_target = tx_edges[["target", "amount", "timestamp_dt"]].rename(columns={"target": "node_id"})
all_node_tx = pd.concat([tx_as_source, tx_as_target])

total_vol = all_node_tx.groupby("node_id")["amount"].sum().rename("total_vol")
recent_tx = all_node_tx[all_node_tx["timestamp_dt"] >= cutoff_48h]
recent_vol = recent_tx.groupby("node_id")["amount"].sum().rename("recent_vol")

vel_df = pd.DataFrame({"total_vol": total_vol, "recent_vol": recent_vol}).fillna(0)
vel_df["velocity_spike_24h"] = vel_df["recent_vol"] / (vel_df["total_vol"] + 1)
target_df["velocity_spike_24h"] = target_df.index.map(vel_df["velocity_spike_24h"]).fillna(0)

# ============================================================
# 10. UNVERIFIED COUNTERPARTY RATIO
# ============================================================
print("[*] Computing unverified counterparty ratio...")
node_kyc = nodes_df.set_index("node_id")[["kyc_verified", "account_age_days"]].to_dict("index")

def is_unverified(nid):
    info = node_kyc.get(nid, {})
    return info.get("kyc_verified", 1) == 0 or info.get("account_age_days", 365) < 30

# Build counterparty sets via tx edges
counterparties = {}
for _, row in tx_edges.iterrows():
    src, tgt = row["source"], row["target"]
    counterparties.setdefault(src, set()).add(tgt)
    counterparties.setdefault(tgt, set()).add(src)

unv_ratio = {}
for nid in target_df.index:
    cps = counterparties.get(nid, set())
    if len(cps) == 0:
        unv_ratio[nid] = 0.0
    else:
        unv_count = sum(1 for cp in cps if is_unverified(cp))
        unv_ratio[nid] = unv_count / len(cps)

target_df["unverified_counterparty_ratio"] = target_df.index.map(unv_ratio).fillna(0)

# ============================================================
# 11 & 12. ACCOUNT AGE & VERIFIED BUSINESS FLAG
# ============================================================
target_df["is_verified_business"] = (
    (target_df["account_subtype"].isin(["Verified_Corporate_Payroll", "Verified_Utility_Merchant"])) &
    (target_df["kyc_verified"] == 1)
).astype(int)

# ============================================================
# FINALIZE AND SAVE
# ============================================================
target_df = target_df.reset_index()

# Round floats
float_cols = ["flow_pass_through_ratio", "structuring_tx_ratio", "pagerank_score",
              "local_clustering_coef", "velocity_spike_24h", "unverified_counterparty_ratio",
              "total_in_amount", "total_out_amount"]
for col in float_cols:
    if col in target_df.columns:
        target_df[col] = target_df[col].round(6)

out_path = os.path.join(DATA_DIR, "Account_Features_FraudCircle.csv")
target_df.to_csv(out_path, index=False)

print(f"\n[OK] Feature table saved: {out_path} ({len(target_df)} rows)")
print(f"\n--- Feature Columns ({len(target_df.columns)}) ---")
print(", ".join(target_df.columns.tolist()))
print(f"\n--- Fraud Label Distribution ---")
print(target_df["ground_truth_fraud"].value_counts().to_string())
print(f"\n--- Sample Features (Fraud cases) ---")
fraud_sample = target_df[target_df["ground_truth_fraud"] == 1].head(10)
print(fraud_sample[["node_id", "scenario_tag", "in_degree", "out_degree",
                     "structuring_tx_ratio", "shared_device_degree",
                     "in_cycle_flag", "velocity_spike_24h"]].to_string())
print("\n[DONE] Feature engineering complete.")
