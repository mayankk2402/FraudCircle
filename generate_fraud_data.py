import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import networkx as nx
from faker import Faker
import random
import datetime

fake = Faker('en_IN')
Faker.seed(42)
np.random.seed(42)
random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

nodes = []
edges = []
edge_counter = [0]
BASE_TS = datetime.datetime(2026, 9, 25, 0, 0, 0)


def ts_offset(hours=0, minutes=0, days=0):
    return (BASE_TS + datetime.timedelta(hours=hours, minutes=minutes, days=days)).isoformat()


def add_node(node_id, name, node_type, account_subtype="", account_age_days=365,
             kyc_verified=1, ground_truth_fraud=0, scenario_tag="NORMAL"):
    nodes.append({
        "node_id": node_id, "name": name, "node_type": node_type,
        "account_subtype": account_subtype, "account_age_days": account_age_days,
        "kyc_verified": kyc_verified, "ground_truth_fraud": ground_truth_fraud,
        "scenario_tag": scenario_tag
    })


def add_edge(source, target, edge_type, amount, timestamp, scenario_tag="NORMAL"):
    edge_counter[0] += 1
    edges.append({
        "edge_id": f"E_{edge_counter[0]:05d}",
        "source": source, "target": target, "edge_type": edge_type,
        "amount": round(amount, 2), "timestamp": timestamp,
        "scenario_tag": scenario_tag
    })


# ============================================================
# FRAUD TYPOLOGY 1: SMURFING / MULE FUNNEL
# ============================================================
print("[*] Generating FRAUD_SMURFING_FUNNEL...")
TAG_SMURF = "FRAUD_SMURFING_FUNNEL"

add_node("ACC_FRAUD_MASTER", "Rajesh Dubey", "Account", "Personal_Savings",
         account_age_days=45, kyc_verified=0, ground_truth_fraud=1, scenario_tag=TAG_SMURF)
add_node("ACC_EXT_CASHOUT", "Offshore Holdings Ltd", "Account", "Personal_Savings",
         account_age_days=10, kyc_verified=0, ground_truth_fraud=1, scenario_tag=TAG_SMURF)

for i in range(45):
    mid = f"ACC_MULE_{i}"
    add_node(mid, fake.name(), "Account", "Personal_Savings",
             account_age_days=random.randint(5, 30), kyc_verified=0,
             ground_truth_fraud=1, scenario_tag=TAG_SMURF)
    amt = round(random.uniform(49100, 49950), 2)
    ts = ts_offset(hours=random.uniform(0, 36))
    add_edge(mid, "ACC_FRAUD_MASTER", "TRANSFERRED_TO", amt, ts, TAG_SMURF)

total_smurf = sum(e["amount"] for e in edges if e["scenario_tag"] == TAG_SMURF)
add_edge("ACC_FRAUD_MASTER", "ACC_EXT_CASHOUT", "TRANSFERRED_TO",
         round(total_smurf * 0.98, 2), ts_offset(hours=37), TAG_SMURF)

# ============================================================
# FRAUD TYPOLOGY 2: SYBIL DEVICE RING
# ============================================================
print("[*] Generating FRAUD_SYBIL_DEVICE_RING...")
TAG_SYBIL = "FRAUD_SYBIL_DEVICE_RING"

add_node("DEV_FRAUD_RING_1", "Device-AA:BB:CC:DD:EE:FF", "Device", "",
         account_age_days=0, kyc_verified=0, ground_truth_fraud=1, scenario_tag=TAG_SYBIL)
add_node("IP_FRAUD_1", "103.45.67.89", "IP_Address", "",
         account_age_days=0, kyc_verified=0, ground_truth_fraud=1, scenario_tag=TAG_SYBIL)
add_node("MERCH_FRAUD_SHELL", "QuickPay Solutions Pvt Ltd", "Merchant", "Shell_Merchant",
         account_age_days=12, kyc_verified=0, ground_truth_fraud=1, scenario_tag=TAG_SYBIL)

for i in range(15):
    aid = f"ACC_SYBIL_{i}"
    add_node(aid, fake.name(), "Account", "Personal_Savings",
             account_age_days=random.randint(60, 400), kyc_verified=1,
             ground_truth_fraud=1, scenario_tag=TAG_SYBIL)
    add_edge(aid, "DEV_FRAUD_RING_1", "LOGS_IN_FROM", 0, ts_offset(hours=random.uniform(0, 48)), TAG_SYBIL)
    add_edge(aid, "IP_FRAUD_1", "SHARES_IP", 0, ts_offset(hours=random.uniform(0, 48)), TAG_SYBIL)
    amt = round(random.uniform(65000, 140000), 2)
    add_edge(aid, "MERCH_FRAUD_SHELL", "PAID_MERCHANT", amt,
             ts_offset(hours=random.uniform(2, 46)), TAG_SYBIL)

# ============================================================
# FRAUD TYPOLOGY 3: CIRCULAR LOOP
# ============================================================
print("[*] Generating FRAUD_CIRCULAR_LOOP...")
TAG_LOOP = "FRAUD_CIRCULAR_LOOP"
loop_nodes = ["ACC_LOOP_A", "ACC_LOOP_B", "ACC_LOOP_C",
              "ACC_LOOP_D", "ACC_LOOP_E", "ACC_LOOP_F"]
for i, lid in enumerate(loop_nodes):
    add_node(lid, fake.name(), "Account", "Personal_Savings",
             account_age_days=random.randint(30, 90), kyc_verified=0,
             ground_truth_fraud=1, scenario_tag=TAG_LOOP)

amt_loop = 95000.0
for cycle in range(3):
    for i in range(6):
        src = loop_nodes[i]
        tgt = loop_nodes[(i + 1) % 6]
        fee = random.uniform(0.01, 0.02)
        amt_loop = round(amt_loop * (1 - fee), 2)
        add_edge(src, tgt, "TRANSFERRED_TO", amt_loop,
                 ts_offset(hours=cycle * 2 + i * 0.3), TAG_LOOP)

# ============================================================
# FRAUD TYPOLOGY 4: BUST-OUT
# ============================================================
print("[*] Generating FRAUD_BUST_OUT...")
TAG_BUST = "FRAUD_BUST_OUT"
add_node("ACC_SLEEPER_1", "Vikram Mehta", "Account", "Personal_Savings",
         account_age_days=400, kyc_verified=1, ground_truth_fraud=1, scenario_tag=TAG_BUST)

# Normal activity for 29 days
for d in range(29):
    tgt_name = f"ACC_NORMAL_RECV_{d}"
    add_node(tgt_name, fake.name(), "Account", "Personal_Savings",
             account_age_days=random.randint(100, 1000), kyc_verified=1,
             ground_truth_fraud=0, scenario_tag="NORMAL")
    add_edge("ACC_SLEEPER_1", tgt_name, "TRANSFERRED_TO",
             round(random.uniform(500, 3000), 2), ts_offset(days=d), TAG_BUST)

# Burst in last 24 hours
burst_accounts = []
for i in range(8):
    bid = f"ACC_BUST_DRAIN_{i}"
    add_node(bid, fake.name(), "Account", "Personal_Savings",
             account_age_days=random.randint(2, 7), kyc_verified=0,
             ground_truth_fraud=1, scenario_tag=TAG_BUST)
    burst_accounts.append(bid)
    amt = round(random.uniform(150000, 250000), 2)
    add_edge("ACC_SLEEPER_1", bid, "TRANSFERRED_TO", amt,
             ts_offset(days=29, hours=random.uniform(0, 23)), TAG_BUST)

# ============================================================
# LEGIT 1: CORPORATE PAYROLL
# ============================================================
print("[*] Generating LEGIT_CORP_PAYROLL...")
TAG_PAYROLL = "LEGIT_CORP_PAYROLL"
add_node("ACC_CORP_INFOSYS_PAYROLL", "Infosys BPO Ltd - Payroll", "Account",
         "Verified_Corporate_Payroll", account_age_days=1800, kyc_verified=1,
         ground_truth_fraud=0, scenario_tag=TAG_PAYROLL)

for i in range(50):
    eid = f"ACC_EMP_{i}"
    dev_id = f"DEV_EMP_{i}"
    add_node(eid, fake.name(), "Account", "Personal_Savings",
             account_age_days=random.randint(200, 2000), kyc_verified=1,
             ground_truth_fraud=0, scenario_tag=TAG_PAYROLL)
    add_node(dev_id, f"Device-EMP-{fake.hexify('^^:^^:^^:^^:^^:^^')}", "Device", "",
             account_age_days=0, kyc_verified=0, ground_truth_fraud=0, scenario_tag=TAG_PAYROLL)
    add_edge(eid, dev_id, "LOGS_IN_FROM", 0, ts_offset(days=1), TAG_PAYROLL)
    salary = round(random.uniform(35000, 85000), 2)
    add_edge("ACC_CORP_INFOSYS_PAYROLL", eid, "TRANSFERRED_TO", salary,
             ts_offset(days=1, hours=random.uniform(9, 11)), TAG_PAYROLL)

# ============================================================
# LEGIT 2: UTILITY HUB
# ============================================================
print("[*] Generating LEGIT_UTILITY_HUB...")
TAG_UTILITY = "LEGIT_UTILITY_HUB"
add_node("MERCH_TATA_POWER", "Tata Power Mumbai", "Merchant",
         "Verified_Utility_Merchant", account_age_days=2500, kyc_verified=1,
         ground_truth_fraud=0, scenario_tag=TAG_UTILITY)

for i in range(120):
    uid = f"ACC_UTIL_CUST_{i}"
    add_node(uid, fake.name(), "Account", "Personal_Savings",
             account_age_days=random.randint(100, 3000), kyc_verified=1,
             ground_truth_fraud=0, scenario_tag=TAG_UTILITY)
    amt = round(random.uniform(800, 4500), 2)
    add_edge(uid, "MERCH_TATA_POWER", "PAID_MERCHANT", amt,
             ts_offset(days=random.randint(0, 28)), TAG_UTILITY)

# ============================================================
# LEGIT 3: WEDDING SPEND
# ============================================================
print("[*] Generating LEGIT_WEDDING_SPEND...")
TAG_WEDDING = "LEGIT_WEDDING_SPEND"
add_node("ACC_WEDDING_FAMILY", "Sharma Family Trust", "Account", "Personal_Savings",
         account_age_days=1200, kyc_verified=1, ground_truth_fraud=0, scenario_tag=TAG_WEDDING)
add_node("DEV_WEDDING_1", "Device-WEDDING-F1:A2:B3:C4:D5:E6", "Device", "",
         account_age_days=0, kyc_verified=0, ground_truth_fraud=0, scenario_tag=TAG_WEDDING)
add_edge("ACC_WEDDING_FAMILY", "DEV_WEDDING_1", "LOGS_IN_FROM", 0, ts_offset(days=24), TAG_WEDDING)

wedding_merchants = [
    ("MERCH_JEWELER_TANISHQ", "Tanishq Jewellers", 250000, 350000),
    ("MERCH_HOTEL_TAJ", "Taj Hotels Bangalore", 180000, 220000),
    ("MERCH_CATERING_HALDIRAM", "Haldiram Catering", 80000, 120000),
    ("MERCH_AIRLINE_INDIGO", "IndiGo Airlines", 40000, 75000),
    ("MERCH_DECOR_FERNS", "Ferns N Petals Decor", 50000, 90000),
    ("MERCH_PHOTO_STUDIO", "Wedding Clicks Studio", 35000, 55000),
    ("MERCH_CLOTHING_MANYAVAR", "Manyavar Clothing", 30000, 50000),
    ("MERCH_VENUE_ITC", "ITC Grand Chola Venue", 100000, 150000),
]
for mid, mname, lo, hi in wedding_merchants:
    add_node(mid, mname, "Merchant", "Verified_Utility_Merchant",
             account_age_days=random.randint(800, 3000), kyc_verified=1,
             ground_truth_fraud=0, scenario_tag=TAG_WEDDING)
    amt = round(random.uniform(lo, hi), 2)
    add_edge("ACC_WEDDING_FAMILY", mid, "PAID_MERCHANT", amt,
             ts_offset(days=random.randint(24, 29), hours=random.uniform(8, 20)), TAG_WEDDING)

# ============================================================
# NORMAL BASELINE NETWORK (Barabasi-Albert)
# ============================================================
print("[*] Generating Normal Baseline Network (800 accounts, BA graph)...")
TAG_NORMAL = "NORMAL"

ba_graph = nx.barabasi_albert_graph(800, 2, seed=42)

normal_accounts = []
for i in range(800):
    nid = f"ACC_NORM_{i}"
    add_node(nid, fake.name(), "Account", "Personal_Savings",
             account_age_days=random.randint(30, 2500), kyc_verified=1,
             ground_truth_fraud=0, scenario_tag=TAG_NORMAL)
    normal_accounts.append(nid)

normal_merchants = []
for i in range(40):
    mid = f"MERCH_NORM_{i}"
    add_node(mid, fake.company(), "Merchant", "Verified_Utility_Merchant",
             account_age_days=random.randint(365, 3000), kyc_verified=1,
             ground_truth_fraud=0, scenario_tag=TAG_NORMAL)
    normal_merchants.append(mid)

normal_devices = []
for i in range(750):
    did = f"DEV_NORM_{i}"
    add_node(did, f"Device-{fake.hexify('^^:^^:^^:^^:^^:^^')}", "Device", "",
             account_age_days=0, kyc_verified=0, ground_truth_fraud=0, scenario_tag=TAG_NORMAL)
    normal_devices.append(did)

# Assign devices to accounts (1:1 mostly)
for i in range(min(750, 800)):
    add_edge(normal_accounts[i], normal_devices[i % 750], "LOGS_IN_FROM", 0,
             ts_offset(days=random.randint(0, 29)), TAG_NORMAL)

# BA graph edges -> transactions
for u, v in ba_graph.edges():
    src = normal_accounts[u]
    tgt = normal_accounts[v]
    amt = round(random.uniform(200, 25000), 2)
    add_edge(src, tgt, "TRANSFERRED_TO", amt,
             ts_offset(days=random.randint(0, 29), hours=random.uniform(0, 23)), TAG_NORMAL)

# Normal merchant payments
for _ in range(500):
    acc = random.choice(normal_accounts)
    merch = random.choice(normal_merchants)
    amt = round(random.uniform(100, 15000), 2)
    add_edge(acc, merch, "PAID_MERCHANT", amt,
             ts_offset(days=random.randint(0, 29)), TAG_NORMAL)

# ============================================================
# SAVE CSVs
# ============================================================
nodes_df = pd.DataFrame(nodes)
edges_df = pd.DataFrame(edges)

nodes_path = os.path.join(DATA_DIR, "Nodes_FraudCircle.csv")
edges_path = os.path.join(DATA_DIR, "Edges_FraudCircle.csv")

nodes_df.to_csv(nodes_path, index=False)
edges_df.to_csv(edges_path, index=False)

print(f"\n[OK] Nodes saved: {nodes_path} ({len(nodes_df)} rows)")
print(f"[OK] Edges saved: {edges_path} ({len(edges_df)} rows)")
print(f"\n--- Scenario Distribution (Nodes) ---")
print(nodes_df["scenario_tag"].value_counts().to_string())
print(f"\n--- Scenario Distribution (Edges) ---")
print(edges_df["scenario_tag"].value_counts().to_string())
print(f"\n--- Node Types ---")
print(nodes_df["node_type"].value_counts().to_string())
print(f"\n--- Fraud Ground Truth ---")
print(nodes_df["ground_truth_fraud"].value_counts().to_string())
print("\n[DONE] Data generation complete.")
