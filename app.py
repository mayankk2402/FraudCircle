import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import pandas as pd
import numpy as np
from datetime import datetime

import dash
from dash import html, dcc, Input, Output, State, callback, no_update, ctx
import dash_cytoscape as cyto
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from neo4j_sync import generate_cypher_for_scenario, generate_match_query_for_scenario, sync_to_neo4j

# ============================================================
# LOAD REAL DATASETS & MODEL ARTIFACTS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

nodes_df = pd.read_csv(os.path.join(DATA_DIR, "Nodes_FraudCircle.csv"))
edges_df = pd.read_csv(os.path.join(DATA_DIR, "Edges_FraudCircle.csv"))
feat_df = pd.read_csv(os.path.join(DATA_DIR, "Account_Features_FraudCircle.csv"))
feat_map = feat_df.set_index("node_id").to_dict(orient="index")

with open(os.path.join(DATA_DIR, "Model_Predictions_XAI.json"), "r", encoding="utf-8") as f:
    predictions = json.load(f)

pred_map = {p["node_id"]: p for p in predictions}

# ============================================================
# CASE DEFINITIONS & GUIDED WALKTHROUGH CONTENT
# ============================================================
SCENARIO_KEYS = [
    "FRAUD_SMURFING_FUNNEL",
    "FRAUD_SYBIL_DEVICE_RING",
    "FRAUD_CIRCULAR_LOOP",
    "FRAUD_BUST_OUT",
    "LEGIT_CORP_PAYROLL",
    "LEGIT_UTILITY_HUB",
    "LEGIT_WEDDING_SPEND",
]

SCENARIOS = {
    "FRAUD_SMURFING_FUNNEL": {
        "index_num": 1,
        "title": "Smurfing / Mule Funnel (PAN Structuring)",
        "badge": "Mule Funnel",
        "category": "FRAUD",
        "default_layout": "concentric",
        "focal_node": "ACC_FRAUD_MASTER",
        "exposure_inr": "₹ 22,26,841",
        "exposure_short": "₹ 22.3L",
        "entities_count": "47 entities",
        "risk_pct": 99.9,
        "severity": "Critical Risk",
        "one_liner": "45 mule accounts funnel structured transfers (< ₹50k) to a single master account within 36 hours, immediately wired offshore.",
        "scheme_details": "45 distinct mule accounts (ACC_MULE_0 to ACC_MULE_44) transfer amounts between ₹49,100 and ₹49,950 into ACC_FRAUD_MASTER within a 36-hour window. The master account immediately wires the aggregated ₹22.3 Lakhs out to offshore account ACC_EXT_CASHOUT.",
        "why_rules_fail": "Traditional AML rules inspect transactions in isolation. Because each transfer sits deliberately beneath the Indian ₹50,000 PAN card mandatory reporting threshold, static threshold alerts trigger zero warnings. Network analysis reveals the 45:1 in-degree funnel, 97.8% structuring concentration, and 0.98 flow pass-through.",
        "key_signals": [
            ("Structuring Ratio", "97.8% (< ₹50,000 PAN threshold)"),
            ("Flow Pass-Through", "0.98 (min/max inflow-outflow balance)"),
            ("Degree Imbalance", "45 Inflow : 1 Outflow"),
            ("Model Risk Score", "99.9 / 100"),
        ],
        "walkthrough": [
            {
                "step": 1,
                "title": "What Happened?",
                "concept": "Smurfing & Mule Funneling",
                "explanation": "This case represents a classic micro-structuring mule funnel. Criminal networks recruit or compromise individuals ('mules') to open bank accounts. Here, 45 mule accounts sent structured payments under ₹50,000 into a central aggregator ('funnel account'), which aggregated the funds and transferred them out to an offshore account.",
                "highlight_nodes": ["ACC_FRAUD_MASTER", "ACC_EXT_CASHOUT", "ACC_MULE_0", "ACC_MULE_1", "ACC_MULE_2"],
                "evidence": "45 source accounts, 1 master destination, ₹22.3 Lakhs total transferred in 36 hours."
            },
            {
                "step": 2,
                "title": "Follow the Money",
                "concept": "Fund Origin & Cash-Out Path",
                "explanation": "Observing the network graph: 45 independent personal savings accounts initiate outward transfers into Rajesh Dubey (ACC_FRAUD_MASTER). Within hours of accumulation, Rajesh Dubey initiates a single outbound wire of ₹21,82,304 to Offshore Holdings Ltd (ACC_EXT_CASHOUT).",
                "highlight_nodes": ["ACC_FRAUD_MASTER", "ACC_EXT_CASHOUT"],
                "evidence": "Flow pass-through ratio is 0.98, confirming the account operates as an unverified liquidity transit conduit rather than a savings account."
            },
            {
                "step": 3,
                "title": "What Makes This Suspicious?",
                "concept": "Observed Facts vs. Fraud Hypothesis",
                "explanation": "Fact 1: 45 unrelated accounts with different names send money to one recipient within 36 hours. Fact 2: Every single transaction is between ₹49,100 and ₹49,950. Fact 3: The master account has no prior salary or business history. While a wedding gift or fundraiser could have multiple senders, the uniform pricing just below ₹50k indicates intentional evasion.",
                "highlight_nodes": ["ACC_FRAUD_MASTER"],
                "evidence": "44 of 45 incoming transactions fall in the ₹48k-₹49.95k range (97.8% structuring band ratio)."
            },
            {
                "step": 4,
                "title": "Why Did Traditional Rules Miss It?",
                "concept": "Threshold Evasion & Dyadic Blindness",
                "explanation": "Under Indian banking guidelines (Section 139A Income Tax Act), transactions of ₹50,000 and above require mandatory PAN verification. By keeping every transfer between ₹49,100 and ₹49,950, each individual transaction cleared the bank's core banking threshold rule with zero flags. Rule-based systems evaluate (Sender, Receiver) in isolation; only Graph ML evaluates the entire 45:1 fan-in topology.",
                "highlight_nodes": ["ACC_MULE_0", "ACC_MULE_1", "ACC_MULE_2", "ACC_MULE_3"],
                "evidence": "Max single transaction: ₹49,950. Static rule threshold: ₹50,000. Detection delta: 0 alarms raised by legacy AML."
            },
            {
                "step": 5,
                "title": "How Did the Model Reach Its Score?",
                "concept": "Topological Features & SHAP Attribution",
                "explanation": "The XGBoost model assigned Rajesh Dubey a Risk Score of 99.9/100. TreeExplainer SHAP attribution reveals the exact drivers: structuring_tx_ratio (+42.1% impact), in_degree=45 (+31.8%), flow_pass_through_ratio=0.98 (+18.4%), and unverified_counterparty_ratio (+7.2%). Mitigating factors like account history were near zero.",
                "highlight_nodes": ["ACC_FRAUD_MASTER"],
                "evidence": "SHAP sum: +99.9% probability increase over baseline."
            },
            {
                "step": 6,
                "title": "What Should an Investigator Do Next?",
                "concept": "Recommended Statutory Actions",
                "explanation": "Actionable next steps: 1) Apply administrative debit freeze on ACC_FRAUD_MASTER under Section 35A of Banking Regulation Act. 2) File Form STR under Section 12 PMLA with FIU-IND. 3) Request KYC and IP metadata for all 45 mule accounts from respective originating banks to identify common syndicate handlers.",
                "highlight_nodes": ["ACC_FRAUD_MASTER", "ACC_EXT_CASHOUT"],
                "evidence": "Elevator Pitch Summary: 45 mules funneled ₹22.3L in ₹49k increments to evade PAN reporting, rapidly wired offshore with 0.98 pass-through. Graph AI detected the ring instantly."
            }
        ]
    },
    "FRAUD_SYBIL_DEVICE_RING": {
        "index_num": 2,
        "title": "Sybil Ring (Device & IP Triangulation)",
        "badge": "Sybil Ring",
        "category": "FRAUD",
        "default_layout": "concentric",
        "focal_node": "ACC_SYBIL_0",
        "exposure_inr": "₹ 15,40,000",
        "exposure_short": "₹ 15.4L",
        "entities_count": "18 entities",
        "risk_pct": 99.8,
        "severity": "Critical Risk",
        "one_liner": "15 synthetic personal accounts with no direct transfers share the exact same physical device MAC address and IP, funneling payments to a shell merchant.",
        "scheme_details": "15 distinct personal accounts with different names have no direct money transfers between one another. However, bipartite projection reveals all 15 share DEV_FRAUD_RING_1 and IP_FRAUD_1, funneling payments into newly registered unverified merchant MERCH_FRAUD_SHELL.",
        "why_rules_fail": "Because accounts do not transact with each other, legacy transaction monitoring flags zero circular paths or reciprocal transfers. Heterogeneous graph entity resolution exposes the physical device hub linking all 15 synthetic identities.",
        "key_signals": [
            ("Shared Device Logins", "14 other accounts on same MAC"),
            ("Shared IP Logins", "14 accounts on same IP"),
            ("Merchant Counterparty", "Unverified Shell Merchant"),
            ("Model Risk Score", "99.8 / 100"),
        ],
        "walkthrough": [
            {
                "step": 1,
                "title": "What Happened?",
                "concept": "Sybil Identity Multiplication",
                "explanation": "In a Sybil attack, a single criminal operator generates multiple synthetic identity personas to bypass single-account volume limits and credit thresholds. Here, 15 personal accounts were created with different fabricated names.",
                "highlight_nodes": ["DEV_FRAUD_RING_1", "IP_FRAUD_1", "MERCH_FRAUD_SHELL"],
                "evidence": "15 personal accounts, 1 device MAC, 1 IP address, 1 shell merchant destination."
            },
            {
                "step": 2,
                "title": "Follow the Money",
                "concept": "Multi-Entity Convergence",
                "explanation": "Notice that the 15 accounts never send a single rupee to each other. Instead, each account transfers between ₹65,000 and ₹1,40,000 directly into QuickPay Solutions (MERCH_FRAUD_SHELL). Look at the graph: the accounts appear disconnected in a transaction-only graph.",
                "highlight_nodes": ["ACC_SYBIL_0", "MERCH_FRAUD_SHELL", "DEV_FRAUD_RING_1"],
                "evidence": "Total convergent volume: ₹15.4 Lakhs into a 12-day-old unverified merchant entity."
            },
            {
                "step": 3,
                "title": "What Makes This Suspicious?",
                "concept": "Hardware & Network Triangulation",
                "explanation": "When we project the heterogeneous bipartite edges (LOGS_IN_FROM and SHARES_IP), all 15 independent users connect to the exact same device hardware ID (DEV_FRAUD_RING_1) and IP address (IP_FRAUD_1). It is statistically improbable for 15 unrelated adults across India to share one device hardware fingerprint.",
                "highlight_nodes": ["DEV_FRAUD_RING_1", "IP_FRAUD_1"],
                "evidence": "Shared device degree = 14 (14 other accounts co-login from same hardware MAC)."
            },
            {
                "step": 4,
                "title": "Why Did Traditional Rules Miss It?",
                "concept": "The Siloed Identity Flaw",
                "explanation": "Traditional AML monitors account ledgers. Since Account A doesn't know Account B, each account looks like an ordinary consumer making an e-commerce purchase of ₹80,000. Traditional tools do not join network telemetry with banking transactions.",
                "highlight_nodes": ["ACC_SYBIL_0", "ACC_SYBIL_1", "DEV_FRAUD_RING_1"],
                "evidence": "0 direct transfers between accounts. Transaction rules evaluate 15 independent benign events."
            },
            {
                "step": 5,
                "title": "How Did the Model Reach Its Score?",
                "concept": "Feature Attribution & Evidence",
                "explanation": "The model flagged ACC_SYBIL_0 with 99.8/100 risk. The top SHAP features are shared_device_degree (+44.2%), unverified_counterparty_ratio (+31.5%), and shared_ip_degree (+18.1%).",
                "highlight_nodes": ["ACC_SYBIL_0"],
                "evidence": "SHAP confirms that physical infrastructure clustering drove 93.8% of the model's suspicion."
            },
            {
                "step": 6,
                "title": "What Should an Investigator Do Next?",
                "concept": "Enforcement & Hardware Blacklisting",
                "explanation": "1) Blacklist hardware MAC DEV_FRAUD_RING_1 across mobile banking SDKs. 2) Freeze merchant account MERCH_FRAUD_SHELL. 3) Request device telemetry and SIM binding history from Telecom Provider.",
                "highlight_nodes": ["DEV_FRAUD_RING_1", "MERCH_FRAUD_SHELL"],
                "evidence": "Elevator Pitch Summary: 15 synthetic accounts had zero direct transfers, but graph telemetry exposed all 15 operating from 1 physical device funneling ₹15.4L to an unverified shell merchant."
            }
        ]
    },
    "FRAUD_CIRCULAR_LOOP": {
        "index_num": 3,
        "title": "Circular Money Laundering (Round-Tripping)",
        "badge": "Circular Loop",
        "category": "FRAUD",
        "default_layout": "circle",
        "focal_node": "ACC_LOOP_A",
        "exposure_inr": "₹ 5,70,000",
        "exposure_short": "₹ 5.7L",
        "entities_count": "6 entities",
        "risk_pct": 99.9,
        "severity": "Critical Risk",
        "one_liner": "A 6-node closed directed loop rapidly cycles ~₹95k minus fees within 2 hours to artificially inflate balance sheet turnover for fraudulent credit lines.",
        "scheme_details": "A closed directed cycle (ACC_LOOP_A -> ACC_LOOP_B -> ACC_LOOP_C -> ACC_LOOP_D -> ACC_LOOP_E -> ACC_LOOP_F -> ACC_LOOP_A) rapidly cycles ~₹95,000 (minus 1-2% fee per hop) within 2 hours. Funds return to originator to create artificial balance sheet turnover for loan fraud.",
        "why_rules_fail": "Traditional AML checks only (Sender, Receiver) dyads. In this multi-hop ring, no entity receives money back directly from the party they paid. Topological Directed Cycle Detection catches the closed loop.",
        "key_signals": [
            ("Directed Cycle Length", "6 Hops (Closed Loop)"),
            ("In-Cycle Indicator", "Yes (Flag = 1)"),
            ("Flow Pass-Through", "0.99"),
            ("Model Risk Score", "99.9 / 100"),
        ],
        "walkthrough": [
            {
                "step": 1,
                "title": "What Happened?",
                "concept": "Round-Tripping / Layering Cycle",
                "explanation": "Round-tripping involves passing the same pool of funds through multiple intermediary accounts so it returns to the originator. The goal is to fabricate transaction velocity, inflate declared turnover, and qualify for high-limit corporate credit lines or trade finance loans.",
                "highlight_nodes": ["ACC_LOOP_A", "ACC_LOOP_B", "ACC_LOOP_C", "ACC_LOOP_D", "ACC_LOOP_E", "ACC_LOOP_F"],
                "evidence": "6 accounts arranged in a closed directed polygon cycling ₹95,000."
            },
            {
                "step": 2,
                "title": "Follow the Money",
                "concept": "Closed Path Tracing",
                "explanation": "Tracing the flow: A sends ₹95,000 to B -> B sends ₹93,500 to C -> C sends ₹92,100 to D -> D sends ₹90,700 to E -> E sends ₹89,400 to F -> F sends ₹88,100 back to A! The entire cycle completes in less than 2 hours.",
                "highlight_nodes": ["ACC_LOOP_A", "ACC_LOOP_B", "ACC_LOOP_F"],
                "evidence": "Small 1-2% deductions at each hop represent transaction fees or mule cuts."
            },
            {
                "step": 3,
                "title": "What Makes This Suspicious?",
                "concept": "Velocity & Non-Economic Flow",
                "explanation": "In genuine commerce, money flows from buyer to supplier to manufacturer in open chains. Money returning to the original sender within hours minus only transaction fees has zero genuine economic rationale.",
                "highlight_nodes": ["ACC_LOOP_A"],
                "evidence": "Flow pass-through ratio is 0.99. Inflow equals outflow almost perfectly."
            },
            {
                "step": 4,
                "title": "Why Did Traditional Rules Miss It?",
                "concept": "The Dyadic Limitation",
                "explanation": "Traditional AML rules inspect pairs: (A paid B) is legal; (B paid C) is legal. Because A never transacts directly with E or F, dyadic relational databases see 6 separate valid commercial payments. Graph cycle algorithms ($C_k \\le 8$) are mathematically necessary to detect closed walks.",
                "highlight_nodes": ["ACC_LOOP_A", "ACC_LOOP_C", "ACC_LOOP_E"],
                "evidence": "Zero two-way transactions. No sender-receiver pair transacted in reverse."
            },
            {
                "step": 5,
                "title": "How Did the Model Reach Its Score?",
                "concept": "Topological Cycle Features",
                "explanation": "The XGBoost model flags all 6 accounts with 99.9/100 risk. The dominant SHAP feature is in_cycle_flag=1 (+51.2% impact) and flow_pass_through_ratio=0.99 (+34.6%).",
                "highlight_nodes": ["ACC_LOOP_A"],
                "evidence": "Cycle detection feature directly isolates circular round-tripping topology."
            },
            {
                "step": 6,
                "title": "What Should an Investigator Do Next?",
                "concept": "Credit Freeze & Collusion Audit",
                "explanation": "1) Freeze pending credit line and loan disbursements across all 6 entities. 2) Cross-reference beneficial ownership registries to check for shared directors. 3) File joint FIU-IND STR citing circular layering under PMLA Section 12.",
                "highlight_nodes": ["ACC_LOOP_A", "ACC_LOOP_D"],
                "evidence": "Elevator Pitch Summary: 6 accounts rapidly cycled ₹95,000 in a closed directed loop within 2 hours. Traditional dyadic rules saw 6 unrelated transfers, but graph cycle detection flagged the fraud ring with 99.9% risk."
            }
        ]
    },
    "FRAUD_BUST_OUT": {
        "index_num": 4,
        "title": "Sleeper Bust-Out Anomaly (Velocity Surge)",
        "badge": "Bust-Out",
        "category": "FRAUD",
        "default_layout": "concentric",
        "focal_node": "ACC_SLEEPER_1",
        "exposure_inr": "₹ 15,00,000",
        "exposure_short": "₹ 15.0L",
        "entities_count": "9 entities",
        "risk_pct": 99.7,
        "severity": "Critical Risk",
        "one_liner": "An account shows 29 days of dormant ₹1k grocery transfers, then explodes with a 10,000% velocity spike in 24 hours draining ₹15L into brand-new accounts.",
        "scheme_details": "Account ACC_SLEEPER_1 shows normal ₹1,000 grocery payments for 29 days, then in the final 24 hours maxes out ₹15 Lakhs across 8 newly created (<7 days old) accounts that immediately drain the funds.",
        "why_rules_fail": "Historical account age (400 days) and static KYC profiling classify this account as low risk. Graph velocity surge tracking combined with unverified counterparty ratio (100% new unverified accounts) catches the bust-out takeover in real time.",
        "key_signals": [
            ("48h Velocity Surge", "98.2% of 30d volume in 24h"),
            ("Counterparty Age", "100% Unverified (<7d Old)"),
            ("Velocity Multiplier", "10,000% surge"),
            ("Model Risk Score", "99.7 / 100"),
        ],
        "walkthrough": [
            {
                "step": 1,
                "title": "What Happened?",
                "concept": "Sleeper Account Bust-Out",
                "explanation": "Bust-out fraud occurs when an account is established or acquired, maintained with good behavior for months to build credit standing, and then suddenly maxed out to its limit before being abandoned.",
                "highlight_nodes": ["ACC_SLEEPER_1"],
                "evidence": "Account age: 400 days. 29 days of small transactions, 1 day of massive drain."
            },
            {
                "step": 2,
                "title": "Follow the Money",
                "concept": "Sudden Fan-Out Drain",
                "explanation": "For 29 days, Vikram Mehta (ACC_SLEEPER_1) sends ₹500 to ₹3,000 to utility and grocery merchants. On Day 30, the account initiates 8 sudden transfers of ₹1,50,000 to ₹2,50,000 each to newly opened accounts.",
                "highlight_nodes": ["ACC_SLEEPER_1", "ACC_BUST_DRAIN_0", "ACC_BUST_DRAIN_1"],
                "evidence": "Total volume in 24 hours: ₹15,00,000 (98.2% of total 30-day activity)."
            },
            {
                "step": 3,
                "title": "What Makes This Suspicious?",
                "concept": "Temporal Anomaly & Counterparty Risk",
                "explanation": "A 10,000% increase in daily transaction volume coincides with transferring funds to counterparties that are less than 7 days old with unverified KYC credentials.",
                "highlight_nodes": ["ACC_BUST_DRAIN_0", "ACC_BUST_DRAIN_1"],
                "evidence": "100% of receiving counterparties are newly created unverified accounts."
            },
            {
                "step": 4,
                "title": "Why Did Traditional Rules Miss It?",
                "concept": "Static Whitelisting Flaw",
                "explanation": "Because ACC_SLEEPER_1 is 400 days old with fully verified KYC, static profiling tools treat it as a trusted prime account. Traditional systems update risk tiers monthly rather than computing streaming graph velocity.",
                "highlight_nodes": ["ACC_SLEEPER_1"],
                "evidence": "Legacy trust score: 'Low Risk Tier 1' due to 400-day account tenure."
            },
            {
                "step": 5,
                "title": "How Did the Model Reach Its Score?",
                "concept": "Dynamic Velocity Graph Features",
                "explanation": "The model flagged the account with 99.7/100 risk. Key SHAP drivers: velocity_spike_24h (+48.6%), unverified_counterparty_ratio (+36.2%), and out_degree (+12.4%).",
                "highlight_nodes": ["ACC_SLEEPER_1"],
                "evidence": "Dynamic velocity features override static account age."
            },
            {
                "step": 6,
                "title": "What Should an Investigator Do Next?",
                "concept": "Immediate Inter-Bank Recall",
                "explanation": "1) Trigger UPI / NEFT inter-bank recall on the 8 destination drain accounts. 2) Place a hold on Vikram Mehta's account. 3) Notify victim of potential account takeover or credential theft.",
                "highlight_nodes": ["ACC_SLEEPER_1"],
                "evidence": "Elevator Pitch Summary: A 400-day verified account showed normal activity for 29 days, then suffered a 10,000% velocity surge draining ₹15L to 8 unverified new accounts. Graph temporal features caught the bust-out immediately."
            }
        ]
    },
    "LEGIT_CORP_PAYROLL": {
        "index_num": 5,
        "title": "Corporate Payroll Hub (Legitimate)",
        "badge": "Payroll Hub",
        "category": "LEGIT",
        "default_layout": "concentric",
        "focal_node": "ACC_CORP_INFOSYS_PAYROLL",
        "exposure_inr": "₹ 0.0 (Cleared)",
        "exposure_short": "₹ 0.0",
        "entities_count": "101 entities",
        "risk_pct": 0.2,
        "severity": "Cleared",
        "one_liner": "Infosys BPO Ltd sends salaries (₹35k–₹85k) to 50 verified employee accounts on the 1st of the month. High out-degree is cleared as legitimate corporate operations.",
        "scheme_details": "Infosys BPO Ltd sends salaries (₹35k–₹85k) to 50 verified employee accounts on the 1st of the month. Despite high out-degree (50:0), it is a verified corporate entity operating for 1,800 days with distinct devices.",
        "why_rules_fail": "Naive high-outflow anomaly detectors flag this as a smurfing fan-out. FraudCircle verifies corporate KYC credentials, distinct employee devices, and absence of returning fund cycles to suppress false positives.",
        "key_signals": [
            ("Entity Age", "1,800 days verified corporate"),
            ("Verified Business", "Yes (KYC = 1)"),
            ("Return Flow", "Zero return cycles (Acyclic)"),
            ("Model Risk Score", "0.2 / 100 (Cleared)"),
        ],
        "walkthrough": [
            {
                "step": 1,
                "title": "What Happened?",
                "concept": "Legitimate Corporate Salary Disbursement",
                "explanation": "On the 1st of the month, Infosys BPO Ltd disburses monthly salaries ranging between ₹35,000 and ₹85,000 across 50 employee accounts. This creates a massive 50:0 out-degree star topology.",
                "highlight_nodes": ["ACC_CORP_INFOSYS_PAYROLL"],
                "evidence": "50 outbound salary wires in a single morning."
            },
            {
                "step": 2,
                "title": "Why Could This Look Like Fraud?",
                "concept": "The False-Positive Trap",
                "explanation": "If an algorithm looks purely at network centrality or out-degree fan-out, Infosys Payroll looks identical to a mule recruiter fanning out dirty money to 50 runners.",
                "highlight_nodes": ["ACC_CORP_INFOSYS_PAYROLL", "ACC_EMP_0", "ACC_EMP_1"],
                "evidence": "High out-degree (50 transfers). Outflow volume: ~₹30 Lakhs."
            },
            {
                "step": 3,
                "title": "How Does Graph AI Prove It's Legitimate?",
                "concept": "Multi-Signal Context Verification",
                "explanation": "FraudCircle checks 3 critical graph invariants: 1) Counterparties: all 50 employees log in from unique personal mobile devices (shared_device_degree = 0). 2) Topology: zero money ever loops back to the corporation. 3) Entity context: the sender is a 1,800-day-old KYC-verified corporate payroll entity.",
                "highlight_nodes": ["ACC_CORP_INFOSYS_PAYROLL", "DEV_EMP_0"],
                "evidence": "Unique devices per employee, acyclic directed graph, verified corporate KYC."
            },
            {
                "step": 4,
                "title": "Model Interpretation",
                "concept": "Risk Suppression via Negative SHAP",
                "explanation": "The XGBoost model assesses a Risk Score of only 0.2/100 (CLEARED). Strong negative SHAP features actively suppressed suspicion: is_verified_business=1 (-45.2% risk), account_age_days=1800 (-32.1%), and in_cycle_flag=0 (-18.4%).",
                "highlight_nodes": ["ACC_CORP_INFOSYS_PAYROLL"],
                "evidence": "Model actively clears legitimate business activity with zero alert fatigue."
            },
            {
                "step": 5,
                "title": "Comparison with Mule Funnel",
                "concept": "Benchmark Differences",
                "explanation": "Compare with Case #1: In Case #1, accounts share identical structured amounts (<₹50k), lack verified corporate status, and loop money offshore. In Case #5, amounts reflect varied salaries, the sender is a 5-year corporate taxpayer, and funds disperse into the economy.",
                "highlight_nodes": ["ACC_CORP_INFOSYS_PAYROLL"],
                "evidence": "Zero false positive alert generated."
            },
            {
                "step": 6,
                "title": "Investigator Takeaway",
                "concept": "Trust and Operational Efficiency",
                "explanation": "By recognizing verified corporate sub-types and acyclic device networks, FraudCircle eliminates false positives that plague conventional banks, saving hundreds of hours of manual compliance review.",
                "highlight_nodes": ["ACC_CORP_INFOSYS_PAYROLL"],
                "evidence": "Elevator Pitch Summary: Infosys Payroll sent 50 salaries in one morning. Rather than triggering a false smurfing alert, Graph AI verified unique employee devices, corporate KYC, and acyclic flow, clearing it with a 0.2/100 risk score."
            }
        ]
    },
    "LEGIT_UTILITY_HUB": {
        "index_num": 6,
        "title": "Utility Bill Merchant Hub (Legitimate)",
        "badge": "Utility Hub",
        "category": "LEGIT",
        "default_layout": "concentric",
        "focal_node": "MERCH_TATA_POWER",
        "exposure_inr": "₹ 0.0 (Cleared)",
        "exposure_short": "₹ 0.0",
        "entities_count": "121 entities",
        "risk_pct": 0.1,
        "severity": "Cleared",
        "one_liner": "Tata Power Mumbai receives 120 retail electricity bill payments (₹800–₹4,500). High in-degree is cleared as legitimate utility bill aggregation.",
        "scheme_details": "Tata Power receives 120 payments between ₹800 and ₹4,500. It has high in-degree (120:0), but transactions are well below structuring bands, devices are distinct, and the entity is a 7-year verified utility merchant.",
        "why_rules_fail": "Traditional in-degree volume thresholds flag utility merchants as mule aggregators. FraudCircle checks the ₹48k–₹50k structuring band ratio (0.00%) and verified utility merchant credentials to prevent false alarms.",
        "key_signals": [
            ("Structuring Ratio", "0.00% (Normal Bill Amounts)"),
            ("Average Ticket Size", "₹ 2,450"),
            ("Merchant Tenure", "2,500 days verified merchant"),
            ("Model Risk Score", "0.1 / 100 (Cleared)"),
        ],
        "walkthrough": [
            {
                "step": 1,
                "title": "What Happened?",
                "concept": "Utility Bill Collection",
                "explanation": "120 individual retail electricity consumers pay their monthly power bills to Tata Power Mumbai. The merchant experiences 120 incoming transactions totaling ~₹2.9 Lakhs.",
                "highlight_nodes": ["MERCH_TATA_POWER"],
                "evidence": "120 incoming payments, in-degree = 120."
            },
            {
                "step": 2,
                "title": "Why Could This Look Like Fraud?",
                "concept": "High In-Degree Fan-In",
                "explanation": "Like the mule funnel in Case #1, Tata Power has a massive 120:0 in-degree ratio. Crude heuristic threshold rules that flag accounts receiving money from >50 senders would freeze this utility.",
                "highlight_nodes": ["MERCH_TATA_POWER"],
                "evidence": "In-degree = 120:0. Similar topological shape to a fraud funnel."
            },
            {
                "step": 3,
                "title": "How Does Graph AI Prove It's Legitimate?",
                "concept": "Ticket Distribution & Business Credentials",
                "explanation": "1) Ticket Size: Payments are ₹800 to ₹4,500 (average ₹2,450). Structuring band ratio is exactly 0.00%. 2) Outflow: Funds are retained or transferred to RBI-regulated escrow accounts. 3) Tenure: 2,500 days of verified operations as a registered Indian power utility.",
                "highlight_nodes": ["MERCH_TATA_POWER"],
                "evidence": "Structuring band ratio = 0.00%, verified utility subtype."
            },
            {
                "step": 4,
                "title": "Model Interpretation",
                "concept": "Negative Attribution Weights",
                "explanation": "Risk Score: 0.1/100 (CLEARED). The XGBoost model calculates that is_verified_business=1 and structuring_tx_ratio=0.0 strongly depress the risk score.",
                "highlight_nodes": ["MERCH_TATA_POWER"],
                "evidence": "Model outputs zero suspicion based on verified utility classification."
            },
            {
                "step": 5,
                "title": "Comparison with Mule Funnel",
                "concept": "Contrast with Case #1",
                "explanation": "Case #1 had 45 accounts sending ₹49.9k structuring amounts. Case #6 has 120 accounts paying standard ₹2k electricity bills. Amount distribution and counterparty verification make the difference crystal clear.",
                "highlight_nodes": ["MERCH_TATA_POWER"],
                "evidence": "Clear segregation between structuring evasion and routine bill payments."
            },
            {
                "step": 6,
                "title": "Investigator Takeaway",
                "concept": "Operational Resilience",
                "explanation": "Graph intelligence understands business context. Tata Power operates without account freezes or unnecessary compliance inquiries.",
                "highlight_nodes": ["MERCH_TATA_POWER"],
                "evidence": "Elevator Pitch Summary: Tata Power received 120 payments. Instead of mistaking high in-degree for a mule funnel, Graph AI checked ticket distribution and verified utility status, clearing the entity at 0.1/100."
            }
        ]
    },
    "LEGIT_WEDDING_SPEND": {
        "index_num": 7,
        "title": "High-Spend Family Wedding (Legitimate)",
        "badge": "Life Event",
        "category": "LEGIT",
        "default_layout": "concentric",
        "focal_node": "ACC_WEDDING_FAMILY",
        "exposure_inr": "₹ 0.0 (Cleared)",
        "exposure_short": "₹ 0.0",
        "entities_count": "10 entities",
        "risk_pct": 0.2,
        "severity": "Cleared",
        "one_liner": "Sharma Family spends ₹8 Lakhs across 8 verified vendors in 5 days. Velocity surge is cleared as a legitimate life event.",
        "scheme_details": "Sharma Family spends ₹8 Lakhs in 5 days across Tanishq Jewellers, Taj Hotels, Haldiram Catering, and IndiGo Airlines. Temporal velocity is high, but payments target diverse verified businesses from 1 family device.",
        "why_rules_fail": "Velocity spike thresholds often freeze cards during major life events (weddings, hospitalizations). FraudCircle verifies 100% verified counterparty status and zero circularity to clear the account.",
        "key_signals": [
            ("Counterparty Status", "100% Verified Merchants (0% Unverified)"),
            ("Device Integrity", "Single Private Family Device (0 Shared)"),
            ("Merchant Diversity", "8 Distinct Commercial Sectors"),
            ("Model Risk Score", "0.2 / 100 (Cleared)"),
        ],
        "walkthrough": [
            {
                "step": 1,
                "title": "What Happened?",
                "concept": "Major Life Event Expenditure",
                "explanation": "The Sharma Family Trust prepares for a family wedding, spending ₹8,00,000 across 5 days to pay vendors including Tanishq Jewellers, Taj Hotels Bangalore, Haldiram Catering, and IndiGo Airlines.",
                "highlight_nodes": ["ACC_WEDDING_FAMILY"],
                "evidence": "₹8 Lakhs spent across 8 transactions in 5 days."
            },
            {
                "step": 2,
                "title": "Why Could This Look Like Fraud?",
                "concept": "Velocity Spike Anomaly",
                "explanation": "Like the Sleeper Bust-Out in Case #4, this account shows a sudden 400% surge in spending velocity compared to its normal monthly baseline. Many bank anti-fraud bots freeze cards during weddings.",
                "highlight_nodes": ["ACC_WEDDING_FAMILY"],
                "evidence": "Sudden volume surge of ₹8 Lakhs."
            },
            {
                "step": 3,
                "title": "How Does Graph AI Prove It's Legitimate?",
                "concept": "Counterparty Quality & Category Diversity",
                "explanation": "In Case #4 (Bust-Out), money was drained to brand-new unverified accounts. Here: 1) All 8 counterparties are verified, established Indian corporations. 2) Payments disperse across diverse sectors (jewelry, hospitality, travel, catering). 3) All transactions originate from the family's known personal device.",
                "highlight_nodes": ["ACC_WEDDING_FAMILY", "DEV_WEDDING_1"],
                "evidence": "Unverified counterparty ratio = 0.00%. Counterparties are reputable verified merchants."
            },
            {
                "step": 4,
                "title": "Model Interpretation",
                "concept": "Contextual Feature Balance",
                "explanation": "Model Risk Score: 0.2/100 (CLEARED). The model weighs velocity_spike against unverified_counterparty_ratio=0.0 and shared_device_degree=0. The absence of unverified drain accounts suppresses the alert.",
                "highlight_nodes": ["ACC_WEDDING_FAMILY"],
                "evidence": "Clean counterparty graph overrides pure velocity anomaly."
            },
            {
                "step": 5,
                "title": "Comparison with Sleeper Bust-Out",
                "concept": "Contrasting Case #4 vs Case #7",
                "explanation": "Bust-Out: High velocity -> 8 unverified 4-day-old personal accounts -> instant cash drain. Wedding: High velocity -> 8 verified multi-year merchants -> consumer goods and services received.",
                "highlight_nodes": ["ACC_WEDDING_FAMILY"],
                "evidence": "Clear differentiation between criminal extraction and legitimate life event expenditure."
            },
            {
                "step": 6,
                "title": "Investigator Takeaway",
                "concept": "Customer Experience & Zero Interruption",
                "explanation": "By understanding counterparty context and device authenticity, Graph AI prevents embarrassing card declines and account freezes for high-net-worth customers during family milestones.",
                "highlight_nodes": ["ACC_WEDDING_FAMILY"],
                "evidence": "Elevator Pitch Summary: Sharma Family spent ₹8L in 5 days for a wedding. Rather than freezing their account due to a spending velocity spike, Graph AI verified that 100% of recipients were established merchants, clearing the account at 0.2/100."
            }
        ]
    },
}

# Global Metrics
total_nodes = len(nodes_df)
total_edges = len(edges_df)
fraud_rings = 4
fraud_edges_amount = edges_df[edges_df["scenario_tag"].str.startswith("FRAUD")]["amount"].sum()
exposure_lakhs = round(fraud_edges_amount / 100000, 2)

# ============================================================
# GRAPH SUBGRAPH BUILDER (LIGHT EDITORIAL STYLING)
# ============================================================
def get_subgraph_elements(scenario_tag=None, center_node=None, hops=1, filter_type="ALL", highlighted_nodes=None):
    """Build Cytoscape elements with clean light editorial styling."""
    if scenario_tag:
        sc_edges = edges_df[edges_df["scenario_tag"] == scenario_tag].copy()
        # Ensure all endpoint nodes (source and target) are included even if tagged with another category
        edge_nodes = set(sc_edges["source"]).union(set(sc_edges["target"]))
        sc_nodes = nodes_df[(nodes_df["scenario_tag"] == scenario_tag) | (nodes_df["node_id"].isin(edge_nodes))].copy()
    elif center_node:
        visited = {center_node}
        frontier = {center_node}
        for _ in range(hops):
            new_frontier = set()
            for n in frontier:
                out_targets = edges_df[edges_df["source"] == n]["target"].tolist()
                in_sources = edges_df[edges_df["target"] == n]["source"].tolist()
                new_frontier.update(out_targets + in_sources)
            frontier = new_frontier - visited
            visited.update(frontier)
        sc_nodes = nodes_df[nodes_df["node_id"].isin(visited)].copy()
        sc_edges = edges_df[
            edges_df["source"].isin(visited) & edges_df["target"].isin(visited)
        ].copy()
    else:
        return []

    # Type filtering
    if filter_type == "ACCOUNTS":
        valid_ids = set(sc_nodes[sc_nodes["node_type"] == "Account"]["node_id"])
        sc_nodes = sc_nodes[sc_nodes["node_id"].isin(valid_ids)]
    elif filter_type == "DEVICES":
        valid_ids = set(sc_nodes[sc_nodes["node_type"].isin(["Account", "Device", "IP_Address"])]["node_id"])
        sc_nodes = sc_nodes[sc_nodes["node_id"].isin(valid_ids)]
    elif filter_type == "MERCHANTS":
        valid_ids = set(sc_nodes[sc_nodes["node_type"].isin(["Account", "Merchant"])]["node_id"])
        sc_nodes = sc_nodes[sc_nodes["node_id"].isin(valid_ids)]

    # Final guarantee: Every edge MUST have both source and target present in sc_nodes to avoid Cytoscape canvas crashes
    valid_node_ids = set(sc_nodes["node_id"])
    sc_edges = sc_edges[sc_edges["source"].isin(valid_node_ids) & sc_edges["target"].isin(valid_node_ids)]

    elements = []
    highlight_set = set(highlighted_nodes or [])

    for _, row in sc_nodes.iterrows():
        nid = row["node_id"]
        ntype = row["node_type"]
        fraud = int(row.get("ground_truth_fraud", 0))
        pred = pred_map.get(nid, {})
        risk = float(pred.get("risk_score", 0.0))
        is_highlighted = nid in highlight_set

        # Editorial Node Styling
        if ntype == "Device":
            shape = "diamond"
            color = "#FCF5E8"
            border_color = "#9A6700"
            border_width = 2 if not is_highlighted else 3.5
            size = 32
            node_label = f"{nid}\n[Device]"
            text_color = "#202A36"
        elif ntype == "IP_Address":
            shape = "diamond"
            color = "#F1F5F9"
            border_color = "#64748B"
            border_width = 2 if not is_highlighted else 3.5
            size = 32
            node_label = f"{nid}\n[IP Address]"
            text_color = "#202A36"
        elif ntype == "Merchant":
            subtype = str(row.get("account_subtype", ""))
            shape = "round-rectangle"
            size = 44
            text_color = "#FFFFFF"
            if subtype == "Shell_Merchant" or fraud == 1:
                color = "#B93832"
                border_color = "#202A36"
                border_width = 2
                node_label = f"{row.get('name', nid)}\n[Shell Merchant]"
            else:
                color = "#176B57"
                border_color = "#0E4739"
                border_width = 2
                node_label = f"{row.get('name', nid)}\n[Verified Merchant]"
        else:  # Account
            shape = "ellipse"
            size = 38
            name_val = str(row.get("name", nid))
            if len(name_val) > 15:
                name_val = name_val[:13] + ".."

            if nid in ["ACC_FRAUD_MASTER", "ACC_EXT_CASHOUT"]:
                color = "#B93832"
                border_color = "#202A36"
                border_width = 3 if not is_highlighted else 4.5
                size = 48
                node_label = f"{name_val}\n[Master Hub]"
                text_color = "#FFFFFF"
            elif risk >= 70 or fraud == 1:
                color = "#B93832"
                border_color = "#202A36"
                border_width = 1.5 if not is_highlighted else 3.5
                node_label = f"{name_val}\n({risk:.0f}%)"
                text_color = "#FFFFFF"
            elif risk >= 30:
                color = "#FCF5E8"
                border_color = "#9A6700"
                border_width = 2
                node_label = f"{name_val}\n({risk:.0f}%)"
                text_color = "#202A36"
            else:
                color = "#FFFFFF"
                border_color = "#2457A6"
                border_width = 2
                node_label = f"{name_val}\n[Cleared]"
                text_color = "#202A36"

        if is_highlighted:
            border_color = "#2457A6"
            border_width = 4

        elements.append({
            "data": {
                "id": nid,
                "label": node_label,
                "node_type": ntype,
                "risk_score": risk,
                "color": color,
                "shape": shape,
                "size": size,
                "border_color": border_color,
                "border_width": border_width,
                "text_color": text_color,
            }
        })

    for _, row in sc_edges.iterrows():
        amt = float(row.get("amount", 0))
        etype = str(row.get("edge_type", ""))
        src, tgt = row["source"], row["target"]
        is_highlighted_edge = (src in highlight_set) and (tgt in highlight_set)

        if etype in ("LOGS_IN_FROM", "SHARES_IP"):
            edge_color = "#64748B" if not is_highlighted_edge else "#2457A6"
            edge_label = "Device Login" if etype == "LOGS_IN_FROM" else "IP Access"
            width = 1.2 if not is_highlighted_edge else 2.5
            dash_style = "dashed"
        else:
            is_struct = 48000 <= amt <= 49999
            if is_struct:
                edge_color = "#B93832"
                width = 2.2 if not is_highlighted_edge else 3.5
            elif amt > 50000:
                edge_color = "#596473"
                width = 1.8 if not is_highlighted_edge else 3.0
            else:
                edge_color = "#CBD5E1"
                width = 1.2 if not is_highlighted_edge else 2.5

            edge_label = f"₹ {amt:,.0f}" if amt > 0 else etype.replace("_", " ")
            dash_style = "solid"

        elements.append({
            "data": {
                "id": f"{src}->{tgt}:{etype}",
                "source": src,
                "target": tgt,
                "label": edge_label,
                "edge_color": edge_color,
                "width": width,
                "line_style": dash_style,
            }
        })

    return elements


# ============================================================
# SHAP FEATURE ATTRIBUTION CHART (EDITORIAL THEME)
# ============================================================
def build_shap_chart(node_id):
    """Horizontal diverging bar chart with clean editorial palette."""
    pred = pred_map.get(node_id)
    if not pred or "all_shap_values" not in pred:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            annotations=[dict(text="Select an Account or Merchant to view feature attribution",
                              xref="paper", yref="paper", showarrow=False,
                              font=dict(color="#697586", size=12))]
        )
        return fig

    shap_vals = pred["all_shap_values"]
    sorted_feats = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:6]

    features = [f[0].replace("_", " ").title() for f in sorted_feats][::-1]
    values = [f[1] for f in sorted_feats][::-1]
    colors = ["#B93832" if v > 0 else "#176B57" for v in values]
    text_labels = [f"+{v:.3f} (+Risk)" if v > 0 else f"{v:.3f} (-Risk)" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker=dict(color=colors, line=dict(color="#D8DDE3", width=1)),
        text=text_labels,
        textposition="outside",
        textfont=dict(color="#202A36", size=11, family="IBM Plex Sans, sans-serif"),
        cliponaxis=False,
    ))
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#202A36", family="IBM Plex Sans, sans-serif", size=11),
        xaxis=dict(
            title=dict(text="SHAP Value (+Increases Risk / -Reduces Risk)", font=dict(size=11, color="#596473")),
            zeroline=True,
            zerolinecolor="#202A36",
            zerolinewidth=1.5,
            gridcolor="#EAE7DF",
            tickfont=dict(size=10, color="#596473"),
        ),
        yaxis=dict(
            gridcolor="#EAE7DF",
            tickfont=dict(size=11, color="#202A36"),
        ),
        margin=dict(l=150, r=60, t=10, b=40),
        height=220,
        bargap=0.35,
    )
    return fig


# ============================================================
# CYTOSCAPE STYLESHEET (LIGHT EDITORIAL)
# ============================================================
cyto_stylesheet = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "background-color": "data(color)",
            "shape": "data(shape)",
            "width": "data(size)",
            "height": "data(size)",
            "font-size": "10.5px",
            "font-weight": "600",
            "font-family": "IBM Plex Sans, sans-serif",
            "color": "data(text_color)",
            "text-valign": "center",
            "text-halign": "center",
            "border-color": "data(border_color)",
            "border-width": "data(border_width)",
            "text-wrap": "wrap",
            "text-max-width": "80px",
        }
    },
    {
        "selector": "edge",
        "style": {
            "label": "data(label)",
            "line-color": "data(edge_color)",
            "line-style": "data(line_style)",
            "target-arrow-color": "data(edge_color)",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "font-size": "9.5px",
            "font-weight": "500",
            "font-family": "IBM Plex Mono, monospace",
            "color": "#202A36",
            "text-rotation": "autorotate",
            "width": "data(width)",
            "arrow-scale": 0.95,
            "text-background-color": "#FFFFFF",
            "text-background-opacity": 0.85,
            "text-background-padding": "2px",
            "text-background-shape": "roundrectangle",
        }
    },
    {
        "selector": ":selected",
        "style": {
            "border-color": "#2457A6",
            "border-width": 4,
            "shadow-blur": 8,
            "shadow-color": "#2457A6",
            "shadow-opacity": 0.3,
        }
    }
]

# ============================================================
# APP INITIALIZATION
# ============================================================
cyto.load_extra_layouts()

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="FraudCircle | AI Financial Network Intelligence"
)

# Neo4j Synchronization Modal
neo4j_modal = dbc.Modal(
    [
        dbc.ModalHeader(
            dbc.ModalTitle("Neo4j Graph Database Synchronization", style={"fontSize": "16px", "fontWeight": "600", "color": "#202A36"}),
            style={"backgroundColor": "#FFFFFF", "borderBottom": "1px solid #D8DDE3"}
        ),
        dbc.ModalBody([
            html.P("Synchronize FraudCircle's financial network topology directly into Neo4j AuraDB Cloud or local Neo4j Desktop for multi-hop graph querying.", style={"fontSize": "13px", "color": "#596473", "marginBottom": "16px"}),
            html.Label("Connection Protocol & URI", style={"fontSize": "12px", "color": "#202A36", "fontWeight": "600"}),
            dbc.Input(id="neo4j-uri", value="neo4j://127.0.0.1:7687", placeholder="neo4j://127.0.0.1:7687 or neo4j+s://...databases.neo4j.io", className="mb-2"),
            dbc.Row([
                dbc.Col([
                    html.Label("Username", style={"fontSize": "12px", "color": "#202A36", "fontWeight": "600"}),
                    dbc.Input(id="neo4j-user", value="neo4j", className="mb-2"),
                ], width=6),
                dbc.Col([
                    html.Label("Password", style={"fontSize": "12px", "color": "#202A36", "fontWeight": "600"}),
                    dbc.Input(id="neo4j-pass", type="password", value="password", className="mb-2"),
                ], width=6),
            ]),
            html.Label("Export Scope", style={"fontSize": "12px", "color": "#202A36", "fontWeight": "600"}),
            dcc.Dropdown(
                id="neo4j-scope-dropdown",
                options=[{"label": "Current Active Subgraph Only", "value": "current"}, {"label": "Entire Network (All 1,931 Nodes)", "value": "all"}],
                value="current",
                className="mb-3",
            ),
            dbc.Alert(id="neo4j-status-alert", is_open=False, duration=8000, style={"fontSize": "12px"}),
            html.Div([
                html.Span("Offline Operational Guarantee: ", style={"fontWeight": "600", "color": "#176B57"}),
                html.Span("The platform operates 100% self-contained in memory via NetworkX. Neo4j connectivity is optional.", style={"color": "#596473"}),
            ], style={"fontSize": "12px", "padding": "10px 14px", "backgroundColor": "#F9F8F5", "borderRadius": "4px", "border": "1px solid #D8DDE3"}),
        ], style={"backgroundColor": "#FFFFFF"}),
        dbc.ModalFooter([
            dbc.Button("Export .cypher Script", id="btn-export-cypher-file", className="btn-fc-secondary me-2"),
            dbc.Button("Push to Neo4j", id="btn-sync-neo4j-action", className="btn-fc-primary me-2"),
            dbc.Button("Close", id="btn-close-neo4j-modal", className="btn-fc-secondary"),
        ], style={"backgroundColor": "#FFFFFF", "borderTop": "1px solid #D8DDE3"}),
    ],
    id="neo4j-modal",
    is_open=False,
    size="lg",
)

# FIU-IND STR Form Modal
fiu_str_modal = dbc.Modal(
    [
        dbc.ModalHeader(
            dbc.ModalTitle("Financial Intelligence Unit - India (FIU-IND) STR Preview", style={"fontSize": "16px", "fontWeight": "600", "color": "#202A36"}),
            style={"backgroundColor": "#FFFFFF", "borderBottom": "1px solid #D8DDE3"}
        ),
        dbc.ModalBody(id="fiu-modal-content", style={"backgroundColor": "#F4F1EA", "maxHeight": "72vh", "overflowY": "auto"}),
        dbc.ModalFooter([
            dbc.Button("Download STR JSON", id="btn-download-str-json", className="btn-fc-primary me-2"),
            dbc.Button("Close Preview", id="btn-close-fiu-modal", className="btn-fc-secondary"),
        ], style={"backgroundColor": "#FFFFFF", "borderTop": "1px solid #D8DDE3"}),
    ],
    id="fiu-str-modal",
    is_open=False,
    size="lg",
)

# ============================================================
# TOP EDITORIAL NAVIGATION HEADER
# ============================================================

# ============================================================
# TOP EDITORIAL NAVIGATION HEADER
# ============================================================
def render_navbar():
    return html.Nav([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("FRAUD", style={"color": "#B93832", "fontWeight": "800", "fontSize": "20px", "letterSpacing": "-0.03em"}),
                        html.Span("CIRCLE", style={"color": "#2457A6", "fontWeight": "800", "fontSize": "20px", "letterSpacing": "-0.03em"}),
                        html.Span("Financial Network Intelligence", style={"fontSize": "12px", "color": "#596473", "marginLeft": "12px", "borderLeft": "1px solid #D8DDE3", "paddingLeft": "12px"}),
                    ], id="nav-brand-logo", className="d-flex align-items-center clickable", style={"cursor": "pointer"}),
                ], width=12, md=5),

                dbc.Col([
                    html.Div([
                        html.A("Overview", id="nav-btn-overview", className="fc-nav-item active", n_clicks=0),
                        html.A("Fraud Cases", id="nav-btn-cases", className="fc-nav-item", n_clicks=0),
                        html.A("Investigation Workspace", id="nav-btn-workspace", className="fc-nav-item", n_clicks=0),
                        html.A("False-Positive Benchmarks", id="nav-btn-benchmarks", className="fc-nav-item", n_clicks=0),
                        html.A("Report", id="nav-btn-report", className="fc-nav-item", n_clicks=0),
                        html.A("Methodology", id="nav-btn-about", className="fc-nav-item", n_clicks=0),
                    ], className="d-flex align-items-center justify-content-md-end flex-wrap mt-2 mt-md-0"),
                ], width=12, md=7),
            ], className="align-items-center"),
        ], fluid=True, className="px-4"),
    ], className="fc-navbar")


# ============================================================
# VIEW 1: OVERVIEW PAGE
# ============================================================
def render_overview_view():
    # Table rows for queue
    rows = []
    for sc_key in SCENARIO_KEYS:
        sc = SCENARIOS[sc_key]
        is_fraud = sc["category"] == "FRAUD"
        badge_cls = "badge-critical" if is_fraud else "badge-verified"
        dot_color = "#B93832" if is_fraud else "#176B57"

        rows.append(
            html.Tr([
                html.Td(f"CASE #{sc['index_num']}", className="font-mono", style={"fontWeight": "600", "fontSize": "12px", "color": "#2457A6"}),
                html.Td([
                    html.Div(sc["title"], style={"fontWeight": "600", "color": "#202A36"}),
                    html.Div(sc["one_liner"], style={"fontSize": "12px", "color": "#596473", "marginTop": "2px"}),
                ]),
                html.Td(html.Span(sc["badge"], className="badge-status badge-neutral")),
                html.Td([
                    html.Span(style={"width": "6px", "height": "6px", "borderRadius": "50%", "backgroundColor": dot_color, "display": "inline-block", "marginRight": "6px"}),
                    html.Span(sc["severity"], style={"fontWeight": "600", "color": dot_color}),
                ]),
                html.Td(sc["exposure_inr"], className="font-mono", style={"fontWeight": "600", "color": "#B93832" if is_fraud else "#596473"}),
                html.Td(sc["entities_count"], style={"color": "#596473"}),
                html.Td(
                    dbc.Button("Investigate →", id={"type": "open-workspace-btn", "index": sc_key}, className="btn-fc-primary", size="sm", style={"padding": "4px 10px", "fontSize": "12px"})
                ),
            ], className="fc-table-row")
        )

    return html.Div(id="overview-view-container", children=[
        dbc.Container([
            # Hero Header
            dbc.Row([
                dbc.Col([
                    html.H1("Fraud Intelligence", style={"fontSize": "30px", "fontWeight": "700", "color": "#202A36", "marginBottom": "6px", "letterSpacing": "-0.03em"}),
                    html.P("Investigate suspicious networks, trace fund movement and understand why activity was flagged.",
                           style={"fontSize": "15px", "color": "#596473", "marginBottom": "24px"}),
                ], width=12),
            ]),

            # 4 Summary Metrics
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div("Monitored Entities", className="metric-summary-label"),
                        html.Div(f"{total_nodes:,}", className="metric-summary-value"),
                        html.Div(f"{total_edges:,} financial relationships tracked", className="metric-summary-sub"),
                    ], className="metric-summary-card"),
                ], width=12, sm=6, lg=3, className="mb-3"),

                dbc.Col([
                    html.Div([
                        html.Div("Active Fraud Rings", className="metric-summary-label"),
                        html.Div(f"{fraud_rings} Rings", className="metric-summary-value", style={"color": "#B93832"}),
                        html.Div("4/4 flagged • 0 false negatives", className="metric-summary-sub"),
                    ], className="metric-summary-card"),
                ], width=12, sm=6, lg=3, className="mb-3"),

                dbc.Col([
                    html.Div([
                        html.Div("Exposure Under Investigation", className="metric-summary-label"),
                        html.Div(f"₹ {exposure_lakhs:,.1f}L", className="metric-summary-value", style={"color": "#9A6700"}),
                        html.Div("Coordinated transactional volume", className="metric-summary-sub"),
                    ], className="metric-summary-card"),
                ], width=12, sm=6, lg=3, className="mb-3"),

                dbc.Col([
                    html.Div([
                        html.Div("Verified Suppression", className="metric-summary-label"),
                        html.Div("100.0%", className="metric-summary-value", style={"color": "#176B57"}),
                        html.Div("0 legitimate payroll/utility errors", className="metric-summary-sub"),
                    ], className="metric-summary-card"),
                ], width=12, sm=6, lg=3, className="mb-3"),
            ], className="mb-4"),

            # Prioritized Investigation Queue Table
            html.Div([
                html.Div([
                    html.H2("Prioritized Investigation Queue", style={"fontSize": "18px", "fontWeight": "700", "color": "#202A36", "margin": "0"}),
                    html.Span("Select any case to open its dedicated investigation workspace", style={"fontSize": "13px", "color": "#596473"}),
                ], className="d-flex justify-content-between align-items-center mb-3 flex-wrap"),

                html.Table([
                    html.Thead(
                        html.Tr([
                            html.Th("Case ID"),
                            html.Th("Case Title & Modus Operandi"),
                            html.Th("Typology"),
                            html.Th("Risk Severity"),
                            html.Th("Exposure (INR)"),
                            html.Th("Entities"),
                            html.Th("Action"),
                        ])
                    ),
                    html.Tbody(rows),
                ], className="fc-table mb-4"),
            ]),

            # Legitimate Benchmark Section
            html.Div([
                html.H3("False-Positive Suppression Benchmarks", style={"fontSize": "16px", "fontWeight": "700", "color": "#202A36", "marginBottom": "6px"}),
                html.P("FraudCircle distinguishes legitimate high-degree hubs (e.g. corporate payroll, utility billing, wedding spends) from coordinated crime rings using multi-signal graph topology rather than crude volume thresholds.",
                       style={"fontSize": "13px", "color": "#596473", "marginBottom": "12px"}),
                dbc.Button("Explore Legitimate Activity Benchmarks →", id={"type": "nav-btn", "target": "benchmarks"}, className="btn-fc-secondary", size="sm"),
            ], className="fc-surface p-4 mb-4"),

        ], fluid=True, className="px-4 py-3"),
    ])


# ============================================================
# VIEW 2: FRAUD CASES EXPLORER
# ============================================================
def render_cases_view(filter_val="ALL"):
    cards = []
    for sc_key in SCENARIO_KEYS:
        sc = SCENARIOS[sc_key]
        is_fraud = sc["category"] == "FRAUD"
        if filter_val == "FRAUD" and not is_fraud:
            continue
        if filter_val == "LEGIT" and is_fraud:
            continue

        dot_color = "#B93832" if is_fraud else "#176B57"

        cards.append(
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span(f"CASE #{sc['index_num']}", className="font-mono", style={"fontSize": "12px", "fontWeight": "600", "color": "#2457A6"}),
                        html.Div([
                            html.Span(style={"width": "6px", "height": "6px", "borderRadius": "50%", "backgroundColor": dot_color, "display": "inline-block", "marginRight": "6px"}),
                            html.Span(sc["severity"], style={"fontSize": "12px", "fontWeight": "600", "color": dot_color}),
                        ], className="d-flex align-items-center"),
                    ], className="d-flex justify-content-between align-items-center mb-2"),

                    html.H4(sc["title"], style={"fontSize": "16px", "fontWeight": "700", "color": "#202A36", "marginBottom": "6px"}),
                    html.P(sc["one_liner"], style={"fontSize": "13px", "color": "#596473", "marginBottom": "16px", "minHeight": "40px"}),

                    html.Div([
                        html.Div([
                            html.Span("Exposure: ", style={"color": "#697586", "fontSize": "12px"}),
                            html.Span(sc["exposure_inr"], className="font-mono", style={"fontWeight": "600", "color": "#202A36", "fontSize": "12px"}),
                        ]),
                        html.Div([
                            html.Span("Entities: ", style={"color": "#697586", "fontSize": "12px"}),
                            html.Span(sc["entities_count"], style={"fontWeight": "600", "color": "#202A36", "fontSize": "12px"}),
                        ]),
                    ], className="d-flex justify-content-between align-items-center py-2 mb-3", style={"borderTop": "1px solid #EAE7DF", "borderBottom": "1px solid #EAE7DF"}),

                    dbc.Button("Open Investigation Workspace →", id={"type": "open-workspace-btn", "index": sc_key}, className="btn-fc-primary w-100", size="sm"),
                ], className="case-explorer-card"),
            ], width=12, md=6, lg=4, className="mb-3")
        )

    return html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Case Explorer", style={"fontSize": "28px", "fontWeight": "700", "color": "#202A36", "marginBottom": "4px"}),
                    html.P("Review suspected fraud typologies and legitimate benchmark cases with real network evidence.",
                           style={"fontSize": "14px", "color": "#596473", "marginBottom": "20px"}),
                ], width=12, md=8),

                # Filter Buttons
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button("All Cases (7)", id={"type": "cases-filter-btn", "filter": "ALL"}, className=f"btn-fc-secondary {'active' if filter_val=='ALL' else ''}", size="sm"),
                        dbc.Button("Fraud Rings (4)", id={"type": "cases-filter-btn", "filter": "FRAUD"}, className=f"btn-fc-secondary {'active' if filter_val=='FRAUD' else ''}", size="sm"),
                        dbc.Button("Cleared Benchmarks (3)", id={"type": "cases-filter-btn", "filter": "LEGIT"}, className=f"btn-fc-secondary {'active' if filter_val=='LEGIT' else ''}", size="sm"),
                    ]),
                ], width=12, md=4, className="d-flex align-items-center justify-content-md-end mb-3 mb-md-0"),
            ]),

            dbc.Row(cards),

        ], fluid=True, className="px-4 py-3"),
    ], id="cases-view-container")


# ============================================================
# VIEW 3: DEDICATED INVESTIGATION WORKSPACE
# ============================================================

def build_entity_inspector_content(nid, case_key):
    node_row = nodes_df[nodes_df["node_id"] == nid]
    entity_feat = feat_map.get(nid, {})
    pred = pred_map.get(nid, {})

    if node_row.empty:
        return html.Div(f"Entity {nid} metadata not found.", style={"color": "#B93832"})

    nr = node_row.iloc[0]
    risk = float(pred.get("risk_score", 0.0))
    is_fraud = risk > 50

    return html.Div([
        # Header
        html.Div([
            html.Span("Entity Inspector", style={"fontSize": "11px", "color": "#697586", "textTransform": "uppercase", "fontWeight": "700"}),
            html.Span(f"{risk:.1f}/100 Risk", className="badge-status badge-critical" if is_fraud else "badge-status badge-verified"),
        ], className="d-flex justify-content-between align-items-center mb-2"),

        html.H4(str(nr.get("name", nid)), style={"fontSize": "16px", "fontWeight": "700", "color": "#202A36", "marginBottom": "2px"}),
        html.Div(nid, className="font-mono", style={"fontSize": "12px", "color": "#2457A6", "marginBottom": "10px"}),

        html.Div([
            html.Span("Type: ", style={"color": "#596473", "fontSize": "12px"}),
            html.Span(str(nr.get("node_type", "Account")), style={"fontWeight": "600", "fontSize": "12px", "color": "#202A36"}),
            html.Span(" • Subtype: ", style={"color": "#596473", "fontSize": "12px"}),
            html.Span(str(nr.get("account_subtype", "Personal")), style={"fontWeight": "600", "fontSize": "12px", "color": "#202A36"}),
        ], className="mb-3"),

        html.Hr(style={"borderColor": "#D8DDE3", "margin": "8px 0"}),

        # Real Graph Metrics
        html.Div([
            html.Div("Topological Graph Properties", style={"fontSize": "11px", "fontWeight": "700", "color": "#596473", "textTransform": "uppercase", "marginBottom": "6px"}),
            html.Div([
                html.Div([
                    html.Div("In / Out Degree", style={"fontSize": "11px", "color": "#697586"}),
                    html.Div(f"{int(entity_feat.get('in_degree', 0))} : {int(entity_feat.get('out_degree', 0))}", className="font-mono", style={"fontWeight": "700", "color": "#202A36"}),
                ], className="col-6 mb-2"),
                html.Div([
                    html.Div("Flow Pass-Through", style={"fontSize": "11px", "color": "#697586"}),
                    html.Div(f"{entity_feat.get('flow_through_ratio', 0.0):.1%}", className="font-mono", style={"fontWeight": "700", "color": "#B93832" if entity_feat.get('flow_through_ratio', 0.0) > 0.8 else "#202A36"}),
                ], className="col-6 mb-2"),
                html.Div([
                    html.Div("Structuring Ratio", style={"fontSize": "11px", "color": "#697586"}),
                    html.Div(f"{entity_feat.get('structuring_amount_ratio', 0.0):.1%}", className="font-mono", style={"fontWeight": "700", "color": "#B93832" if entity_feat.get('structuring_amount_ratio', 0.0) > 0.5 else "#202A36"}),
                ], className="col-6 mb-2"),
                html.Div([
                    html.Div("Cycle Membership", style={"fontSize": "11px", "color": "#697586"}),
                    html.Div(f"{int(entity_feat.get('cycle_participant', 0))} (Len: {int(entity_feat.get('cycle_length', 0))})", className="font-mono", style={"fontWeight": "700", "color": "#B93832" if entity_feat.get('cycle_participant', 0) > 0 else "#202A36"}),
                ], className="col-6 mb-2"),
                html.Div([
                    html.Div("PageRank Centrality", style={"fontSize": "11px", "color": "#697586"}),
                    html.Div(f"{entity_feat.get('pagerank', 0.0):.4f}", className="font-mono", style={"fontWeight": "700", "color": "#2457A6"}),
                ], className="col-6 mb-2"),
                html.Div([
                    html.Div("Device Co-Logins", style={"fontSize": "11px", "color": "#697586"}),
                    html.Div(f"{int(entity_feat.get('device_risk_degree', 0))}", className="font-mono", style={"fontWeight": "700", "color": "#B93832" if entity_feat.get('device_risk_degree', 0) > 1 else "#202A36"}),
                ], className="col-6 mb-2"),
            ], className="row g-2"),
        ], className="mb-3"),

        html.Hr(style={"borderColor": "#D8DDE3", "margin": "8px 0"}),

        # Compliance & KYC Status
        html.Div([
            html.Div("Compliance & Identity Verification", style={"fontSize": "11px", "fontWeight": "700", "color": "#596473", "textTransform": "uppercase", "marginBottom": "6px"}),
            html.Div([
                html.Span("KYC Status: ", style={"fontSize": "12px", "color": "#596473"}),
                html.Span("Verified (Aadhaar/PAN)" if nr.get("kyc_verified") == 1 else "Unverified Shell",
                          style={"fontSize": "12px", "fontWeight": "600", "color": "#176B57" if nr.get("kyc_verified") == 1 else "#B93832"}),
            ], className="mb-1"),
            html.Div([
                html.Span("Account Tenure: ", style={"fontSize": "12px", "color": "#596473"}),
                html.Span(f"{nr.get('account_age_days', 0)} days", className="font-mono", style={"fontSize": "12px", "color": "#202A36"}),
            ], className="mb-1"),
            html.Div([
                html.Span("Ground Truth: ", style={"fontSize": "12px", "color": "#596473"}),
                html.Span("Fraud Synthetic Ring" if nr.get("ground_truth_fraud") == 1 else "Legitimate Baseline",
                          style={"fontSize": "12px", "fontWeight": "600", "color": "#B93832" if nr.get("ground_truth_fraud") == 1 else "#176B57"}),
            ], className="mb-1"),
        ]),
    ])

def render_workspace_view(current_case_key, active_tab="summary", show_guide=False, guide_step=1, presentation_mode=False, graph_layout="concentric", graph_filter="ALL", selected_node=None, audit_logs=None):
    sc = SCENARIOS.get(current_case_key, SCENARIOS["FRAUD_SMURFING_FUNNEL"])
    is_fraud = sc["category"] == "FRAUD"
    dot_color = "#B93832" if is_fraud else "#176B57"

    # Contextual Walkthrough Data for active step
    walkthrough_steps = sc.get("walkthrough", [])
    step_data = walkthrough_steps[guide_step - 1] if (walkthrough_steps and 1 <= guide_step <= len(walkthrough_steps)) else None
    highlight_nodes = step_data.get("highlight_nodes", []) if (show_guide and step_data) else []

    # 1. Breadcrumbs & Top Case Header
    header_section = html.Div([
        # Breadcrumbs
        html.Div([
            html.Span("Overview", className="clickable", id={"type": "nav-btn", "target": "overview"}),
            html.Span("/", className="breadcrumb-separator"),
            html.Span("Fraud Cases", className="clickable", id={"type": "nav-btn", "target": "cases"}),
            html.Span("/", className="breadcrumb-separator"),
            html.Span(f"Case #{sc['index_num']}: {sc['title']}", style={"color": "#202A36", "fontWeight": "600"}),
        ], className="breadcrumb-trail"),

        # Title Row & Actions
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1(f"Case #{sc['index_num']}: {sc['title']}", style={"fontSize": "24px", "fontWeight": "700", "color": "#202A36", "margin": "0 12px 0 0"}),
                    html.Span(sc["badge"], className="badge-status badge-neutral me-2"),
                    html.Span([
                        html.Span(style={"width": "6px", "height": "6px", "borderRadius": "50%", "backgroundColor": dot_color, "display": "inline-block", "marginRight": "5px"}),
                        html.Span(sc["severity"], style={"color": dot_color, "fontWeight": "600", "fontSize": "12px"}),
                    ], className="badge-status badge-critical" if is_fraud else "badge-status badge-verified"),
                ], className="d-flex align-items-center flex-wrap mb-1"),
                html.P(f"Estimated Exposure: {sc['exposure_inr']} • Involved Entities: {sc['entities_count']}",
                       style={"fontSize": "13px", "color": "#596473", "margin": "0"}),
            ], width=12, md=7),

            dbc.Col([
                html.Div([
                    dbc.Button(
                        "Exit Walkthrough" if (show_guide and not presentation_mode) else "✨ Explain This Case",
                        id={"type": "guide-mode-btn", "action": "toggle-guide"},
                        className=f"btn-fc-primary {'active' if (show_guide and not presentation_mode) else ''} me-2",
                        size="sm"
                    ),
                    dbc.Button(
                        "Exit Presentation" if presentation_mode else "📺 Presentation Mode",
                        id={"type": "guide-mode-btn", "action": "toggle-presentation"},
                        className=f"btn-fc-secondary {'active' if presentation_mode else ''}",
                        size="sm"
                    ),
                ], className="d-flex align-items-center justify-content-md-end mt-2 mt-md-0"),
            ], width=12, md=5),
        ], className="align-items-center mb-3"),
    ])

    # 2. Interactive Guided Walkthrough Box (Step-by-Step in Normal Mode)
    guide_section = html.Div()
    if show_guide and step_data and not presentation_mode:
        guide_section = html.Div([
            html.Div([
                html.Div([
                    html.Span(f"STEP {step_data['step']} OF {len(walkthrough_steps)}", className="font-mono", style={"fontSize": "11px", "fontWeight": "700", "color": "#2457A6", "letterSpacing": "0.05em"}),
                    html.H3(step_data["title"], style={"fontSize": "18px", "fontWeight": "700", "color": "#202A36", "margin": "2px 0 0 0"}),
                ]),
                html.Span(step_data["concept"], className="badge-status badge-neutral"),
            ], className="d-flex justify-content-between align-items-center mb-2"),

            html.P(step_data["explanation"], style={"fontSize": "14px", "color": "#202A36", "lineHeight": "1.6", "marginBottom": "10px"}),

            html.Div([
                html.Strong("Relevant Evidence: ", style={"color": "#2457A6", "fontSize": "12px"}),
                html.Span(step_data["evidence"], style={"fontSize": "12px", "color": "#596473"}),
            ], style={"backgroundColor": "#F4F1EA", "padding": "8px 12px", "borderRadius": "4px", "marginBottom": "14px"}),

            html.Div([
                html.Div([
                    dbc.Button("◀ Previous Step", id={"type": "guide-step-btn", "direction": "prev"}, className="btn-fc-secondary me-2", size="sm", disabled=(guide_step <= 1)),
                    dbc.Button("Next Step ▶", id={"type": "guide-step-btn", "direction": "next"}, className="btn-fc-primary me-2", size="sm", disabled=(guide_step >= len(walkthrough_steps))),
                    dbc.Button("Restart", id={"type": "guide-step-btn", "direction": "restart"}, className="btn-fc-secondary", size="sm"),
                ]),
                dbc.Button("Close Guide", id={"type": "guide-mode-btn", "action": "close-guide"}, className="btn-fc-secondary", size="sm"),
            ], className="d-flex justify-content-between align-items-center"),
        ], className="walkthrough-guide-box")

    # In Presentation Mode, render full-canvas investigative walkthrough view!
    if presentation_mode and step_data:
        return html.Div([
            dbc.Container([
                # Presentation Mode Top Bar
                html.Div([
                    html.Div([
                        html.Span("● PRESENTATION MODE: ", style={"fontWeight": "700", "color": "#38BDF8", "marginRight": "8px"}),
                        html.Span(f"Case #{sc['index_num']} — {sc['title']}", style={"fontWeight": "600", "color": "#F1F5F9"}),
                    ]),
                    html.Div([
                        dbc.Button("◀ Prev Step", id={"type": "guide-step-btn", "direction": "prev"}, className="btn-fc-secondary me-2", size="sm", disabled=(guide_step <= 1)),
                        html.Span(f"Step {guide_step} of {len(walkthrough_steps)}", style={"fontSize": "12.5px", "margin": "0 8px", "color": "#F1F5F9", "fontWeight": "600"}),
                        (dbc.Button("Next Step ▶", id={"type": "guide-step-btn", "direction": "next"}, className="btn-fc-primary me-2", size="sm") if guide_step < len(walkthrough_steps) else dbc.Button("Finish Walkthrough ✓", id={"type": "guide-mode-btn", "action": "exit-presentation"}, className="btn-fc-primary me-2", size="sm")),
                        dbc.Button("Restart", id={"type": "guide-step-btn", "direction": "restart"}, className="btn-fc-secondary me-2", size="sm"),
                        dbc.Button("Exit Presentation", id={"type": "guide-mode-btn", "action": "exit-presentation"}, className="btn-fc-secondary", size="sm"),
                    ], className="d-flex align-items-center"),
                ], className="presentation-mode-bar"),

                # Walkthrough narrative cue card
                html.Div([
                    html.Div([
                        html.Span(f"STEP {step_data['step']} OF {len(walkthrough_steps)} • {step_data['concept']}", className="font-mono", style={"fontSize": "11px", "fontWeight": "700", "color": "#2457A6", "letterSpacing": "0.05em"}),
                        html.H3(step_data["title"], style={"fontSize": "19px", "fontWeight": "700", "color": "#202A36", "margin": "2px 0 6px 0"}),
                    ]),
                    html.P(step_data["explanation"], style={"fontSize": "14px", "color": "#202A36", "lineHeight": "1.6", "marginBottom": "8px"}),
                    html.Div([
                        html.Strong("Key Graph Observation: ", style={"color": "#2457A6", "fontSize": "12.5px"}),
                        html.Span(step_data["evidence"], style={"fontSize": "12.5px", "color": "#202A36"}),
                    ], style={"backgroundColor": "#F4F1EA", "padding": "8px 12px", "borderRadius": "4px"}),
                    html.Div([
                        html.Strong("🎤 Presenter Cue: ", style={"color": "#9A6700"}),
                        html.Span("Point to the highlighted node cluster on the network canvas below. Explain how the topological pattern exposes coordination that traditional transaction-level rules failed to flag.")
                    ], className="presenter-cue-box"),
                ], className="fc-surface p-3 mb-3"),

                # Graph Controls Toolbar in Presentation Mode
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Span("Layout: ", style={"fontSize": "12px", "color": "#596473", "fontWeight": "600", "marginRight": "6px"}),
                            dbc.ButtonGroup([
                                dbc.Button("Concentric", id={"type": "graph-layout-btn", "layout": "concentric"}, className=f"btn-fc-secondary {'active' if graph_layout=='concentric' else ''}", size="sm"),
                                dbc.Button("Force-Directed", id={"type": "graph-layout-btn", "layout": "cose"}, className=f"btn-fc-secondary {'active' if graph_layout=='cose' else ''}", size="sm"),
                                dbc.Button("Circular Loop", id={"type": "graph-layout-btn", "layout": "circle"}, className=f"btn-fc-secondary {'active' if graph_layout=='circle' else ''}", size="sm"),
                            ], className="me-3"),
                        ], width=12, md=6, className="d-flex align-items-center mb-2 mb-md-0"),

                        dbc.Col([
                            html.Span("Filter: ", style={"fontSize": "12px", "color": "#596473", "fontWeight": "600", "marginRight": "6px"}),
                            dbc.ButtonGroup([
                                dbc.Button("All Nodes", id={"type": "graph-filter-btn", "filter": "ALL"}, className=f"btn-fc-secondary {'active' if graph_filter=='ALL' else ''}", size="sm"),
                                dbc.Button("Accounts", id={"type": "graph-filter-btn", "filter": "ACCOUNTS"}, className=f"btn-fc-secondary {'active' if graph_filter=='ACCOUNTS' else ''}", size="sm"),
                                dbc.Button("Devices/IPs", id={"type": "graph-filter-btn", "filter": "DEVICES"}, className=f"btn-fc-secondary {'active' if graph_filter=='DEVICES' else ''}", size="sm"),
                                dbc.Button("Merchants", id={"type": "graph-filter-btn", "filter": "MERCHANTS"}, className=f"btn-fc-secondary {'active' if graph_filter=='MERCHANTS' else ''}", size="sm"),
                            ]),
                        ], width=12, md=6, className="d-flex align-items-center justify-content-md-end"),
                    ]),
                ], className="fc-surface p-2 mb-2"),

                # Graph Canvas and Interactive Entity Inspector in Presentation Mode
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            cyto.Cytoscape(
                                id="graph-canvas",
                                layout={"name": graph_layout, "animate": True, "animationDuration": 400},
                                style={"width": "100%", "height": "calc(100vh - 380px)", "minHeight": "480px", "backgroundColor": "#FFFFFF", "borderRadius": "6px"},
                                stylesheet=cyto_stylesheet,
                                elements=get_subgraph_elements(scenario_tag=current_case_key, filter_type=graph_filter, highlighted_nodes=highlight_nodes),
                                responsive=True,
                            ),
                            html.Div([
                                html.Span("Legend: ", style={"fontSize": "11px", "fontWeight": "600", "color": "#596473", "marginRight": "8px"}),
                                html.Span("● Fraud Entity (Crimson)", style={"fontSize": "11px", "color": "#B93832", "marginRight": "12px"}),
                                html.Span("● Cleared Account (White/Navy)", style={"fontSize": "11px", "color": "#2457A6", "marginRight": "12px"}),
                                html.Span("◆ Device / IP (Amber)", style={"fontSize": "11px", "color": "#9A6700", "marginRight": "12px"}),
                                html.Span("■ Verified Merchant (Green)", style={"fontSize": "11px", "color": "#176B57", "marginRight": "12px"}),
                                html.Span("✦ Highlighted (Active Step)", style={"fontSize": "11px", "fontWeight": "700", "color": "#2457A6"}),
                            ], className="p-2 border-top", style={"borderColor": "#D8DDE3", "backgroundColor": "#F9F8F5"}),
                        ], className="fc-surface overflow-hidden"),
                    ], width=12, lg=8, className="mb-3 mb-lg-0"),

                    dbc.Col([
                        html.Div(id="workspace-entity-inspector", children=[
                            build_entity_inspector_content(selected_node if selected_node and selected_node in nodes_df['node_id'].values else sc['focal_node'], current_case_key)
                        ], className="fc-surface p-3 h-100", style={"maxHeight": "calc(100vh - 380px)", "overflowY": "auto"}),
                    ], width=12, lg=4),
                ]),
            ], fluid=True, className="px-4 py-3"),
        ])

    # 3. Workspace Sub-Navigation Tabs
    workspace_tabs = html.Div([
        html.Div("1. Summary", id={"type": "ws-tab-btn", "tab": "summary"}, className=f"fc-tab {'active' if active_tab=='summary' else ''}", n_clicks=0),
        html.Div("2. Network Evidence", id={"type": "ws-tab-btn", "tab": "network"}, className=f"fc-tab {'active' if active_tab=='network' else ''}", n_clicks=0),
        html.Div("3. Why It Was Flagged", id={"type": "ws-tab-btn", "tab": "why"}, className=f"fc-tab {'active' if active_tab=='why' else ''}", n_clicks=0),
        html.Div("4. Model Explanation", id={"type": "ws-tab-btn", "tab": "xai"}, className=f"fc-tab {'active' if active_tab=='xai' else ''}", n_clicks=0),
        html.Div("5. Investigation Report", id={"type": "ws-tab-btn", "tab": "report"}, className=f"fc-tab {'active' if active_tab=='report' else ''}", n_clicks=0),
    ], className="fc-tabs")

    # Tab 1: Summary Sub-View
    summary_content = html.Div([
        dbc.Row([
            # Left: Modus Operandi & Detection Logic
            dbc.Col([
                html.Div([
                    html.H4("Case Narrative & Modus Operandi", style={"fontSize": "15px", "fontWeight": "700", "color": "#202A36", "marginBottom": "8px"}),
                    html.P(sc["scheme_details"], style={"fontSize": "14px", "color": "#202A36", "lineHeight": "1.6", "marginBottom": "16px"}),

                    html.H4("Why Conventional Rules Missed This Case", style={"fontSize": "15px", "fontWeight": "700", "color": "#202A36", "marginBottom": "8px"}),
                    html.P(sc["why_rules_fail"], style={"fontSize": "14px", "color": "#596473", "lineHeight": "1.6", "marginBottom": "16px"}),

                    html.H4("Key Topological Metrics", style={"fontSize": "15px", "fontWeight": "700", "color": "#202A36", "marginBottom": "8px"}),
                    html.Div([
                        html.Div([
                            html.Span(f"{k}: ", style={"color": "#596473", "fontWeight": "500"}),
                            html.Strong(v, style={"color": "#202A36"}),
                        ], style={"backgroundColor": "#F4F1EA", "padding": "6px 12px", "borderRadius": "4px", "marginRight": "8px", "marginBottom": "8px", "fontSize": "12px", "display": "inline-block"})
                        for k, v in sc["key_signals"]
                    ], className="mb-3"),

                    dbc.Button("Explore Network Graph →", id={"type": "ws-tab-btn", "tab": "network"}, className="btn-fc-primary", size="sm"),
                ], className="fc-surface p-4 h-100"),
            ], width=12, md=7, className="mb-3 mb-md-0"),

            # Right: Primary Suspect Dossier
            dbc.Col([
                html.Div([
                    html.H4("Primary Subject Dossier", style={"fontSize": "15px", "fontWeight": "700", "color": "#202A36", "marginBottom": "12px"}),
                    html.Div(id="workspace-primary-dossier", children=[
                        html.Div([
                            html.Span("Entity Identifier", style={"fontSize": "11px", "color": "#697586", "textTransform": "uppercase"}),
                            html.Div(sc["focal_node"], className="font-mono", style={"fontSize": "15px", "fontWeight": "700", "color": "#2457A6", "marginBottom": "8px"}),

                            html.Span("Assessed Risk Score", style={"fontSize": "11px", "color": "#697586", "textTransform": "uppercase"}),
                            html.Div([
                                html.Span(f"{sc['risk_pct']:.1f}", style={"fontSize": "26px", "fontWeight": "700", "color": dot_color}),
                                html.Span("/ 100", style={"fontSize": "13px", "color": "#697586", "marginLeft": "4px"}),
                            ], className="mb-2"),

                            html.Div([
                                html.Span("Classification: ", style={"fontSize": "12px", "color": "#596473"}),
                                html.Strong(sc["severity"], style={"fontSize": "12px", "color": dot_color}),
                            ], className="mb-3"),

                            html.Div([
                                html.P(pred_map.get(sc["focal_node"], {}).get("narrative", "No narrative recorded for this node."),
                                       style={"fontSize": "13px", "color": "#596473", "lineHeight": "1.5", "backgroundColor": "#F9F8F5", "padding": "12px", "borderRadius": "4px"}),
                            ]),
                        ]),
                    ]),
                ], className="fc-surface p-4 h-100"),
            ], width=12, md=5),
        ]),
    ])

    # Tab 2: Network Evidence (Dominant Graph Canvas)
    network_content = html.Div([
        # Graph Controls Toolbar
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Span("Layout: ", style={"fontSize": "12px", "color": "#596473", "fontWeight": "600", "marginRight": "6px"}),
                    dbc.ButtonGroup([
                        dbc.Button("Concentric", id={"type": "graph-layout-btn", "layout": "concentric"}, className=f"btn-fc-secondary {'active' if graph_layout=='concentric' else ''}", size="sm"),
                        dbc.Button("Force-Directed", id={"type": "graph-layout-btn", "layout": "cose"}, className=f"btn-fc-secondary {'active' if graph_layout=='cose' else ''}", size="sm"),
                        dbc.Button("Circular Loop", id={"type": "graph-layout-btn", "layout": "circle"}, className=f"btn-fc-secondary {'active' if graph_layout=='circle' else ''}", size="sm"),
                    ], className="me-3"),
                ], width=12, md=6, className="d-flex align-items-center mb-2 mb-md-0"),

                dbc.Col([
                    html.Span("Filter: ", style={"fontSize": "12px", "color": "#596473", "fontWeight": "600", "marginRight": "6px"}),
                    dbc.ButtonGroup([
                        dbc.Button("All Nodes", id={"type": "graph-filter-btn", "filter": "ALL"}, className=f"btn-fc-secondary {'active' if graph_filter=='ALL' else ''}", size="sm"),
                        dbc.Button("Accounts", id={"type": "graph-filter-btn", "filter": "ACCOUNTS"}, className=f"btn-fc-secondary {'active' if graph_filter=='ACCOUNTS' else ''}", size="sm"),
                        dbc.Button("Devices/IPs", id={"type": "graph-filter-btn", "filter": "DEVICES"}, className=f"btn-fc-secondary {'active' if graph_filter=='DEVICES' else ''}", size="sm"),
                        dbc.Button("Merchants", id={"type": "graph-filter-btn", "filter": "MERCHANTS"}, className=f"btn-fc-secondary {'active' if graph_filter=='MERCHANTS' else ''}", size="sm"),
                    ]),
                ], width=12, md=6, className="d-flex align-items-center justify-content-md-end"),
            ]),
        ], className="fc-surface p-2 mb-2"),

        # Graph and Side Node Inspector
        dbc.Row([
            dbc.Col([
                html.Div([
                    cyto.Cytoscape(
                        id="graph-canvas",
                        layout={"name": graph_layout, "animate": True, "animationDuration": 400},
                        style={"width": "100%", "height": "calc(100vh - 350px)", "minHeight": "450px", "backgroundColor": "#FFFFFF", "borderRadius": "6px"},
                        stylesheet=cyto_stylesheet,
                        elements=get_subgraph_elements(scenario_tag=current_case_key, filter_type=graph_filter, highlighted_nodes=highlight_nodes),
                        responsive=True,
                    ),
                    # Compact Legend
                    html.Div([
                        html.Span("Legend: ", style={"fontSize": "11px", "fontWeight": "600", "color": "#596473", "marginRight": "8px"}),
                        html.Span("● Fraud Entity (Crimson)", style={"fontSize": "11px", "color": "#B93832", "marginRight": "12px"}),
                        html.Span("● Cleared Account (White/Navy)", style={"fontSize": "11px", "color": "#2457A6", "marginRight": "12px"}),
                        html.Span("◆ Device / IP (Amber)", style={"fontSize": "11px", "color": "#9A6700", "marginRight": "12px"}),
                        html.Span("■ Verified Merchant (Green)", style={"fontSize": "11px", "color": "#176B57"}),
                    ], className="p-2 border-top", style={"borderColor": "#D8DDE3", "backgroundColor": "#F9F8F5"}),
                ], className="fc-surface overflow-hidden"),
            ], width=12, lg=8, className="mb-3 mb-lg-0"),

            # Right Side Inspector for clicked node
            dbc.Col([
                html.Div(id="workspace-entity-inspector", children=[
                    build_entity_inspector_content(selected_node if selected_node and selected_node in nodes_df['node_id'].values else sc['focal_node'], current_case_key)
                ], className="fc-surface p-3 h-100", style={"maxHeight": "calc(100vh - 350px)", "overflowY": "auto"}),
            ], width=12, lg=4),
        ]),
    ])

    # Tab 3: Why It Was Flagged (Evidence Breakdown & Legal Context)
    why_content = html.Div([
        html.Div([
            html.H3("Evidence Assessment & Legal Context", style={"fontSize": "18px", "fontWeight": "700", "color": "#202A36", "marginBottom": "6px"}),
            html.P("Separating observed transactional facts from algorithmic model interpretations and legal thresholds.",
                   style={"fontSize": "14px", "color": "#596473", "marginBottom": "20px"}),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("1. Observed Facts (Ground Truth)", style={"fontSize": "14px", "fontWeight": "700", "color": "#202A36", "marginBottom": "8px"}),
                        html.Ul([
                            html.Li(f"Transaction timestamps occurred within an accelerated {sc.get('entities_count')} cluster.", style={"marginBottom": "6px"}),
                            html.Li(f"Primary focal entity {sc['focal_node']} handled {sc['exposure_inr']} in aggregate volume.", style={"marginBottom": "6px"}),
                            html.Li("Counterparty telemetry and device metadata exhibit abnormal clustering.", style={"marginBottom": "6px"}),
                        ], style={"fontSize": "13px", "color": "#596473", "paddingLeft": "18px"}),
                    ], className="fc-surface p-3 mb-3"),
                ], width=12, md=6),

                dbc.Col([
                    html.Div([
                        html.H4("2. Algorithmic Interpretation", style={"fontSize": "14px", "fontWeight": "700", "color": "#202A36", "marginBottom": "8px"}),
                        html.Ul([
                            html.Li(f"XGBoost Topological Classifier scored this entity at {sc['risk_pct']:.1f}/100 suspicion.", style={"marginBottom": "6px"}),
                            html.Li("Graph invariants (in-degree / cycle detection) deviate > 4 standard deviations from normal Barabási–Albert baseline.", style={"marginBottom": "6px"}),
                            html.Li("Notice: High risk score represents model suspicion, not a formal legal determination of guilt.", style={"marginBottom": "6px", "color": "#9A6700"}),
                        ], style={"fontSize": "13px", "color": "#596473", "paddingLeft": "18px"}),
                    ], className="fc-surface p-3 mb-3"),
                ], width=12, md=6),
            ]),

            html.Div([
                html.H4("Indian Banking Compliance (RBI / DPDP Act Framework)", style={"fontSize": "14px", "fontWeight": "700", "color": "#202A36", "marginBottom": "8px"}),
                html.P("Under the Prevention of Money Laundering Act (PMLA) 2002 and RBI Master Directions on KYC/AML, banks must monitor for structured payments that avoid the ₹50,000 threshold. Under the Digital Personal Data Protection (DPDP) Act 2023, automated decisions must maintain transparent, explainable audit trails without unauthorized profile extraction.",
                       style={"fontSize": "13px", "color": "#596473", "lineHeight": "1.6", "margin": "0"}),
            ], className="fc-surface p-3"),
        ], className="fc-surface p-4"),
    ])

    # Tab 4: Model Explanation (XAI with SHAP)
    xai_content = html.Div([
        html.Div([
            html.H3("Model Interpretability & Feature Attribution", style={"fontSize": "18px", "fontWeight": "700", "color": "#202A36", "marginBottom": "6px"}),
            html.P("SHAP TreeExplainer calculates the exact mathematical contribution of each graph feature toward the model's suspicion score.",
                   style={"fontSize": "14px", "color": "#596473", "marginBottom": "20px"}),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("SHAP Feature Attribution (Diverging Bar Chart)", style={"fontSize": "14px", "fontWeight": "700", "color": "#202A36", "marginBottom": "12px"}),
                        dcc.Graph(
                            id="workspace-shap-chart",
                            figure=build_shap_chart(sc["focal_node"]),
                            config={"displayModeBar": False},
                            style={"height": "240px"}
                        ),
                    ], className="fc-surface p-3 mb-3"),
                ], width=12, lg=7),

                dbc.Col([
                    html.Div([
                        html.H4("Topological Feature Glossary", style={"fontSize": "14px", "fontWeight": "700", "color": "#202A36", "marginBottom": "10px"}),
                        html.Div([
                            html.Div([html.Strong("In-Degree: "), html.Span("Number of distinct accounts sending funds to this node.", style={"color": "#596473"})], className="mb-2", style={"fontSize": "12px"}),
                            html.Div([html.Strong("Flow Pass-Through: "), html.Span("min(inflow, outflow) / max(inflow, outflow). High ratio indicates a transit or layering conduit.", style={"color": "#596473"})], className="mb-2", style={"fontSize": "12px"}),
                            html.Div([html.Strong("Structuring Ratio: "), html.Span("Fraction of payments falling in the ₹48,000–₹49,999 band just below the Indian PAN threshold.", style={"color": "#596473"})], className="mb-2", style={"fontSize": "12px"}),
                            html.Div([html.Strong("In-Cycle Flag: "), html.Span("1 if the node participates in any directed cycle <= 8 hops in the transaction network.", style={"color": "#596473"})], className="mb-2", style={"fontSize": "12px"}),
                            html.Div([html.Strong("Correlation vs Causality: "), html.Span("High SHAP values indicate strong correlation with trained fraud patterns, but require human investigator confirmation.", style={"color": "#9A6700"})], style={"fontSize": "12px"}),
                        ], style={"backgroundColor": "#F9F8F5", "padding": "12px", "borderRadius": "4px"}),
                    ], className="fc-surface p-3 mb-3"),
                ], width=12, lg=5),
            ]),
        ], className="fc-surface p-4"),
    ])

    # Tab 5: Investigation Report & Actions
    report_content = html.Div([
        html.Div([
            html.Div([
                html.H3("Investigation Report & Regulatory Enforcement", style={"fontSize": "18px", "fontWeight": "700", "color": "#202A36", "margin": "0"}),
                dbc.Button("Generate FIU-IND STR Form", id={"type": "open-modal-btn", "modal": "fiu"}, className="btn-fc-primary", size="sm"),
            ], className="d-flex justify-content-between align-items-center mb-3 flex-wrap"),

            # Actions Bar
            html.Div([
                html.Span("Human-in-the-Loop Actions: ", style={"fontSize": "13px", "fontWeight": "600", "color": "#202A36", "marginRight": "12px"}),
                dbc.Button("Quarantine Subgraph (Section 35A)", id={"type": "hitl-action-btn", "action": "quarantine"}, className="btn-fc-secondary me-2", size="sm", style={"color": "#B93832"}),
                dbc.Button("Whitelist as Verified Business", id={"type": "hitl-action-btn", "action": "whitelist"}, className="btn-fc-secondary me-2", size="sm", style={"color": "#176B57"}),
                dbc.Button("Sync Case to Neo4j", id={"type": "open-modal-btn", "modal": "neo4j"}, className="btn-fc-secondary", size="sm"),
            ], className="fc-surface p-3 mb-3 d-flex align-items-center flex-wrap"),

            # Audit Trail Log
            html.H4("Investigation Audit Trail", style={"fontSize": "14px", "fontWeight": "700", "color": "#202A36", "marginBottom": "8px"}),
            html.Div(id="workspace-audit-log", children=(
                [
                    html.Div([
                        html.Span(entry["time"], className="font-mono me-2", style={"color": "#697586", "fontSize": "12px"}),
                        html.Span(f"{entry['badge']}: ", style={"fontWeight": "600", "color": entry["color"], "fontSize": "12px"}),
                        html.Span(entry["text"], style={"fontSize": "12px", "color": "#202A36"}),
                    ], style={"padding": "8px 12px", "backgroundColor": "#F9F8F5", "borderRadius": "4px", "marginBottom": "4px", "borderLeft": f"3px solid {entry['color']}"})
                    for entry in (audit_logs.get(current_case_key, []) if isinstance(audit_logs, dict) else [])
                ] + [
                    html.Div([
                        html.Span(datetime.now().strftime("[%H:%M:%S]"), className="font-mono me-2", style={"color": "#697586", "fontSize": "12px"}),
                        html.Span("CASE INITIALIZED: ", style={"fontWeight": "600", "color": "#2457A6", "fontSize": "12px"}),
                        html.Span(f"Case #{sc['index_num']} loaded into active investigation session with full heterogeneous graph telemetry.", style={"fontSize": "12px", "color": "#596473"}),
                    ], style={"padding": "8px 12px", "backgroundColor": "#F9F8F5", "borderRadius": "4px", "marginBottom": "4px"})
                ]
            )),
        ], className="fc-surface p-4"),
    ])

    active_content = summary_content
    if active_tab == "network":
        active_content = network_content
    elif active_tab == "why":
        active_content = why_content
    elif active_tab == "xai":
        active_content = xai_content
    elif active_tab == "report":
        active_content = report_content

    return html.Div(id="workspace-view-container", children=[
        dbc.Container([
            header_section,
            guide_section,
            workspace_tabs,
            active_content,
        ], fluid=True, className="px-4 py-3"),
    ])


# ============================================================
# VIEW 4: LEGITIMATE ACTIVITY BENCHMARKS COMPARISON
# ============================================================
def render_benchmarks_view():
    comparisons = [
        {
            "title": "Study 1: Smurfing Mule Funnel vs. Corporate Payroll Hub",
            "fraud_name": "Case #1: Mule Structuring Funnel",
            "fraud_desc": "45 accounts send ~₹49.5k into master account; 98% pass-through out to offshore account. Risk: 99.9/100.",
            "legit_name": "Case #5: Infosys BPO Ltd Payroll",
            "legit_desc": "1 corporate account pays ₹35k–₹85k salaries to 50 employees on the 1st of the month. Risk: 0.2/100.",
            "why_different": "Topologically, both exhibit massive star fan-out/in. But Infosys has: 1) unique personal devices per employee, 2) zero return cycles, and 3) 5-year corporate tax registration. Graph AI recognizes corporate payroll and suppresses the false positive.",
            "fraud_case_key": "FRAUD_SMURFING_FUNNEL",
            "legit_case_key": "LEGIT_CORP_PAYROLL",
        },
        {
            "title": "Study 2: Shell Merchant Funnel vs. Tata Power Utility Hub",
            "fraud_name": "Case #2: Sybil Attack to Shell Merchant",
            "fraud_desc": "15 synthetic accounts co-logging from 1 device MAC transfer high payments into 12-day-old unverified merchant.",
            "legit_name": "Case #6: Tata Power Mumbai Utility Hub",
            "legit_desc": "120 individual accounts pay ₹800–₹4,500 electricity bills to Tata Power. In-degree = 120:0. Risk: 0.1/100.",
            "why_different": "Traditional rules flag any account receiving >50 transfers. Graph AI inspects ticket distributions: Tata Power receives small utility bill payments (0.00% structuring band) from distinct IP/devices into an established 7-year utility merchant.",
            "fraud_case_key": "FRAUD_SYBIL_DEVICE_RING",
            "legit_case_key": "LEGIT_UTILITY_HUB",
        },
        {
            "title": "Study 3: Sleeper Bust-Out Anomaly vs. High-Spend Family Wedding",
            "fraud_name": "Case #4: Sleeper Bust-Out Drain",
            "fraud_desc": "Dormant ₹1k transactions for 29 days, then 10,000% velocity surge draining ₹15L to 8 unverified 4-day-old accounts.",
            "legit_name": "Case #7: Sharma Family Wedding Spend",
            "legit_desc": "400% velocity surge spending ₹8 Lakhs in 5 days across jewelry, hotel, airline, and catering vendors. Risk: 0.2/100.",
            "why_different": "Both show severe velocity spikes. But in Case #7, 100% of receiving counterparties are established corporate vendors (Tanishq, Taj Hotels, IndiGo), originating from 1 personal family device. Zero funds drain to suspicious shell accounts.",
            "fraud_case_key": "FRAUD_BUST_OUT",
            "legit_case_key": "LEGIT_WEDDING_SPEND",
        },
    ]

    cards = []
    for comp in comparisons:
        cards.append(
            html.Div([
                html.H3(comp["title"], style={"fontSize": "17px", "fontWeight": "700", "color": "#202A36", "marginBottom": "12px"}),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Span("SUSPICIOUS PATTERN", className="badge-status badge-critical mb-2"),
                            html.H4(comp["fraud_name"], style={"fontSize": "14px", "fontWeight": "700", "color": "#B93832", "marginBottom": "4px"}),
                            html.P(comp["fraud_desc"], style={"fontSize": "13px", "color": "#596473", "marginBottom": "8px"}),
                            dbc.Button("View Fraud Case →", id={"type": "open-workspace-btn", "index": comp["fraud_case_key"]}, className="btn-fc-secondary", size="sm"),
                        ], className="fc-surface p-3 h-100", style={"borderLeft": "3px solid #B93832"}),
                    ], width=12, md=6, className="mb-3 mb-md-0"),

                    dbc.Col([
                        html.Div([
                            html.Span("VERIFIED LEGITIMATE BENCHMARK", className="badge-status badge-verified mb-2"),
                            html.H4(comp["legit_name"], style={"fontSize": "14px", "fontWeight": "700", "color": "#176B57", "marginBottom": "4px"}),
                            html.P(comp["legit_desc"], style={"fontSize": "13px", "color": "#596473", "marginBottom": "8px"}),
                            dbc.Button("View Benchmark Case →", id={"type": "open-workspace-btn", "index": comp["legit_case_key"]}, className="btn-fc-secondary", size="sm"),
                        ], className="fc-surface p-3 h-100", style={"borderLeft": "3px solid #176B57"}),
                    ], width=12, md=6),
                ], className="mb-3"),

                html.Div([
                    html.Strong("The Architectural Difference: ", style={"color": "#2457A6", "fontSize": "13px"}),
                    html.Span(comp["why_different"], style={"fontSize": "13px", "color": "#202A36"}),
                ], style={"backgroundColor": "#F4F1EA", "padding": "10px 14px", "borderRadius": "4px"}),

            ], className="fc-surface p-4 mb-4")
        )

    return html.Div(id="benchmarks-view-container", children=[
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Legitimate Activity & False-Positive Suppression", style={"fontSize": "28px", "fontWeight": "700", "color": "#202A36", "marginBottom": "6px"}),
                    html.P("Why high network connectivity alone does not prove fraud: Comparative studies between criminal typologies and legitimate banking hubs.",
                           style={"fontSize": "14px", "color": "#596473", "marginBottom": "24px"}),
                ], width=12),
            ]),

            html.Div(cards),

        ], fluid=True, className="px-4 py-3"),
    ])


# ============================================================
# VIEW 5: PRINT-FRIENDLY FORMAL REPORT VIEW
# ============================================================
def render_report_view(current_case_key):
    sc = SCENARIOS.get(current_case_key, SCENARIOS["FRAUD_SMURFING_FUNNEL"])
    is_fraud = sc["category"] == "FRAUD"
    pred = pred_map.get(sc["focal_node"], {})

    return html.Div(id="report-view-container", children=[
        dbc.Container([
            # Top Toolbar (no print)
            html.Div([
                html.Div([
                    html.Span("Viewing Official Case Dossier: ", style={"color": "#596473", "fontSize": "13px"}),
                    html.Strong(f"Case #{sc['index_num']} ({sc['title']})", style={"color": "#202A36", "fontSize": "13px"}),
                ]),
                html.Div([
                    dbc.Button("Back to Workspace", id={"type": "nav-btn", "target": "workspace"}, className="btn-fc-secondary me-2", size="sm"),
                    dbc.Button("Generate FIU-IND STR Form", id={"type": "open-modal-btn", "modal": "fiu"}, className="btn-fc-primary", size="sm"),
                ]),
            ], className="fc-surface p-3 mb-4 d-flex justify-content-between align-items-center no-print"),

            # Formal Dossier Document
            html.Div([
                # Document Header
                html.Div([
                    html.H2("FINANCIAL CRIME INVESTIGATION REPORT", style={"fontSize": "22px", "fontWeight": "800", "color": "#202A36", "letterSpacing": "0.02em", "marginBottom": "2px"}),
                    html.Div("CONFIDENTIAL • BANKING COMPLIANCE & LEGAL ENFORCEMENT MEMORANDUM", style={"fontSize": "11px", "color": "#697586", "letterSpacing": "0.05em", "marginBottom": "16px"}),
                    html.Hr(style={"borderColor": "#D8DDE3"}),
                ], className="text-center"),

                dbc.Row([
                    dbc.Col([
                        html.Div([html.Strong("Case Reference: "), f"FC-CASE-2026-{sc['index_num']:03d}"], style={"fontSize": "12px", "color": "#202A36"}),
                        html.Div([html.Strong("Typology Classification: "), sc["title"]], style={"fontSize": "12px", "color": "#596473"}),
                        html.Div([html.Strong("Target Focal Node: "), f"{sc['focal_node']}"], className="font-mono", style={"fontSize": "12px", "color": "#2457A6"}),
                    ], width=6),
                    dbc.Col([
                        html.Div([html.Strong("Assessed Risk Score: "), f"{sc['risk_pct']:.1f} / 100 ({sc['severity']})"], style={"fontSize": "12px", "color": "#B93832" if is_fraud else "#176B57"}),
                        html.Div([html.Strong("Financial Volume at Risk: "), sc["exposure_inr"]], className="font-mono", style={"fontSize": "12px", "color": "#202A36"}),
                        html.Div([html.Strong("Generated Timestamp: "), datetime.now().strftime("%d-%B-%Y %H:%M IST")], style={"fontSize": "12px", "color": "#596473"}),
                    ], width=6),
                ], className="mb-4 p-3", style={"backgroundColor": "#F9F8F5", "borderRadius": "4px", "border": "1px solid #D8DDE3"}),

                html.H4("1. Executive Summary & Observed Modus Operandi", style={"fontSize": "15px", "fontWeight": "700", "color": "#202A36", "marginBottom": "6px"}),
                html.P(sc["scheme_details"], style={"fontSize": "13px", "color": "#202A36", "lineHeight": "1.6", "marginBottom": "16px"}),

                html.H4("2. Why Traditional Transaction Rules Failed", style={"fontSize": "15px", "fontWeight": "700", "color": "#202A36", "marginBottom": "6px"}),
                html.P(sc["why_rules_fail"], style={"fontSize": "13px", "color": "#596473", "lineHeight": "1.6", "marginBottom": "16px"}),

                html.H4("3. Key Topological Graph Signals", style={"fontSize": "15px", "fontWeight": "700", "color": "#202A36", "marginBottom": "6px"}),
                html.Table([
                    html.Tr([html.Th("Metric Signal", style={"fontSize": "11px", "color": "#697586"}), html.Th("Recorded Value", style={"fontSize": "11px", "color": "#697586"})]),
                    *[html.Tr([html.Td(k, style={"fontSize": "12px", "color": "#202A36", "fontWeight": "500"}), html.Td(v, style={"fontSize": "12px", "color": "#2457A6", "fontWeight": "600"})]) for k, v in sc["key_signals"]],
                ], className="table table-sm mb-4", style={"borderColor": "#D8DDE3"}),

                html.H4("4. Algorithmic Narrative & Statutory Grounds", style={"fontSize": "15px", "fontWeight": "700", "color": "#202A36", "marginBottom": "6px"}),
                html.P(pred.get("narrative", "No narrative available."), style={"fontSize": "13px", "color": "#202A36", "lineHeight": "1.6", "backgroundColor": "#F9F8F5", "padding": "12px", "borderRadius": "4px", "marginBottom": "16px"}),

                html.H4("5. Recommended Legal & Investigative Actions", style={"fontSize": "15px", "fontWeight": "700", "color": "#202A36", "marginBottom": "6px"}),
                html.Ul([
                    html.Li("Execute administrative debit restriction under Section 35A of Banking Regulation Act 1949.", style={"fontSize": "13px", "color": "#596473", "marginBottom": "4px"}),
                    html.Li("Submit Suspicious Transaction Report (Form STR) to FIU-IND via FINnet 2.0 gateway.", style={"fontSize": "13px", "color": "#596473", "marginBottom": "4px"}),
                    html.Li("Request hardware telemetry and IP triangulation from Telecom Service Providers.", style={"fontSize": "13px", "color": "#596473"}),
                ], style={"paddingLeft": "18px", "marginBottom": "24px"}),

                html.Div([
                    html.Span("Disclaimer: ", style={"fontWeight": "600"}),
                    html.Span("This report compiles topological graph intelligence and algorithmic risk scoring. It constitutes investigative evidence and does not represent a final judicial conviction."),
                ], style={"fontSize": "11px", "color": "#697586", "borderTop": "1px solid #D8DDE3", "paddingTop": "12px"}),

            ], className="investigation-report-doc"),

        ], fluid=True, className="px-4 py-3"),
    ])


# ============================================================
# VIEW 6: ABOUT THE SYSTEM & METHODOLOGY
# ============================================================
def render_about_view():
    return html.Div(id="about-view-container", children=[
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("About FraudCircle", style={"fontSize": "28px", "fontWeight": "700", "color": "#202A36", "marginBottom": "6px"}),
                    html.P("Detecting coordinated financial crime rings through heterogeneous graph neural intelligence and explainable AI.",
                           style={"fontSize": "15px", "color": "#596473", "marginBottom": "24px"}),
                ], width=12),
            ]),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H3("1. The Core Innovation: Network vs. Transaction", style={"fontSize": "17px", "fontWeight": "700", "color": "#202A36", "marginBottom": "8px"}),
                        html.P("Traditional anti-money laundering (AML) evaluates single transactions. Criminal syndicates exploit this by fragmenting money across hundreds of mule accounts and shell merchants in amounts below mandatory PAN reporting thresholds (₹50,000 in India). FraudCircle models the entire financial ecosystem as a heterogeneous multigraph.",
                               style={"fontSize": "13.5px", "color": "#596473", "lineHeight": "1.6"}),
                    ], className="fc-surface p-4 mb-4"),

                    html.Div([
                        html.H3("2. Topological Graph Feature Engineering", style={"fontSize": "17px", "fontWeight": "700", "color": "#202A36", "marginBottom": "8px"}),
                        html.P(r"Every node is enriched with 13 exact topological properties computed in real time: in/out degree ratios, min/max flow pass-through, structuring band concentration (₹48k–₹49.9k), bipartite device/IP co-login degrees, directed cycle membership ($C_k \le 8$), PageRank centrality, and 48-hour velocity spikes.",
                               style={"fontSize": "13.5px", "color": "#596473", "lineHeight": "1.6"}),
                    ], className="fc-surface p-4 mb-4"),
                ], width=12, md=6),

                dbc.Col([
                    html.Div([
                        html.H3("3. Explainable AI (XAI) with SHAP", style={"fontSize": "17px", "fontWeight": "700", "color": "#202A36", "marginBottom": "8px"}),
                        html.P("Black-box AI models are rejected by banking regulators. FraudCircle incorporates TreeExplainer SHAP (SHapley Additive exPlanations) to decompose every risk score into exact positive and negative feature percentages, generating automated human-readable narratives for compliance officers.",
                               style={"fontSize": "13.5px", "color": "#596473", "lineHeight": "1.6"}),
                    ], className="fc-surface p-4 mb-4"),

                    html.Div([
                        html.H3("4. Regulatory Compliance Alignment", style={"fontSize": "17px", "fontWeight": "700", "color": "#202A36", "marginBottom": "8px"}),
                        html.P("Architected for Indian banking compliance under Section 12 of the Prevention of Money Laundering Act (PMLA) 2002, Reserve Bank of India (RBI) KYC Master Directions, and Digital Personal Data Protection (DPDP) Act 2023. Integrated directly with FIU-IND FINnet 2.0 reporting standards.",
                               style={"fontSize": "13.5px", "color": "#596473", "lineHeight": "1.6"}),
                    ], className="fc-surface p-4 mb-4"),
                ], width=12, md=6),
            ]),

        ], fluid=True, className="px-4 py-3"),
    ])


# ============================================================
# MASTER APPLICATION LAYOUT
# ============================================================
app.layout = html.Div([
    neo4j_modal,
    fiu_str_modal,

    # File Download Components
    dcc.Download(id="download-str-json"),
    dcc.Download(id="download-cypher"),

    # Persistent Top Navigation Bar
    render_navbar(),

    # Dynamic View Container
    html.Div(id="main-view-container", children=render_overview_view()),

    # Browser Location & URL Router
    dcc.Location(id="url", refresh=False),

    # Application State Stores (Session Persistence for Refresh / Back-Forward Safety)
    dcc.Store(id="current-view-store", data="overview", storage_type="session"),
    dcc.Store(id="current-case-store", data="FRAUD_SMURFING_FUNNEL", storage_type="session"),
    dcc.Store(id="workspace-tab-store", data="summary", storage_type="session"),
    dcc.Store(id="guide-active-store", data=False, storage_type="session"),
    dcc.Store(id="guide-step-store", data=1, storage_type="session"),
    dcc.Store(id="presentation-mode-store", data=False, storage_type="session"),
    dcc.Store(id="cases-filter-store", data="ALL", storage_type="session"),
    dcc.Store(id="selected-node-store", data="ACC_FRAUD_MASTER", storage_type="session"),
    dcc.Store(id="audit-log-store", data={}, storage_type="session"),
    dcc.Store(id="graph-layout-store", data="concentric", storage_type="session"),
    dcc.Store(id="graph-filter-store", data="ALL", storage_type="session"),
], style={"backgroundColor": "#F4F1EA", "minHeight": "100vh"})


# ============================================================
# CENTRAL VIEW ROUTER & NAVIGATION CALLBACKS
# ============================================================

# Nav item click & Brand Logo click -> update current-view-store
@callback(
    [Output("current-view-store", "data", allow_duplicate=True),
     Output("guide-active-store", "data", allow_duplicate=True),
     Output("presentation-mode-store", "data", allow_duplicate=True)],
    [
        Input("nav-brand-logo", "n_clicks"),
        Input("nav-btn-overview", "n_clicks"),
        Input("nav-btn-cases", "n_clicks"),
        Input("nav-btn-workspace", "n_clicks"),
        Input("nav-btn-benchmarks", "n_clicks"),
        Input("nav-btn-report", "n_clicks"),
        Input("nav-btn-about", "n_clicks"),
        Input({"type": "nav-btn", "target": dash.ALL}, "n_clicks"),
    ],
    prevent_initial_call=True
)
def handle_view_navigation(*args):
    # CRITICAL: Verify an actual user click occurred!
    # In Dash, when dynamic components mount into the DOM, ctx.triggered contains
    # prop_id with value=None or value=0.
    # If no trigger has value > 0, IT IS A DOM MOUNT EVENT, NOT A CLICK!
    triggered = ctx.triggered
    if not triggered or not triggered[0].get("value"):
        return no_update, no_update, no_update

    t = ctx.triggered_id
    if not t:
        return no_update, no_update, no_update

    # Top brand logo click -> return home/overview, reset walkthrough and presentation mode
    if t == "nav-brand-logo":
        return "overview", False, False
    elif t == "nav-btn-overview":
        return "overview", False, False
    elif t == "nav-btn-cases":
        return "cases", False, False
    elif t == "nav-btn-workspace":
        return "workspace", no_update, no_update
    elif t == "nav-btn-benchmarks":
        return "benchmarks", False, False
    elif t == "nav-btn-report":
        return "report", False, False
    elif t == "nav-btn-about":
        return "about", False, False
    elif isinstance(t, dict) and t.get("type") == "nav-btn":
        target = t.get("target", "overview")
        if target != "workspace":
            return target, False, False
        return target, no_update, no_update
    return no_update, no_update, no_update


# Update Navbar Active State Classes
@callback(
    [Output("nav-btn-overview", "className"),
     Output("nav-btn-cases", "className"),
     Output("nav-btn-workspace", "className"),
     Output("nav-btn-benchmarks", "className"),
     Output("nav-btn-report", "className"),
     Output("nav-btn-about", "className")],
    Input("current-view-store", "data")
)
def update_navbar_active_state(current_view):
    base = "fc-nav-item"
    return (
        f"{base} active" if current_view == "overview" else base,
        f"{base} active" if current_view == "cases" else base,
        f"{base} active" if current_view == "workspace" else base,
        f"{base} active" if current_view == "benchmarks" else base,
        f"{base} active" if current_view == "report" else base,
        f"{base} active" if current_view == "about" else base,
    )


# Open Case Workspace Button Clicked
@callback(
    [Output("current-case-store", "data"),
     Output("current-view-store", "data", allow_duplicate=True),
     Output("guide-active-store", "data", allow_duplicate=True),
     Output("guide-step-store", "data", allow_duplicate=True),
     Output("workspace-tab-store", "data", allow_duplicate=True),
     Output("presentation-mode-store", "data", allow_duplicate=True),
     Output("selected-node-store", "data", allow_duplicate=True)],
    Input({"type": "open-workspace-btn", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True
)
def handle_open_workspace_btn(n_clicks_list):
    triggered = ctx.triggered
    if not triggered or not triggered[0].get("value"):
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update

    t = ctx.triggered_id
    if t and isinstance(t, dict):
        selected_case = t.get("index", "FRAUD_SMURFING_FUNNEL")
        focal = SCENARIOS.get(selected_case, SCENARIOS["FRAUD_SMURFING_FUNNEL"])["focal_node"]
        return selected_case, "workspace", False, 1, "summary", False, focal
    return no_update, no_update, no_update, no_update, no_update, no_update, no_update


# Cases View Filter Buttons
@callback(
    Output("cases-filter-store", "data"),
    Input({"type": "cases-filter-btn", "filter": dash.ALL}, "n_clicks"),
    prevent_initial_call=True
)
def handle_cases_filter(clicks):
    triggered = ctx.triggered
    if not triggered or not triggered[0].get("value"):
        return no_update

    t = ctx.triggered_id
    if t and isinstance(t, dict):
        return t.get("filter", "ALL")
    return no_update


# Workspace Sub-Tabs Click
@callback(
    Output("workspace-tab-store", "data", allow_duplicate=True),
    Input({"type": "ws-tab-btn", "tab": dash.ALL}, "n_clicks"),
    prevent_initial_call=True
)
def handle_workspace_tabs(clicks):
    triggered = ctx.triggered
    if not triggered or not triggered[0].get("value"):
        return no_update

    t = ctx.triggered_id
    if t and isinstance(t, dict):
        return t.get("tab", "summary")
    return no_update


# Toggle Explain Guide & Presentation Mode
@callback(
    [Output("guide-active-store", "data", allow_duplicate=True),
     Output("presentation-mode-store", "data", allow_duplicate=True),
     Output("guide-step-store", "data", allow_duplicate=True)],
    Input({"type": "guide-mode-btn", "action": dash.ALL}, "n_clicks"),
    [State("guide-active-store", "data"),
     State("presentation-mode-store", "data")],
    prevent_initial_call=True
)
def handle_guide_mode_toggle(clicks, guide_active, pmode_active):
    triggered = ctx.triggered
    if not triggered or not triggered[0].get("value"):
        return no_update, no_update, no_update

    t = ctx.triggered_id
    if not t or not isinstance(t, dict):
        return no_update, no_update, no_update

    action = t.get("action")
    if action == "toggle-guide":
        new_guide = not guide_active
        return new_guide, False, 1 if new_guide else no_update
    elif action == "toggle-presentation":
        new_pmode = not pmode_active
        # When opening presentation mode, activate walkthrough at step 1
        return (True, True, 1) if new_pmode else (False, False, no_update)
    elif action == "close-guide":
        return False, False, 1
    elif action == "exit-presentation":
        return False, False, 1
    return no_update, no_update, no_update


# Step Next / Previous in Walkthrough
@callback(
    Output("guide-step-store", "data", allow_duplicate=True),
    Input({"type": "guide-step-btn", "direction": dash.ALL}, "n_clicks"),
    [State("guide-step-store", "data"),
     State("current-case-store", "data")],
    prevent_initial_call=True
)
def handle_guide_step_navigation(clicks, current_step, case_key):
    triggered = ctx.triggered
    if not triggered or not triggered[0].get("value"):
        return no_update

    t = ctx.triggered_id
    if not t or not isinstance(t, dict):
        return no_update

    sc = SCENARIOS.get(case_key, SCENARIOS["FRAUD_SMURFING_FUNNEL"])
    max_steps = len(sc.get("walkthrough", []))
    direction = t.get("direction")
    if direction == "next":
        return min(current_step + 1, max_steps)
    elif direction == "prev":
        return max(current_step - 1, 1)
    elif direction == "restart":
        return 1
    elif direction == "finish":
        return current_step
    return current_step


# Graph Layout and Filter Controls
@callback(
    [Output("graph-layout-store", "data"),
     Output("graph-filter-store", "data")],
    [Input({"type": "graph-layout-btn", "layout": dash.ALL}, "n_clicks"),
     Input({"type": "graph-filter-btn", "filter": dash.ALL}, "n_clicks")],
    [State("graph-layout-store", "data"),
     State("graph-filter-store", "data")],
    prevent_initial_call=True
)
def handle_graph_layout_filter(layout_clicks, filter_clicks, current_layout, current_filter):
    triggered = ctx.triggered
    if not triggered or not triggered[0].get("value"):
        return no_update, no_update

    t = ctx.triggered_id
    if not t or not isinstance(t, dict):
        return no_update, no_update

    btn_type = t.get("type")
    if btn_type == "graph-layout-btn":
        return t.get("layout", "concentric"), current_filter
    elif btn_type == "graph-filter-btn":
        return current_layout, t.get("filter", "ALL")
    return no_update, no_update


# Render the Main View Container
@callback(
    Output("main-view-container", "children"),
    [Input("current-view-store", "data"),
     Input("current-case-store", "data"),
     Input("workspace-tab-store", "data"),
     Input("guide-active-store", "data"),
     Input("guide-step-store", "data"),
     Input("presentation-mode-store", "data"),
     Input("cases-filter-store", "data"),
     Input("graph-layout-store", "data"),
     Input("graph-filter-store", "data"),
     Input("selected-node-store", "data"),
     Input("audit-log-store", "data")]
)
def render_main_view(view, case_key, ws_tab, guide_active, guide_step, pmode, cases_filter, g_layout, g_filter, selected_node=None, audit_logs=None):
    if view == "cases":
        return render_cases_view(filter_val=cases_filter)
    elif view == "workspace":
        return render_workspace_view(
            current_case_key=case_key,
            active_tab=ws_tab,
            show_guide=guide_active,
            guide_step=guide_step,
            presentation_mode=pmode,
            graph_layout=g_layout or "concentric",
            graph_filter=g_filter or "ALL",
            selected_node=selected_node,
            audit_logs=audit_logs
        )
    elif view == "benchmarks":
        return render_benchmarks_view()
    elif view == "report":
        return render_report_view(case_key)
    elif view == "about":
        return render_about_view()
    else:
        return render_overview_view()


# Graph Tap Node -> update Selected Node Inspector and Store
@callback(
    [Output("workspace-entity-inspector", "children"),
     Output("selected-node-store", "data", allow_duplicate=True)],
    Input("graph-canvas", "tapNodeData"),
    State("current-case-store", "data"),
    prevent_initial_call=True
)
def handle_graph_node_click(tap_data, case_key):
    if not tap_data or "id" not in tap_data:
        return no_update, no_update

    nid = tap_data["id"]
    content = build_entity_inspector_content(nid, case_key)
    return content, nid


# Human in the Loop (HITL) Actions (Persisted per Case)
@callback(
    Output("audit-log-store", "data", allow_duplicate=True),
    Input({"type": "hitl-action-btn", "action": dash.ALL}, "n_clicks"),
    [State("current-case-store", "data"),
     State("audit-log-store", "data")],
    prevent_initial_call=True
)
def handle_hitl_action(clicks, case_key, log_dict):
    triggered = ctx.triggered
    if not triggered or not triggered[0].get("value"):
        return no_update

    t = ctx.triggered_id
    if not t or not isinstance(t, dict):
        return no_update
    action = t.get("action")
    sc = SCENARIOS.get(case_key, SCENARIOS["FRAUD_SMURFING_FUNNEL"])
    focal = sc["focal_node"]
    now = datetime.now().strftime("[%H:%M:%S]")

    if action == "quarantine":
        entry = {
            "time": now,
            "badge": "QUARANTINE ORDER",
            "color": "#B93832",
            "text": f"Section 35A administrative hold applied to {focal} and connected counterparties. Settlement rails restricted."
        }
    elif action == "whitelist":
        entry = {
            "time": now,
            "badge": "EXEMPTION ISSUED",
            "color": "#176B57",
            "text": f"Entity {focal} verified as legitimate business. False positive suppressed; audit exception certificate issued."
        }
    else:
        return no_update

    if not isinstance(log_dict, dict):
        log_dict = {}
    case_logs = list(log_dict.get(case_key, []))
    case_logs.insert(0, entry)
    log_dict[case_key] = case_logs
    return log_dict


# FIU Modal Open / Close
@callback(
    Output("fiu-str-modal", "is_open"),
    [Input({"type": "open-modal-btn", "modal": "fiu"}, "n_clicks"),
     Input("btn-close-fiu-modal", "n_clicks")],
    State("fiu-str-modal", "is_open"),
    prevent_initial_call=True
)
def handle_fiu_modal(open_clicks, close_clicks, is_open):
    triggered = ctx.triggered
    if not triggered or not triggered[0].get("value"):
        return no_update

    t = ctx.triggered_id
    if not t:
        return no_update
    if t == "btn-close-fiu-modal":
        return False
    return True


# Update FIU STR Modal Content
@callback(
    Output("fiu-modal-content", "children"),
    Input("fiu-str-modal", "is_open"),
    State("current-case-store", "data"),
    prevent_initial_call=True
)
def update_fiu_modal_content(is_open, case_key):
    if not is_open:
        return no_update

    sc = SCENARIOS.get(case_key, SCENARIOS["FRAUD_SMURFING_FUNNEL"])
    focal = sc["focal_node"]
    pred = pred_map.get(focal, {})

    return html.Div([
        html.Div([
            html.H4("FINANCIAL INTELLIGENCE UNIT - INDIA", style={"color": "#202A36", "fontWeight": "800", "fontSize": "17px", "marginBottom": "2px"}),
            html.Div("SUSPICIOUS TRANSACTION REPORT (STR) • SECTION 12 PMLA 2002", style={"fontSize": "11px", "color": "#697586", "letterSpacing": "0.06em", "marginBottom": "14px"}),
        ], className="text-center pb-2 border-bottom mb-3", style={"borderColor": "#D8DDE3"}),

        dbc.Row([
            dbc.Col([
                html.Div([html.Strong("Filing Reference: "), f"FIU-STR-2026-{abs(hash(focal)) % 100000:05d}"], style={"fontSize": "12px", "color": "#202A36"}),
                html.Div([html.Strong("Reporting Institution: "), "FraudCircle Banking Platform"], style={"fontSize": "12px", "color": "#596473"}),
                html.Div([html.Strong("Date & Time: "), datetime.now().strftime("%d-%B-%Y %H:%M IST")], style={"fontSize": "12px", "color": "#596473"}),
            ], width=6),
            dbc.Col([
                html.Div([html.Strong("Subject Primary Entity: "), focal], className="font-mono", style={"fontSize": "12px", "color": "#2457A6"}),
                html.Div([html.Strong("Suspicion Score: "), f"{sc['risk_pct']:.1f} / 100"], style={"fontSize": "12px", "color": "#B93832" if sc['category']=='FRAUD' else '#176B57'}),
                html.Div([html.Strong("Typology: "), sc["title"]], style={"fontSize": "12px", "color": "#596473"}),
            ], width=6),
        ], className="p-3 mb-3", style={"backgroundColor": "#FFFFFF", "borderRadius": "4px", "border": "1px solid #D8DDE3"}),

        html.H5("1. Grounds of Suspicion & Graph Evidence", style={"fontSize": "13px", "fontWeight": "700", "color": "#202A36", "marginBottom": "6px"}),
        html.P(pred.get("narrative", "Substantial graph evidence indicates coordinated structuring or ring behavior."),
               style={"fontSize": "12.5px", "color": "#202A36", "backgroundColor": "#FFFFFF", "padding": "12px", "borderRadius": "4px", "border": "1px solid #D8DDE3", "lineHeight": "1.6"}),

        html.H5("2. Statutory Enforcement Requests", style={"fontSize": "13px", "fontWeight": "700", "color": "#202A36", "marginBottom": "6px"}),
        html.Ul([
            html.Li("Provisional freezing order under Section 5(1) Prevention of Money Laundering Act.", style={"fontSize": "12px", "color": "#596473"}),
            html.Li("Request IP / Device MAC correlation data from Telecom Service Provider (TSP).", style={"fontSize": "12px", "color": "#596473"}),
            html.Li("Flag connected recipient accounts across Inter-Bank settlement rails.", style={"fontSize": "12px", "color": "#596473"}),
        ], style={"paddingLeft": "18px"}),
    ], className="fc-surface p-4")


# Download FIU STR JSON File
@callback(
    Output("download-str-json", "data"),
    Input("btn-download-str-json", "n_clicks"),
    State("current-case-store", "data"),
    prevent_initial_call=True
)
def download_str_file(n_clicks, case_key):
    if not n_clicks:
        return no_update
    sc = SCENARIOS.get(case_key, SCENARIOS["FRAUD_SMURFING_FUNNEL"])
    focal = sc["focal_node"]
    pred = pred_map.get(focal, {})
    report_dict = {
        "filing_institution": "FraudCircle Compliance Platform",
        "case_id": f"FC-CASE-2026-{sc['index_num']:03d}",
        "filing_reference": f"FIU-STR-2026-{abs(hash(focal)) % 100000:05d}",
        "timestamp": datetime.now().isoformat(),
        "regulatory_statute": "Section 12 Prevention of Money Laundering Act (PMLA) 2002",
        "focal_subject": focal,
        "typology": sc["title"],
        "exposure_inr": sc["exposure_inr"],
        "suspicion_score": sc["risk_pct"],
        "severity": sc["severity"],
        "key_signals": dict(sc["key_signals"]),
        "algorithmic_narrative": pred.get("narrative", ""),
    }
    return dict(content=json.dumps(report_dict, indent=2), filename=f"FIU_STR_{focal}.json")


# Neo4j Modal Open / Close
@callback(
    Output("neo4j-modal", "is_open"),
    [Input({"type": "open-modal-btn", "modal": "neo4j"}, "n_clicks"),
     Input("btn-close-neo4j-modal", "n_clicks")],
    State("neo4j-modal", "is_open"),
    prevent_initial_call=True
)
def handle_neo4j_modal(open_clicks, close_clicks, is_open):
    triggered = ctx.triggered
    if not triggered or not triggered[0].get("value"):
        return no_update

    t = ctx.triggered_id
    if not t:
        return no_update
    if t == "btn-close-neo4j-modal":
        return False
    return True


# Download Neo4j Cypher Script
@callback(
    Output("download-cypher", "data"),
    Input("btn-export-cypher-file", "n_clicks"),
    State("current-case-store", "data"),
    prevent_initial_call=True
)
def download_cypher_file(n_clicks, case_key):
    if not n_clicks:
        return no_update
    cypher_text = generate_cypher_for_scenario(nodes_df, edges_df, case_key)
    return dict(content=cypher_text, filename=f"FraudCircle_{case_key}.cypher")


# Push to Neo4j Action
@callback(
    [Output("neo4j-status-alert", "children"),
     Output("neo4j-status-alert", "color"),
     Output("neo4j-status-alert", "is_open")],
    Input("btn-sync-neo4j-action", "n_clicks"),
    [State("neo4j-uri", "value"),
     State("neo4j-user", "value"),
     State("neo4j-pass", "value"),
     State("current-case-store", "data"),
     State("neo4j-scope-dropdown", "value")],
    prevent_initial_call=True
)
def execute_neo4j_sync(n_clicks, uri, user, pwd, case_key, scope):
    if not n_clicks:
        return no_update, no_update, False
    target = case_key if scope == "current" else None
    try:
        res = sync_to_neo4j(uri=uri, username=user, password=pwd, nodes_df=nodes_df, edges_df=edges_df, scenario_tag=target)
        if res.get("success"):
            return f"Synchronized successfully: {res.get('nodes', 0)} nodes, {res.get('edges', 0)} edges.", "success", True
        else:
            err = res.get("error", "Connection error")
            return f"Neo4j Connection Note: {err} (Offline mode remains active)", "warning", True
    except Exception as e:
        return f"Neo4j Note: {str(e)} (Offline mode active)", "warning", True


# ============================================================
# SERVER STARTUP
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  FRAUDCIRCLE | Enterprise Financial Intelligence")
    print("  Design Philosophy: Editorial Light (Palantir / Linear / Stripe)")
    print("  URL: http://0.0.0.0:8050")
    print("="*60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=8050)


# ============================================================
# TWO-WAY URL SYNCHRONIZATION & BROWSER ROUTING
# ============================================================
VIEW_PATH_MAP = {
    "overview": "/",
    "cases": "/cases",
    "workspace": "/workspace",
    "benchmarks": "/benchmarks",
    "report": "/report",
    "about": "/about"
}
PATH_VIEW_MAP = {
    "/": "overview",
    "/overview": "overview",
    "/cases": "cases",
    "/workspace": "workspace",
    "/benchmarks": "benchmarks",
    "/report": "report",
    "/about": "about"
}

@callback(
    [Output("url", "pathname"),
     Output("url", "search")],
    [Input("current-view-store", "data"),
     Input("current-case-store", "data")],
    [State("url", "pathname"),
     State("url", "search")],
    prevent_initial_call=True
)
def sync_store_to_url(current_view, current_case, current_path, current_search):
    target_path = VIEW_PATH_MAP.get(current_view, "/")
    target_search = f"?case={current_case}" if current_view in ["workspace", "report"] else ""
    if current_path == target_path and current_search == target_search:
        return no_update, no_update
    return target_path, target_search

@callback(
    [Output("current-view-store", "data", allow_duplicate=True),
     Output("current-case-store", "data", allow_duplicate=True)],
    [Input("url", "pathname"),
     Input("url", "search")],
    [State("current-view-store", "data"),
     State("current-case-store", "data")],
    prevent_initial_call=True
)
def sync_url_to_store(pathname, search, current_view, current_case):
    if not pathname:
        return no_update, no_update

    target_view = PATH_VIEW_MAP.get(pathname)
    if not target_view:
        return no_update, no_update

    target_case = current_case
    if search and "case=" in search:
        import urllib.parse
        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(search).query)
        param_case = parsed.get("case", [None])[0]
        if param_case and param_case in SCENARIOS:
            target_case = param_case

    if target_view == current_view and target_case == current_case:
        return no_update, no_update
    return target_view, target_case
