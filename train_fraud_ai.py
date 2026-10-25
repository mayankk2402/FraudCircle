import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from xgboost import XGBClassifier
import shap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
print("[*] Loading feature data...")
feat_df = pd.read_csv(os.path.join(DATA_DIR, "Account_Features_FraudCircle.csv"))
print(f"    Total samples: {len(feat_df)}")
print(f"    Fraud: {feat_df['ground_truth_fraud'].sum()}, Legit: {(feat_df['ground_truth_fraud']==0).sum()}")

FEATURE_COLS = [
    "in_degree", "out_degree", "total_in_amount", "total_out_amount",
    "flow_pass_through_ratio", "structuring_tx_ratio",
    "shared_device_degree", "shared_ip_degree", "in_cycle_flag",
    "pagerank_score", "local_clustering_coef", "velocity_spike_24h",
    "unverified_counterparty_ratio", "account_age_days", "is_verified_business"
]

X = feat_df[FEATURE_COLS].fillna(0).values
y = feat_df["ground_truth_fraud"].values

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================
n_fraud = int(y.sum())
n_legit = int((y == 0).sum())
spw = n_legit / max(n_fraud, 1)
print(f"\n[*] scale_pos_weight = {spw:.2f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"    Train: {len(X_train)} | Test: {len(X_test)}")

# ============================================================
# TRAIN XGBClassifier
# ============================================================
print("\n[*] Training XGBClassifier...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=spw,
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42,
    n_jobs=1,
    verbosity=0
)
model.fit(X_train, y_train)

# Save model
model.save_model(os.path.join(MODELS_DIR, "fraud_xgb_model.json"))
print("[OK] Model saved to models/fraud_xgb_model.json")

# ============================================================
# EVALUATION
# ============================================================
print("\n" + "="*60)
print("  VALIDATION REPORT")
print("="*60)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  ROC-AUC:   {auc:.4f}")
print("\n" + classification_report(y_test, y_pred, target_names=["Legitimate", "Fraudulent"]))

# ============================================================
# SHAP EXPLAINABILITY
# ============================================================
print("[*] Computing SHAP values for ALL nodes...")
explainer = shap.TreeExplainer(model)
X_all = feat_df[FEATURE_COLS].fillna(0).values
shap_values = explainer.shap_values(X_all)

y_all_prob = model.predict_proba(X_all)[:, 1]
risk_scores = np.clip(y_all_prob * 100, 0, 100).round(2)

# ============================================================
# GENERATE NARRATIVES & PREDICTIONS
# ============================================================
print("[*] Generating XAI narratives...")

def generate_narrative(row_idx, risk_score, shap_vals, feature_names, node_data):
    """Generate a plain-English investigator narrative."""
    sv_pairs = list(zip(feature_names, shap_vals))
    sv_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    top4 = sv_pairs[:4]

    node_id = node_data["node_id"]
    name = node_data.get("name", node_id)
    scenario = node_data.get("scenario_tag", "UNKNOWN")
    is_fraud = risk_score > 50

    parts = []
    for feat, sv in top4:
        direction = "+" if sv > 0 else "-"
        impact_pct = abs(sv) * 100 / (abs(shap_vals).sum() + 1e-9) * 100
        impact_pct = min(impact_pct, 99)
        val = node_data.get(feat, "N/A")
        if sv > 0:
            parts.append(f"{feat}={val} (+{impact_pct:.0f}% risk)")
        else:
            parts.append(f"{feat}={val} (-{impact_pct:.0f}% risk)")

    if is_fraud:
        narrative = (f"FLAGGED: Entity '{name}' ({node_id}) flagged with Risk Score "
                     f"{risk_score:.0f}/100. Key risk drivers: {'; '.join(parts)}. "
                     f"Recommend immediate investigation under RBI AML guidelines.")
    else:
        narrative = (f"CLEARED: Entity '{name}' ({node_id}) assessed with Risk Score "
                     f"{risk_score:.0f}/100. Mitigating factors: {'; '.join(parts)}. "
                     f"No suspicious activity detected. Compliant with DPDP Act.")
    return narrative


predictions = []
for idx in range(len(feat_df)):
    row = feat_df.iloc[idx]
    sv = shap_values[idx]
    rs = risk_scores[idx]

    top4_indices = np.argsort(np.abs(sv))[::-1][:4]
    top4_features = []
    for fi in top4_indices:
        fname = FEATURE_COLS[fi]
        top4_features.append({
            "feature": fname,
            "shap_value": round(float(sv[fi]), 6),
            "feature_value": float(row[fname]) if pd.notna(row[fname]) else 0,
            "impact_direction": "positive_risk" if sv[fi] > 0 else "negative_risk"
        })

    row_dict = row.to_dict()
    for k, v in row_dict.items():
        if isinstance(v, (np.integer,)):
            row_dict[k] = int(v)
        elif isinstance(v, (np.floating,)):
            row_dict[k] = float(v)

    narrative = generate_narrative(idx, rs, sv, FEATURE_COLS, row_dict)

    predictions.append({
        "node_id": row["node_id"],
        "name": row.get("name", ""),
        "node_type": row["node_type"],
        "scenario_tag": row["scenario_tag"],
        "ground_truth_fraud": int(row["ground_truth_fraud"]),
        "risk_score": float(rs),
        "predicted_fraud": 1 if rs > 50 else 0,
        "top_shap_features": top4_features,
        "all_shap_values": {FEATURE_COLS[i]: round(float(sv[i]), 6) for i in range(len(FEATURE_COLS))},
        "narrative": narrative
    })

# Save JSON
json_path = os.path.join(DATA_DIR, "Model_Predictions_XAI.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=2, ensure_ascii=False)

# Save CSV
pred_df = pd.DataFrame([{
    "node_id": p["node_id"],
    "name": p["name"],
    "node_type": p["node_type"],
    "scenario_tag": p["scenario_tag"],
    "ground_truth_fraud": p["ground_truth_fraud"],
    "risk_score": p["risk_score"],
    "predicted_fraud": p["predicted_fraud"],
    "narrative": p["narrative"],
    "top_feature_1": p["top_shap_features"][0]["feature"] if len(p["top_shap_features"]) > 0 else "",
    "top_feature_1_shap": p["top_shap_features"][0]["shap_value"] if len(p["top_shap_features"]) > 0 else 0,
    "top_feature_2": p["top_shap_features"][1]["feature"] if len(p["top_shap_features"]) > 1 else "",
    "top_feature_2_shap": p["top_shap_features"][1]["shap_value"] if len(p["top_shap_features"]) > 1 else 0,
} for p in predictions])
csv_path = os.path.join(DATA_DIR, "Model_Predictions_XAI.csv")
pred_df.to_csv(csv_path, index=False)

print(f"\n[OK] Predictions saved: {json_path}")
print(f"[OK] Predictions CSV: {csv_path}")

# ============================================================
# EDGE-CASE VERIFICATION AUDIT
# ============================================================
print("\n" + "="*60)
print("  EDGE-CASE VERIFICATION AUDIT")
print("="*60)

audit_scenarios = {
    "FRAUD_SMURFING_FUNNEL": {"expected": "FRAUD", "threshold_min": 80},
    "FRAUD_SYBIL_DEVICE_RING": {"expected": "FRAUD", "threshold_min": 80},
    "FRAUD_CIRCULAR_LOOP": {"expected": "FRAUD", "threshold_min": 80},
    "FRAUD_BUST_OUT": {"expected": "FRAUD", "threshold_min": 80},
    "LEGIT_CORP_PAYROLL": {"expected": "LEGIT", "threshold_max": 25},
    "LEGIT_UTILITY_HUB": {"expected": "LEGIT", "threshold_max": 25},
    "LEGIT_WEDDING_SPEND": {"expected": "LEGIT", "threshold_max": 25},
}

all_passed = True
for scenario, criteria in audit_scenarios.items():
    scenario_preds = [p for p in predictions if p["scenario_tag"] == scenario]
    if not scenario_preds:
        print(f"  [WARN] No predictions for {scenario}")
        all_passed = False
        continue

    # For fraud scenarios, check key actors; for legit, check the hub entity
    if criteria["expected"] == "FRAUD":
        # Get average risk of fraud-labeled nodes
        fraud_nodes = [p for p in scenario_preds if p["ground_truth_fraud"] == 1]
        if fraud_nodes:
            avg_risk = np.mean([p["risk_score"] for p in fraud_nodes])
            max_risk = np.max([p["risk_score"] for p in fraud_nodes])
            status = "PASS" if avg_risk >= criteria["threshold_min"] else "FAIL"
            if status == "FAIL":
                all_passed = False
            print(f"  {status:4s} | {scenario:30s} | Avg Risk: {avg_risk:6.1f} | Max: {max_risk:6.1f} | Expected >= {criteria['threshold_min']}")
    else:
        # Get hub / key legitimate entity
        key_entities = {
            "LEGIT_CORP_PAYROLL": "ACC_CORP_INFOSYS_PAYROLL",
            "LEGIT_UTILITY_HUB": "MERCH_TATA_POWER",
            "LEGIT_WEDDING_SPEND": "ACC_WEDDING_FAMILY",
        }
        key_id = key_entities.get(scenario, "")
        key_preds = [p for p in scenario_preds if p["node_id"] == key_id]
        if key_preds:
            risk = key_preds[0]["risk_score"]
            status = "PASS" if risk <= criteria["threshold_max"] else "FAIL"
            if status == "FAIL":
                all_passed = False
            print(f"  {status:4s} | {scenario:30s} | Key Entity Risk: {risk:6.1f} | Expected <= {criteria['threshold_max']}")
        else:
            # Fallback: average of legit nodes
            legit_nodes = [p for p in scenario_preds if p["ground_truth_fraud"] == 0]
            if legit_nodes:
                avg_risk = np.mean([p["risk_score"] for p in legit_nodes])
                status = "PASS" if avg_risk <= criteria["threshold_max"] else "FAIL"
                if status == "FAIL":
                    all_passed = False
                print(f"  {status:4s} | {scenario:30s} | Avg Risk: {avg_risk:6.1f} | Expected <= {criteria['threshold_max']}")

print("\n" + "="*60)
if all_passed:
    print("  ALL 7 EDGE-CASE SCENARIOS PASSED!")
else:
    print("  SOME SCENARIOS DID NOT MEET THRESHOLDS (see above)")
print("="*60)
print("\n[DONE] Training pipeline complete.")
