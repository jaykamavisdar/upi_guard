# app.py — UPIGuard 2.0 Streamlit Demo
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="UPIGuard",
    page_icon="🛡️",
    layout="centered"
)

# ── Load models (cached so they load only once) ──────────────────
@st.cache_resource
def load_models():
    xgb = XGBClassifier()
    xgb.load_model('upiguard_models/xgboost_model.json')
    cnn     = load_model('upiguard_models/cnn_model.keras')
    scaler  = joblib.load('upiguard_models/scaler.pkl')
    f_cols  = joblib.load('upiguard_models/feature_cols.pkl')
    samples = pd.read_csv('upiguard_models/demo_samples.csv')
    return xgb, cnn, scaler, f_cols, samples

xgb_model, cnn_model, scaler, feature_cols, demo_df = load_models()

# ── Scoring functions (same as Kaggle) ───────────────────────────
def get_behavioral_score(row):
    score  = 0.4 * min(abs(row.get('amt_deviation', 0)), 3) / 3
    score += 0.3 * row.get('is_night', 0)
    score += 0.3 * row.get('is_high_amt', 0)
    return float(np.clip(score, 0, 1))

def get_rule_score(row):
    r1 = float(row.get('is_high_amt', 0) == 1 and row.get('is_night', 0) == 1)
    r2 = float(row.get('card1_tx_count', 0) > 10000)
    r3 = float(abs(row.get('amt_deviation', 0)) > 2.5)
    return float(np.mean([r1, r2, r3]))

def hybrid_score(p1, p2, p3, p4):
    return float(np.clip((0.55*p1 + 0.30*p2 + 0.10*p3 + 0.05*p4) * 100, 0, 100))

def decision_label(score):
    if score < 30:  return "ALLOW",      "🟢", "success"
    elif score < 70: return "OTP VERIFY", "🟡", "warning"
    else:            return "BLOCK",      "🔴", "error"

# ── UI ───────────────────────────────────────────────────────────
st.markdown("## 🛡️ UPIGuard 2.0")
st.markdown("*Hybrid AI Fraud Detection — Real-time UPI Transaction Analyzer*")
st.divider()

# ── Scenario selector ────────────────────────────────────────────
st.markdown("### Simulate a UPI Transaction")

col1, col2 = st.columns(2)
with col1:
    upi_id = st.text_input("Pay to (UPI ID)", value="merchant@okhdfc")
    amount = st.number_input("Amount (₹)", min_value=1, max_value=500000, value=4500)

with col2:
    scenario = st.selectbox("Transaction scenario", [
        "Normal purchase",
        "High-value transfer",
        "Suspicious pattern",
        "Late night transfer"
    ])
    device = st.selectbox("Device", [
        "My phone (trusted)",
        "New device",
        "Unknown browser"
    ])

# Map scenario → demo sample index
scenario_map = {
    "Normal purchase"      : demo_df[demo_df['isFraud']==0].index[0],
    "High-value transfer"  : demo_df[demo_df['risk_score']>40].index[0],
    "Suspicious pattern"   : demo_df[demo_df['isFraud']==1].index[0],
    "Late night transfer"  : demo_df[demo_df['is_night']==1].index[0]
    if 'is_night' in demo_df.columns else demo_df.index[2],
}

# ── Analyze button ───────────────────────────────────────────────
if st.button("🔍 Analyze with UPIGuard", use_container_width=True, type="primary"):

    with st.spinner("Running hybrid AI analysis..."):
        time.sleep(1.2)   # simulate real processing feel

        # Get the demo row matching scenario
        idx = scenario_map[scenario]
        row = demo_df.loc[idx]
        X_row = pd.DataFrame([row[feature_cols]], columns=feature_cols)

        # Get all 4 scores
        P1 = float(xgb_model.predict_proba(X_row)[:, 1][0])

        X_scaled = scaler.transform(X_row)
        X_cnn    = X_scaled.reshape(1, X_scaled.shape[1], 1)
        P2 = float(cnn_model.predict(X_cnn, verbose=0)[0][0])

        P3 = get_behavioral_score(row)
        P4 = get_rule_score(row)

        risk = hybrid_score(P1, P2, P3, P4)
        dec, icon, dtype = decision_label(risk)

    st.divider()
    st.markdown("### Risk Analysis Result")

    # ── Score display ────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Score", f"{risk:.1f} / 100")
    c2.metric("Decision",   f"{icon} {dec}")
    c3.metric("Actual Label", "FRAUD" if row['isFraud']==1 else "LEGIT")

    # ── Risk bar ─────────────────────────────────────────────────
    bar_color = "#2ecc71" if risk < 30 else "#f39c12" if risk < 70 else "#e74c3c"
    st.markdown(f"""
    <div style="background:#eee;border-radius:8px;height:16px;margin:8px 0 16px">
      <div style="width:{risk:.0f}%;background:{bar_color};height:16px;
                  border-radius:8px;transition:width 1s ease"></div>
    </div>""", unsafe_allow_html=True)

    # ── Decision alert ────────────────────────────────────────────
    if dtype == "success":
        st.success(f"Transaction ALLOWED — Low fraud risk ({risk:.1f}/100)")
    elif dtype == "warning":
        st.warning(f"OTP Verification required — Medium risk ({risk:.1f}/100)")
    else:
        st.error(f"Transaction BLOCKED — High fraud risk ({risk:.1f}/100)")

    # ── Component scores ──────────────────────────────────────────
    st.divider()
    st.markdown("### Hybrid Engine Breakdown")
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("XGBoost P₁", f"{P1*100:.1f}",  delta="55% weight")
    cc2.metric("CNN P₂",     f"{P2*100:.1f}",  delta="30% weight")
    cc3.metric("Behavioral P₃", f"{P3*100:.1f}", delta="10% weight")
    cc4.metric("Rules P₄",  f"{P4*100:.1f}",  delta="5% weight")

    # ── Explanation ───────────────────────────────────────────────
    st.divider()
    st.markdown("### Why was this flagged?")
    st.markdown(f"""
    | Factor | Value | Impact |
    |--------|-------|--------|
    | Transaction Amount | ₹{amount:,} | {'High — above 95th percentile' if amount > 20000 else 'Normal range'} |
    | Device | {device} | {'New device increases risk' if 'New' in device or 'Unknown' in device else 'Trusted device reduces risk'} |
    | Time | {'Late night' if 'night' in scenario.lower() else 'Daytime'} | {'Night transactions are higher risk' if 'night' in scenario.lower() else 'Daytime is normal'} |
    | XGBoost confidence | {P1*100:.1f}% | Primary fraud signal |
    """)

    # ── Transaction log ───────────────────────────────────────────
    st.divider()
    st.markdown("### Transaction Log")
    st.dataframe(pd.DataFrame([{
        'UPI ID'     : upi_id,
        'Amount'     : f"₹{amount:,}",
        'Risk Score' : f"{risk:.1f}",
        'Decision'   : dec,
        'XGBoost'    : f"{P1:.4f}",
        'CNN'        : f"{P2:.4f}",
        'Behavioral' : f"{P3:.4f}",
        'Rules'      : f"{P4:.4f}",
    }]), use_container_width=True)
# ```

# ---

# ## Step 3 — Deploy free on Streamlit Cloud

# Create `requirements.txt` in the same folder:
# ```
# streamlit
# xgboost
# tensorflow
# scikit-learn
# pandas
# numpy
# joblib
# ```

# Then:
# 1. Push both files + `upiguard_models/` folder to a **GitHub repo**
# 2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → sign in with GitHub → New app → select your repo → `app.py`
# 3. Click Deploy — live URL in 2 minutes, free forever

# Your folder structure should look like:
# ```
# upiguard-demo/
# ├── app.py
# ├── requirements.txt
# └── upiguard_models/
#     ├── xgboost_model.json
#     ├── cnn_model.keras   
#     ├── scaler.pkl
#     ├── feature_cols.pkl
#     └── demo_samples.csv