import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

st.set_page_config(
    page_title="Royal Enfield — Telemetry Diagnostics", 
    page_icon="🏍️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Royal Enfield Red Branding)
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; color: #1E1E1E; }
    
    /* Top Navigation Bar */
    .top-bar { 
        background: #F8F9FA; 
        border-bottom: 3px solid #8B0000; 
        padding: 16px 24px; 
        border-radius: 8px; 
        margin-bottom: 24px; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    h1 { color: #8B0000 !important; font-family: 'Trebuchet MS', sans-serif; font-weight: 800; margin-bottom: 0px !important; }
    
    /* Metric Cards Styling */
    div[data-testid="stMetric"] { 
        background: #F8F9FA; 
        border: 1px solid #E0E0E0; 
        border-left: 5px solid #8B0000; 
        padding: 16px 20px; 
        border-radius: 8px; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetric"]:nth-child(2) { border-left-color: #D32F2F; }
    div[data-testid="stMetric"]:nth-child(3) { border-left-color: #E65100; }
    div[data-testid="stMetric"]:nth-child(4) { border-left-color: #2E7D32; }
    
    /* Primary Buttons */
    div.stButton > button[kind="primary"] { 
        background: linear-gradient(135deg, #8B0000 0%, #A00C00 100%) !important; 
        color: #FFFFFF !important; 
        border: none !important; 
        font-weight: 700; 
    }
    
    /* Target Table Headers */
    div[data-testid="stDataFrame"] th {
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        color: #8B0000 !important;
        background-color: #F1F3F5 !important;
    }
    
    /* Risk Definition Box Styles */
    .def-box {
        background-color: #F8F9FA;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 16px;
        height: 100%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .def-high { border-left: 5px solid #D32F2F; }
    .def-med { border-left: 5px solid #E65100; }
    .def-low { border-left: 5px solid #2E7D32; }
    .def-title { font-weight: 800; font-size: 1.05rem; margin-bottom: 6px; }
    .def-text { font-size: 0.88rem; color: #555555; line-height: 1.4; }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_all_artifacts():
    """Loads PKL or JSON model alongside metadata JSON report."""
    pkl_path = os.path.join(BASE_DIR, "best_oxygen_sensor_model_V1.pkl")
    json_path = os.path.join(BASE_DIR, "o2_sensor_xgboost.json")
    meta_path = os.path.join(BASE_DIR, "model_metadata_report_V1.json")
    
    # Fallback path checks
    if not os.path.exists(pkl_path):
        pkl_path = os.path.join(BASE_DIR, "best_oxygen_sensor_model.pkl")
    if not os.path.exists(json_path):
        json_path = os.path.join(BASE_DIR, "o2_with_new_labeeled_data.json")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(BASE_DIR, "model_metadata_report.json")

    model, metadata, model_type = None, {}, "Unknown"

    # 1. Try loading PKL model via joblib
    if os.path.exists(pkl_path):
        try:
            model = joblib.load(pkl_path)
            model_type = "PKL (LightGBM/Joblib)"
        except Exception as e:
            st.warning(f"⚠️ PKL file present but failed to load: {e}")

    # 2. Try loading native JSON XGBoost model if PKL loading failed
    if model is None and os.path.exists(json_path):
        try:
            model = xgb.XGBClassifier(enable_categorical=True)
            model.load_model(json_path)
            model_type = "JSON (XGBoost Native)"
        except Exception as e:
            st.warning(f"⚠️ JSON file present but failed to load: {e}")

    # 3. Load Metadata JSON
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            st.warning(f"⚠️ Metadata JSON load error: {e}")

    return model, metadata, model_type

model, metadata, model_type = load_all_artifacts()

# Top Header Navigation Bar
st.markdown("""
<div class="top-bar">
    <div>
        <h1 style="margin: 0;">🏍️ ROYAL ENFIELD TELEMETRY DIAGNOSTICS</h1>
        <div style="color: #666666; font-size: 0.88rem; margin-top: 4px; font-weight: 500;">O2 Sensor Predictive Intelligence & Dynamic Risk Calibration</div>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("❌ Diagnostic Engine Offline: Model file missing (`best_oxygen_sensor_model_V1.pkl` or `o2_sensor_xgboost.json`).")
else:
    # Sidebar Threshold Calibration Controls
    st.sidebar.header("⚙️ Diagnostics Settings")
    st.sidebar.caption(f"Engine: **{model_type}**")
    
    default_meta_thresh = float(metadata.get("decision_threshold", 0.0546))
    
    med_threshold = st.sidebar.slider(
        "Medium Risk Cutoff (Decision Threshold)",
        min_value=0.01,
        max_value=0.30,
        value=default_meta_thresh,
        step=0.005,
        help="Calibrated model threshold for operational failure risk."
    )
    
    high_threshold = st.sidebar.slider(
        "High Risk Cutoff (Critical Failures)",
        min_value=0.05,
        max_value=0.60,
        value=0.15,
        step=0.01,
        help="Higher cutoff for immediate priority failure replacement."
    )

def preprocess_and_aggregate_raw_telemetry(raw_df: pd.DataFrame, expected_features: list) -> tuple:
    """Performs end-to-end preprocessing, temporal diff calculation, and VIN feature aggregation on raw telemetry CSVs."""
    df_proc = raw_df.dropna(subset=['vin']).copy() if 'vin' in raw_df.columns else raw_df.copy()
    
    # Dynamic sort alignment to prevent list length mismatches (Fixes ValueError)
    sort_cols = [c for c in ['vin', 'days_to_event', 'ts_timestamp_ist'] if c in df_proc.columns]
    if sort_cols:
        ascending_flags = [False if col == 'days_to_event' else True for col in sort_cols]
        df_proc = df_proc.sort_values(by=sort_cols, ascending=ascending_flags)
    
    # Standardize column naming variations across raw telemetry streams
    df_proc = df_proc.rename(columns={
        'o2_sensor_voltage': 'oxy_sensor_voltage',
        'o2_voltage': 'oxy_sensor_voltage',
        'intake_manifold_pressure': 'imap'
    })

    # Advanced Telemetry Drift & Physical Metric Engineering
    if 'oxy_sensor_voltage' in df_proc.columns and 'vin' in df_proc.columns:
        df_proc['oxy_voltage_diff'] = df_proc.groupby('vin')['oxy_sensor_voltage'].diff().fillna(0)
    else:
        df_proc['oxy_voltage_diff'] = 0.0

    if 'imap' in df_proc.columns and 'vin' in df_proc.columns:
        df_proc['imap_diff'] = df_proc.groupby('vin')['imap'].diff().fillna(0)
    else:
        df_proc['imap_diff'] = 0.0

    if 'mff_adap_add' in df_proc.columns and 'maf' in df_proc.columns:
        df_proc['fuel_to_air_ratio'] = df_proc['mff_adap_add'] / (df_proc['maf'] + 1e-5)
    else:
        df_proc['fuel_to_air_ratio'] = 0.0

    # Impute missing numeric values using median values
    num_cols = df_proc.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df_proc[col] = df_proc[col].fillna(df_proc[col].median() if not df_proc[col].isnull().all() else 0.0)

    # Group unique VIN identifiers
    grouped_vins = df_proc['vin'].unique() if 'vin' in df_proc.columns else np.array([f"Vehicle_{i+1}" for i in range(len(df_proc))])
    
    # Create output feature matrix matching expected model features
    if not expected_features:
        expected_features = [f"{c}_{stat}" for c in num_cols for stat in ['mean', 'std', 'min', 'max']]

    feature_matrix = pd.DataFrame(0.0, index=grouped_vins, columns=expected_features)

    # Calculate statistical aggregates (_mean, _std, _min, _max) per VIN
    if 'vin' in df_proc.columns:
        base_num = [c for c in num_cols if c not in ['oxy_voltage_diff', 'imap_diff', 'fuel_to_air_ratio']]
        for col in base_num:
            for stat in ['mean', 'std', 'min', 'max']:
                feat_name = f"{col}_{stat}"
                if feat_name in expected_features:
                    grouped_vals = df_proc.groupby('vin')[col].agg(stat).fillna(0)
                    feature_matrix.loc[grouped_vals.index, feat_name] = grouped_vals

        adv_mappings = {
            'oxy_voltage_diff': 'adv_oxy_voltage_diff',
            'imap_diff': 'adv_imap_diff',
            'fuel_to_air_ratio': 'adv_fuel_to_air_ratio'
        }
        for orig_col, target_prefix in adv_mappings.items():
            if orig_col in df_proc.columns:
                for stat in ['mean', 'std', 'max']:
                    feat_name = f"{target_prefix}_{stat}"
                    if feat_name in expected_features:
                        grouped_vals = df_proc.groupby('vin')[orig_col].agg(stat).fillna(0)
                        feature_matrix.loc[grouped_vals.index, feat_name] = grouped_vals

    return feature_matrix, grouped_vins

predict_file = st.file_uploader("📂 Drop Raw Telemetry Dataset Below (CSV Format)", type=["csv"], key="uploader")

# Reset session cache whenever a new file is uploaded
if "active_uploaded_file" in st.session_state and st.session_state["active_uploaded_file"] != (predict_file.name if predict_file else None):
    st.session_state.pop("fleet_df", None)

if predict_file and model is not None:
    st.session_state["active_uploaded_file"] = predict_file.name
    raw_df = pd.read_csv(predict_file, low_memory=False)
    
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.info(f"📁 Raw File: **{predict_file.name}** | Telemetry Rows: **{raw_df.shape[0]:,}**")
    with col_b:
        run_btn = st.button("🚀 Run Fleet Diagnostics", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Aggregating telemetry features per VIN and scoring fleet..."):
            expected_features = metadata.get("features_list", [])
            feature_matrix, grouped_vins = preprocess_and_aggregate_raw_telemetry(raw_df, expected_features)

            # Predict probabilities
            if hasattr(model, "predict_proba"):
                proba_array = model.predict_proba(feature_matrix)[:, 1]
            else:
                proba_array = model.predict(feature_matrix)

            st.session_state["fleet_df"] = pd.DataFrame({
                "vin": grouped_vins,
                "raw_score": proba_array
            })

    # Render outputs dynamically using sidebar slider values
    if "fleet_df" in st.session_state:
        fleet_df = st.session_state["fleet_df"].copy()
        total_units = len(fleet_df)

        # Dynamic Threshold Evaluation
        conditions = [
            (fleet_df["raw_score"] >= high_threshold),
            (fleet_df["raw_score"] >= med_threshold) & (fleet_df["raw_score"] < high_threshold)
        ]
        choices = ["🚨 HIGH RISK", "⚠️ MEDIUM RISK"]
        fleet_df["risk_flag"] = np.select(conditions, choices, default="✅ LOW RISK / HEALTHY")
        fleet_df["failure_probability"] = np.round(fleet_df["raw_score"], 4)

        high_count = int((fleet_df["risk_flag"] == "🚨 HIGH RISK").sum())
        med_count = int((fleet_df["risk_flag"] == "⚠️ MEDIUM RISK").sum())
        low_ok_count = int((fleet_df["risk_flag"] == "✅ LOW RISK / HEALTHY").sum())

        st.markdown("### 📊 Dynamic Fleet Diagnostics Overview")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Evaluated Fleet Units (VINs)", f"{total_units:,}")
        m2.metric(f"High Risk (≥ {high_threshold:.2f})", f"{high_count:,}")
        m3.metric(f"Medium Risk (≥ {med_threshold:.4f})", f"{med_count:,}")
        m4.metric("Low Risk / Healthy", f"{low_ok_count:,}")

        st.markdown("<br>", unsafe_allow_html=True)
        
        display_df = fleet_df[["vin", "failure_probability", "risk_flag"]].sort_values("failure_probability", ascending=False)

        col_header, col_download = st.columns([3, 1])
        with col_header:
            st.markdown("### 🔍 Fleet Risk Breakdown Table")
        with col_download:
            csv_data = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results CSV",
                data=csv_data,
                file_name="fleet_o2_risk_diagnostics.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )

        st.dataframe(
            display_df,
            column_config={
                "vin": st.column_config.TextColumn("VIN"),
                "failure_probability": st.column_config.NumberColumn("Failure Probability / Score", format="%.4f"),
                "risk_flag": st.column_config.TextColumn("Risk Classification")
            },
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("### 📋 Risk Category Definitions")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
                <div class="def-box def-high">
                    <div class="def-title" style="color: #D32F2F;">🚨 HIGH RISK (Score ≥ {high_threshold:.2f})</div>
                    <div class="def-text">Vehicles exhibiting severe anomaly probability scores. Immediate technical inspection and O2 sensor replacement strongly recommended.</div>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
                <div class="def-box def-med">
                    <div class="def-title" style="color: #E65100;">⚠️ MEDIUM RISK (Score ≥ {med_threshold:.4f})</div>
                    <div class="def-text">Vehicles exhibiting moderate degradation or elevated readings. Targeted for preventive maintenance during next service schedule.</div>
                </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
                <div class="def-box def-low">
                    <div class="def-title" style="color: #2E7D32;">✅ LOW RISK / HEALTHY (Score < {med_threshold:.4f})</div>
                    <div class="def-text">Vehicles operating within expected healthy parameters. Standard telemetry monitoring continues without intervention.</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)