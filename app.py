import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Manufacturing Output Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- THEME-ADAPTIVE CSS (LIGHT & DARK MODE READY) ---
st.markdown("""
<style>
    /* Header card adapts dynamically */
    .header-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 14px 20px;
        margin-bottom: 20px;
    }

    .header-title {
        font-weight: 700;
        font-size: 17px;
        color: var(--text-color);
        letter-spacing: -0.3px;
        line-height: 1.2;
    }

    .header-subtitle {
        font-size: 11px;
        color: var(--text-color);
        opacity: 0.65;
        letter-spacing: 0.6px;
        font-weight: 600;
        margin-top: 3px;
    }

    /* Formatted Output Colored Metric Cards */
    .output-card {
        border-radius: 10px;
        padding: 14px 18px;
        border: 1px solid transparent;
        text-align: center;
        transition: transform 0.15s ease-in-out;
    }
    .output-card:hover {
        transform: translateY(-2px);
    }
    .card-green {
        background: rgba(16, 185, 129, 0.12);
        border-color: rgba(16, 185, 129, 0.35);
    }
    .card-green .card-label { color: #10b981; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
    .card-green .card-value { color: #059669; font-size: 26px; font-weight: 800; margin-top: 4px; }

    .card-purple {
        background: rgba(139, 92, 246, 0.12);
        border-color: rgba(139, 92, 246, 0.35);
    }
    .card-purple .card-label { color: #8b5cf6; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
    .card-purple .card-value { color: #7c3aed; font-size: 22px; font-weight: 800; margin-top: 6px; }

    .card-amber {
        background: rgba(245, 158, 11, 0.12);
        border-color: rgba(245, 158, 11, 0.35);
    }
    .card-amber .card-label { color: #f59e0b; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
    .card-amber .card-value { color: #d97706; font-size: 26px; font-weight: 800; margin-top: 4px; }

    /* Primary buttons adhere cleanly to theme */
    .stButton > button[kind="primary"] {
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL DEFINITIONS ---
NUM_FIELDS = [
    ("Injection_Temperature", "Injection temperature (°C)", 215.3, 0.1, "Process Settings"),
    ("Injection_Pressure", "Injection pressure", 116.0, 0.1, "Process Settings"),
    ("Cycle_Time", "Cycle time (s)", 35.8, 0.1, "Process Settings"),
    ("Cooling_Time", "Cooling time (s)", 11.9, 0.1, "Process Settings"),
    ("Material_Viscosity", "Material viscosity", 251.6, 1.0, "Material and Environment"),
    ("Ambient_Temperature", "Ambient temperature (°C)", 22.9, 0.1, "Material and Environment"),
    ("Machine_Age", "Machine age (years)", 7.8, 0.1, "Machine and Crew"),
    ("Operator_Experience", "Operator experience (years)", 30.8, 0.1, "Machine and Crew"),
    ("Maintenance_Hours", "Maintenance hours", 50.5, 1.0, "Machine and Crew"),
    ("Efficiency_Score", "Efficiency score", 0.192, 0.001, "Performance Indicators"),
    ("Machine_Utilization", "Machine utilization", 0.356, 0.01, "Performance Indicators"),
]

CATEGORIES = {
    "Shift": ["Day", "Evening", "Night"],
    "Machine_Type": ["Type_A", "Type_B", "Type_C"],
    "Material_Grade": ["Standard", "Economy", "Premium"],
    "Day_of_Week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
}

NUM_KEYS = [f[0] for f in NUM_FIELDS]
SWEEP_KEYS = NUM_KEYS[:9]
RMSE_VALUE = 3.5046
INTERCEPT = 28.689407618082345
SCALER_MEAN = np.array(
    [215.309875, 116.283, 35.6485, 11.94675, 253.572158, 22.9011465, 7.838625, 30.2438931, 50.9275, 1.8831425,
     47.548125, 0.18809, 0.35712625])
SCALER_SCALE = np.array(
    [11.73449466, 14.62648235, 8.41869929, 2.27534381, 72.80728409, 2.75120178, 3.90936959, 26.79225764, 17.46359767,
     0.27440402, 8.68525742, 0.16907937, 0.19591908])
COEFS = np.array(
    [2.29240292, 1.19700802, -0.57535719, 2.66640341, -0.94357647, -0.12181049, -2.22931392, 0.94803023, -0.07133237,
     -0.90177507, -10.04769799, 2.94594198, -0.42857523, 1.89896735, 0.16298047, -2.06194782, 2.11703076, -0.31173003,
     -1.80530073, -1.58878971, 1.52549867, 0.06329105, -0.68449779, -0.05095825, -0.11434662, -0.53717988, 0.25852863,
     0.06129878, 1.06715512])
CAT_ORDER = {'Shift': ['Day', 'Evening', 'Night'], 'Machine_Type': ['Type_A', 'Type_B', 'Type_C'],
             'Material_Grade': ['Economy', 'Premium', 'Standard'],
             'Day_of_Week': ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']}
NUM_COLS = ['Injection_Temperature', 'Injection_Pressure', 'Cycle_Time', 'Cooling_Time', 'Material_Viscosity',
            'Ambient_Temperature', 'Machine_Age', 'Operator_Experience', 'Maintenance_Hours',
            'Temperature_Pressure_Ratio', 'Total_Cycle_Time', 'Efficiency_Score', 'Machine_Utilization']


@st.cache_resource
def get_model():
    for path in ["model_output.pkl", "manufacturing_frontend/model_output.pkl"]:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
    return None


model = get_model()


def predict_df(df):
    df_clean = df.copy()
    if "Total_Cycle_Time" not in df_clean.columns:
        df_clean["Total_Cycle_Time"] = df_clean["Cycle_Time"] + df_clean["Cooling_Time"]
    if "Temperature_Pressure_Ratio" not in df_clean.columns:
        df_clean["Temperature_Pressure_Ratio"] = df_clean["Injection_Temperature"] / df_clean["Injection_Pressure"]
    if model is not None:
        try:
            return np.round(model.predict(df_clean), 2)
        except Exception:
            pass
    X_num = df_clean[NUM_COLS].fillna(pd.Series(SCALER_MEAN, index=NUM_COLS)).values
    X_num_scaled = (X_num - SCALER_MEAN) / SCALER_SCALE
    cat_cols = [(df_clean[c].values == cat).astype(float) for c in CAT_ORDER for cat in CAT_ORDER[c]]
    X_cat = np.column_stack(cat_cols)
    return np.maximum(0.0, np.round(np.hstack([X_num_scaled, X_cat]).dot(COEFS) + INTERCEPT, 2))


def nice(name):
    return name.replace("_", " ")


# --- HEADER WITH THEME-AWARE SVG LOGO ---
st.markdown("""
<div class="header-card">
    <div style="display:flex; align-items:center; gap:14px;">
        <svg width="44" height="44" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="4" y="20" width="16" height="34" rx="2" fill="currentColor" fill-opacity="0.85"/>
            <rect x="44" y="20" width="16" height="34" rx="2" fill="currentColor" fill-opacity="0.85"/>
            <path d="M7 18 L12 8 L17 18 Z" fill="currentColor" fill-opacity="0.7"/>
            <path d="M47 18 L52 8 L57 18 Z" fill="currentColor" fill-opacity="0.7"/>
            <path d="M18 24 L32 44 L46 24" stroke="currentColor" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
            <circle cx="32" cy="27" r="8" fill="var(--secondary-background-color)" stroke="currentColor" stroke-width="2.5"/>
            <circle cx="32" cy="27" r="3.5" fill="currentColor"/>
        </svg>
        <div>
            <div class="header-title">Manufacturing Equipment Output Prediction</div>
            <div class="header-subtitle">INDUSTRIAL INTELLIGENCE • ACTIVE MODEL</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Sensitivity Analysis", "Batch Scoring", "Model Details"])

# TAB 1: PREDICTION
with tab1:
    row = {}
    groups = {}
    for f in NUM_FIELDS:
        groups.setdefault(f[4], []).append(f)
    for group, fields in groups.items():
        with st.container(border=True):
            st.markdown(f"**{group}**")
            cols = st.columns(4)
            for i, (key, label, default, step, _) in enumerate(fields):
                row[key] = cols[i % 4].number_input(label, value=float(default), step=float(step), key=f"p_{key}")

    with st.container(border=True):
        st.markdown("**Operational Context**")
        cols = st.columns(4)
        for i, (key, options) in enumerate(CATEGORIES.items()):
            row[key] = cols[i].selectbox(nice(key), options, key=f"p_{key}")

    row["Total_Cycle_Time"] = round(row["Cycle_Time"] + row["Cooling_Time"], 2)
    row["Temperature_Pressure_Ratio"] = round(row["Injection_Temperature"] / row["Injection_Pressure"], 3)
    st.session_state["base_row"] = row

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Cycle Time", f"{row['Total_Cycle_Time']} s")
    col_b.metric("Temp / Pressure Ratio", f"{row['Temperature_Pressure_Ratio']}")
    actual = col_c.number_input("Observed output (optional)", min_value=0.0, value=None, step=0.1)

    if st.button("Run Prediction", type="primary", use_container_width=True):
        pred = float(predict_df(pd.DataFrame([row]))[0])
        st.divider()

        # Formulated Color Cards
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="output-card card-green">
                <div class="card-label">Predicted Output</div>
                <div class="card-value">{pred:.2f} <span style="font-size:14px; font-weight:600;">parts/hr</span></div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="output-card card-purple">
                <div class="card-label">Confidence Interval</div>
                <div class="card-value">{max(0.0, pred - RMSE_VALUE):.2f} – {pred + RMSE_VALUE:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="output-card card-amber">
                <div class="card-label">Model Uncertainty</div>
                <div class="card-value">± {RMSE_VALUE:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        if actual is not None:
            gap = pred - actual
            if abs(gap) > RMSE_VALUE:
                st.warning(f"Deviation detected: {gap:.2f} parts/hr from prediction.")
            else:
                st.success(f"Within expected variance. Deviation: {gap:.2f}")

# TAB 2: SENSITIVITY ANALYSIS
with tab2:
    base = st.session_state.get("base_row", {f[0]: f[2] for f in NUM_FIELDS} | {k: v[0] for k, v in CATEGORIES.items()})
    with st.container(border=True):
        c1, c2 = st.columns(2)
        key = c1.selectbox("Parameter", SWEEP_KEYS, format_func=nice)
        pct = c2.slider("Variation Range (%)", 5, 30, 12)
        run = st.button("Execute Simulation", use_container_width=True)

    if run:
        curr_val = float(base[key])
        xs = [curr_val * (1 - pct / 100 + (2 * pct / 100) * i / 20) for i in range(21)]
        test_rows = []
        for x in xs:
            r = dict(base)
            r[key] = x
            if key in ["Cycle_Time", "Cooling_Time"]:
                r["Total_Cycle_Time"] = r["Cycle_Time"] + r["Cooling_Time"]
            if key in ["Injection_Temperature", "Injection_Pressure"] and r["Injection_Pressure"] != 0:
                r["Temperature_Pressure_Ratio"] = r["Injection_Temperature"] / r["Injection_Pressure"]
            test_rows.append(r)
        ys = predict_df(pd.DataFrame(test_rows))

        with st.container(border=True):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='lines+markers',
                line=dict(color='#0284c7', width=2.5),
                marker=dict(size=5, color='#0284c7'),
                name="Sensitivity"
            ))
            fig.add_vline(
                x=curr_val, line_dash="dash", line_color="#eab308", line_width=1.8,
                annotation_text="Baseline", annotation_position="top left"
            )

            fig.update_layout(
                template=None,
                height=340,
                margin=dict(l=30, r=20, t=20, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    title=nice(key),
                    gridcolor="rgba(128, 128, 128, 0.2)",
                    zerolinecolor="rgba(128, 128, 128, 0.2)"
                ),
                yaxis=dict(
                    title="Predicted Parts Per Hour",
                    gridcolor="rgba(128, 128, 128, 0.2)",
                    zerolinecolor="rgba(128, 128, 128, 0.2)"
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(pd.DataFrame({nice(key): [round(x, 4) for x in xs], "Predicted Output": ys}), hide_index=True,
                     use_container_width=True)

# TAB 3: BATCH SCORING
with tab3:
    with st.container(border=True):
        file = st.file_uploader("Upload CSV file for batch prediction", type="csv")
        if file:
            df = pd.read_csv(file)
            df["Predicted_Parts_Per_Hour"] = predict_df(df)
            if "Parts_Per_Hour" in df.columns:
                df["Residual"] = (df["Parts_Per_Hour"] - df["Predicted_Parts_Per_Hour"]).round(2)
                df["Status"] = df["Residual"].apply(lambda r: "Underperforming" if r < -RMSE_VALUE else "Within Range")
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.download_button("Export Results as CSV", df.to_csv(index=False), "predictions.csv", "text/csv")

# TAB 4: MODEL DETAILS
with tab4:
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R-Squared", "0.9059")
        c2.metric("RMSE", "3.5046")
        c3.metric("MAE", "2.7158")
        c4.metric("MSE", "12.2820")
        st.divider()
