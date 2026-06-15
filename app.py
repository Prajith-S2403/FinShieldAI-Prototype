# ──────────────────────────────────────────────────────────────
# FinShieldAI  –  Mule Account Detection Dashboard
# ──────────────────────────────────────────────────────────────
import io
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="FinShieldAI – Mule Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS / Dark Cyber Theme ─────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=Share+Tech+Mono&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: #070b14;
        color: #c8d8f0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1422 0%, #0a1020 100%);
        border-right: 1px solid #1a2a4a;
    }
    [data-testid="stSidebar"] * { color: #8ba8d4 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #5eead4 !important; }

    /* ── Header strip ── */
    .fin-header {
        background: linear-gradient(135deg, #0d1f3c 0%, #091628 50%, #0d1f3c 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 22px 32px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 16px;
        box-shadow: 0 0 40px rgba(0,200,255,0.06);
    }
    .fin-header h1 {
        font-size: 2rem;
        font-weight: 900;
        background: linear-gradient(90deg, #38bdf8, #5eead4, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .fin-header p { color: #64748b; margin: 0; font-size: 0.85rem; }

    /* ── KPI cards ── */
    .kpi-card {
        background: linear-gradient(145deg, #0f1f35, #0c1828);
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 22px 20px;
        text-align: center;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
        transition: transform 0.2s, box-shadow 0.2s;
        height: 140px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 32px rgba(56,189,248,0.12);
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 2.4rem;
        font-weight: 900;
        line-height: 1;
        font-family: 'Share Tech Mono', monospace;
    }
    .kpi-sub { font-size: 0.75rem; color: #475569; margin-top: 4px; }

    .kpi-blue   .kpi-value { color: #38bdf8; text-shadow: 0 0 20px rgba(56,189,248,0.4); }
    .kpi-red    .kpi-value { color: #f87171; text-shadow: 0 0 20px rgba(248,113,113,0.4); }
    .kpi-orange .kpi-value { color: #fb923c; text-shadow: 0 0 20px rgba(251,146,60,0.4); }
    .kpi-teal   .kpi-value { color: #34d399; text-shadow: 0 0 20px rgba(52,211,153,0.4); }

    /* ── Section titles ── */
    .section-title {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #38bdf8;
        margin: 28px 0 14px;
        border-left: 3px solid #38bdf8;
        padding-left: 10px;
    }

    /* ── Detail card ── */
    .detail-card {
        background: #0c1828;
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .risk-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }
    .badge-critical { background: rgba(239,68,68,0.2);  color: #f87171; border: 1px solid #ef4444; }
    .badge-high     { background: rgba(249,115,22,0.2); color: #fb923c; border: 1px solid #f97316; }
    .badge-medium   { background: rgba(234,179,8,0.2);  color: #facc15; border: 1px solid #eab308; }
    .badge-low      { background: rgba(34,197,94,0.2);  color: #4ade80; border: 1px solid #22c55e; }

    /* ── Reason code chips ── */
    .reason-chip {
        display: inline-block;
        background: rgba(56,189,248,0.1);
        border: 1px solid rgba(56,189,248,0.3);
        color: #7dd3fc;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.74rem;
        margin: 3px 3px 3px 0;
        font-family: 'Share Tech Mono', monospace;
    }

    /* ── STR report box ── */
    .str-box {
        background: #060e1a;
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 16px 20px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.78rem;
        color: #7dd3fc;
        white-space: pre-wrap;
        line-height: 1.7;
    }

    /* ── Plotly charts background ── */
    .js-plotly-plot { border-radius: 12px; }

    /* ── Inputs ── */
    .stSelectbox > div > div { background: #0c1828 !important; border-color: #1e3a5f !important; }
    .stTextInput > div > div > input { background: #0c1828; border-color: #1e3a5f; color: #c8d8f0; }

    /* hide default streamlit footer */
    footer { visibility: hidden; }

    /* ── Download button ── */
    .stDownloadButton button {
        background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        letter-spacing: 0.5px;
        transition: opacity 0.2s;
    }
    .stDownloadButton button:hover { opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────
FEATURES = [
    "sweep_ratio",
    "drawdown_score",
    "credit_debit_ratio",
    "community_risk_score",
    "transaction_velocity",
    "account_age_days",
    "govt_alert_flag",
]

FEATURE_LABELS = {
    "sweep_ratio":           "High Sweep Ratio",
    "drawdown_score":        "High Drawdown Score",
    "credit_debit_ratio":    "High Credit/Debit Ratio",
    "community_risk_score":  "Elevated Community Risk",
    "transaction_velocity":  "High Transaction Velocity",
    "account_age_days":      "Very New Account",
    "govt_alert_flag":       "Government Alert Flag",
}

CHART_THEME = dict(
    paper_bgcolor="#070b14",
    plot_bgcolor="#0a1020",
    font_color="#8ba8d4",
)
GRID_STYLE = dict(gridcolor="#1a2a40", zerolinecolor="#1a2a40")


# ── Helper functions ──────────────────────────────────────────

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df[FEATURES]
    y = df["label"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model, scaler


def categorize_risk(score: float) -> str:
    if score >= 0.75:
        return "Critical"
    elif score >= 0.50:
        return "High"
    elif score >= 0.25:
        return "Medium"
    return "Low"


def badge_html(category: str) -> str:
    cls = f"badge-{category.lower()}"
    return f'<span class="risk-badge {cls}">{category}</span>'


def get_reason_codes(row: pd.Series, model: XGBClassifier) -> list[str]:
    """Return human-readable reason codes based on feature importances & values."""
    importances = model.feature_importances_
    ranked = sorted(
        zip(FEATURES, importances, [row[f] for f in FEATURES]),
        key=lambda x: x[1],
        reverse=True,
    )
    reasons = []
    for feat, imp, val in ranked:
        if feat == "account_age_days" and val < 180:
            reasons.append(FEATURE_LABELS[feat])
        elif feat == "govt_alert_flag" and val == 1:
            reasons.append(FEATURE_LABELS[feat])
        elif feat not in ("account_age_days", "govt_alert_flag") and val >= 0.55:
            reasons.append(FEATURE_LABELS[feat])
        if len(reasons) >= 3:
            break
    if not reasons:
        reasons = ["No significant risk signals detected"]
    return reasons


def generate_str_report(
    row: pd.Series,
    risk_score: float,
    category: str,
    prediction: str,
    reasons: list[str],
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    action_map = {
        "Critical": "IMMEDIATE FREEZE & ESCALATE TO COMPLIANCE",
        "High":     "FLAG FOR INVESTIGATION & ENHANCED MONITORING",
        "Medium":   "SCHEDULE MANUAL REVIEW WITHIN 48 HOURS",
        "Low":      "CONTINUE ROUTINE MONITORING",
    }
    action = action_map.get(category, "MONITOR")
    report = textwrap.dedent(f"""
    ══════════════════════════════════════════════════════════════
     FINSHIELDAI  ·  SUSPICIOUS TRANSACTION REPORT (STR)
    ══════════════════════════════════════════════════════════════
     Generated   : {ts}
     Report Type : Mule Account Detection
    ──────────────────────────────────────────────────────────────
     ACCOUNT DETAILS
     Account ID           : {row['account_id']}
     Account Age (days)   : {int(row['account_age_days'])}
     Government Alert     : {'YES ⚠' if row['govt_alert_flag'] == 1 else 'No'}
    ──────────────────────────────────────────────────────────────
     RISK ASSESSMENT
     Risk Score           : {risk_score:.4f}
     Risk Category        : {category.upper()}
     Prediction           : {prediction}
    ──────────────────────────────────────────────────────────────
     RISK INDICATORS
     Sweep Ratio          : {row['sweep_ratio']:.4f}
     Drawdown Score       : {row['drawdown_score']:.4f}
     Credit/Debit Ratio   : {row['credit_debit_ratio']:.4f}
     Community Risk Score : {row['community_risk_score']:.4f}
     Transaction Velocity : {row['transaction_velocity']:.4f}
    ──────────────────────────────────────────────────────────────
     TOP REASON CODES
    {"".join(f"  [{i+1}] {r}" + chr(10) for i, r in enumerate(reasons))}
    ──────────────────────────────────────────────────────────────
     RECOMMENDED ACTION
     {action}
    ══════════════════════════════════════════════════════════════
     FinShieldAI · Automated AML / Mule Detection System
     This report is system-generated. Verify before regulatory filing.
    ══════════════════════════════════════════════════════════════
    """).strip()
    return report


# ── MAIN ──────────────────────────────────────────────────────

def main():
    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🛡️ FinShieldAI")
        st.markdown("**Mule Account Detection**")
        st.markdown("---")

        data_path = st.text_input("CSV Path", value="demo_accounts.csv")
        st.caption("Uses demo_accounts.csv by default.")
        st.markdown("---")

        threshold = st.slider(
            "Detection Threshold", 0.1, 0.9, 0.5, 0.01,
            help="Probability threshold above which an account is flagged as mule."
        )
        st.markdown("---")

        st.markdown("**Filter by Risk Category**")
        show_critical = st.checkbox("🔴 Critical", True)
        show_high     = st.checkbox("🟠 High", True)
        show_medium   = st.checkbox("🟡 Medium", True)
        show_low      = st.checkbox("🟢 Low", True)
        st.markdown("---")
        st.markdown("<p style='font-size:0.7rem;color:#334155;'>FinShieldAI v1.0 · AML Intelligence Platform</p>", unsafe_allow_html=True)

    # ── Header ───────────────────────────────────────────────
    st.markdown(
        """
        <div class="fin-header">
            <div style="font-size:2.5rem">🛡️</div>
            <div>
                <h1>FinShieldAI</h1>
                <p>AI-Powered Mule Account Detection &amp; AML Intelligence Dashboard</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load data ─────────────────────────────────────────────
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        st.error(f"❌ File not found: `{data_path}`. Run `python generate_data.py` first.")
        st.stop()

    # ── Train model ───────────────────────────────────────────
    with st.spinner("🔬 Training XGBoost classifier …"):
        model, scaler = train_model(df)

    # ── Predict ───────────────────────────────────────────────
    X_all = scaler.transform(df[FEATURES])
    df["risk_score"]    = model.predict_proba(X_all)[:, 1]
    df["risk_category"] = df["risk_score"].apply(categorize_risk)
    df["prediction"]    = (df["risk_score"] >= threshold).astype(int)
    df["pred_label"]    = df["prediction"].map({1: "Mule Account", 0: "Normal Account"})

    # ── KPI cards ─────────────────────────────────────────────
    total_accs      = len(df)
    mule_detected   = int(df["prediction"].sum())
    critical_count  = int((df["risk_category"] == "Critical").sum())
    avg_risk        = df["risk_score"].mean()

    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    cards = [
        (k1, "kpi-blue",   "Total Accounts",        f"{total_accs:,}",   "in dataset"),
        (k2, "kpi-red",    "Mule Accounts Detected", f"{mule_detected}",  f"threshold {threshold:.2f}"),
        (k3, "kpi-orange", "Critical Risk Accounts", f"{critical_count}",  "score ≥ 0.75"),
        (k4, "kpi-teal",   "Average Risk Score",     f"{avg_risk:.3f}",   "across all accounts"),
    ]
    for col, cls, label, val, sub in cards:
        with col:
            st.markdown(
                f"""
                <div class="kpi-card {cls}">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{val}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Charts ────────────────────────────────────────────────
    st.markdown('<div class="section-title">Analytics</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1, 1])

    # Chart 1 – Risk Score Distribution
    with c1:
        fig1 = px.histogram(
            df,
            x="risk_score",
            nbins=40,
            color="pred_label",
            barmode="overlay",
            opacity=0.8,
            color_discrete_map={"Mule Account": "#f87171", "Normal Account": "#38bdf8"},
            title="Risk Score Distribution",
            labels={"risk_score": "Risk Score", "pred_label": "Account Type"},
        )
        fig1.update_layout(
            **CHART_THEME,
            title_font_color="#38bdf8",
            legend=dict(font_color="#8ba8d4", bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=50, b=30, l=20, r=20),
        )
        fig1.update_xaxes(**GRID_STYLE)
        fig1.update_yaxes(**GRID_STYLE)
        fig1.update_traces(marker_line_width=0)
        st.plotly_chart(fig1, use_container_width=True)

    # Chart 2 – Mule vs Normal
    with c2:
        counts = df["pred_label"].value_counts().reset_index()
        counts.columns = ["Type", "Count"]
        fig2 = px.pie(
            counts,
            names="Type",
            values="Count",
            hole=0.55,
            color="Type",
            color_discrete_map={"Mule Account": "#f87171", "Normal Account": "#38bdf8"},
            title="Mule vs Normal",
        )
        fig2.update_layout(
            **CHART_THEME,
            title_font_color="#38bdf8",
            legend=dict(font_color="#8ba8d4", bgcolor="rgba(0,0,0,0)", orientation="h"),
            margin=dict(t=50, b=0, l=0, r=0),
            showlegend=True,
        )
        fig2.update_traces(textfont_color="white")
        st.plotly_chart(fig2, use_container_width=True)

    # Chart 3 – Risk Category breakdown
    with c3:
        cat_order = ["Critical", "High", "Medium", "Low"]
        cat_colors = {
            "Critical": "#f87171",
            "High":     "#fb923c",
            "Medium":   "#facc15",
            "Low":      "#4ade80",
        }
        cat_counts = (
            df["risk_category"]
            .value_counts()
            .reindex(cat_order, fill_value=0)
            .reset_index()
        )
        cat_counts.columns = ["Category", "Count"]
        fig3 = px.bar(
            cat_counts,
            x="Category",
            y="Count",
            color="Category",
            color_discrete_map=cat_colors,
            title="Risk Category Count",
        )
        fig3.update_layout(
            **CHART_THEME,
            title_font_color="#38bdf8",
            showlegend=False,
            margin=dict(t=50, b=30, l=20, r=20),
        )
        fig3.update_xaxes(**GRID_STYLE)
        fig3.update_yaxes(**GRID_STYLE)
        fig3.update_traces(marker_line_width=0)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Filtered Account Table ────────────────────────────────
    cat_filter = []
    if show_critical: cat_filter.append("Critical")
    if show_high:     cat_filter.append("High")
    if show_medium:   cat_filter.append("Medium")
    if show_low:      cat_filter.append("Low")

    filtered_df = df[df["risk_category"].isin(cat_filter)].copy()
    filtered_df = filtered_df.sort_values("risk_score", ascending=False)

    st.markdown('<div class="section-title">Account Risk Table</div>', unsafe_allow_html=True)
    display_cols = ["account_id", "risk_score", "risk_category", "pred_label",
                    "sweep_ratio", "drawdown_score", "credit_debit_ratio",
                    "community_risk_score", "transaction_velocity", "govt_alert_flag"]
    st.dataframe(
        filtered_df[display_cols]
        .rename(columns={
            "account_id":           "Account ID",
            "risk_score":           "Risk Score",
            "risk_category":        "Category",
            "pred_label":           "Prediction",
            "sweep_ratio":          "Sweep Ratio",
            "drawdown_score":       "Drawdown",
            "credit_debit_ratio":   "CR/DR Ratio",
            "community_risk_score": "Community Risk",
            "transaction_velocity": "Tx Velocity",
            "govt_alert_flag":      "Gov't Alert",
        })
        .style.background_gradient(subset=["Risk Score"], cmap="RdYlGn_r")
        .format({"Risk Score": "{:.4f}"}),
        use_container_width=True,
        height=320,
    )

    # ── Account Search & Detail ───────────────────────────────
    st.markdown('<div class="section-title">Account Deep Dive</div>', unsafe_allow_html=True)

    all_ids = sorted(df["account_id"].tolist())
    selected_id = st.selectbox("Search / Select Account ID", options=all_ids, index=0)

    row = df[df["account_id"] == selected_id].iloc[0]
    risk_score = float(row["risk_score"])
    category   = row["risk_category"]
    prediction = row["pred_label"]
    reasons    = get_reason_codes(row, model)

    d1, d2 = st.columns([1, 1])
    with d1:
        st.markdown(
            f"""
            <div class="detail-card">
                <div style="margin-bottom:12px">
                    <span style="font-size:1.1rem;font-weight:700;color:#e2e8f0">{selected_id}</span>
                    &nbsp;&nbsp;{badge_html(category)}
                </div>
                <table style="width:100%;border-collapse:collapse;font-size:0.84rem;">
                    <tr>
                        <td style="color:#64748b;padding:5px 0">Risk Score</td>
                        <td style="color:#38bdf8;font-family:'Share Tech Mono',monospace;font-weight:700;font-size:1.3rem">{risk_score:.4f}</td>
                    </tr>
                    <tr>
                        <td style="color:#64748b;padding:5px 0">Prediction</td>
                        <td style="color:#e2e8f0">{prediction}</td>
                    </tr>
                    <tr>
                        <td style="color:#64748b;padding:5px 0">Account Age</td>
                        <td style="color:#e2e8f0">{int(row['account_age_days'])} days</td>
                    </tr>
                    <tr>
                        <td style="color:#64748b;padding:5px 0">Gov't Alert</td>
                        <td style="color:{'#f87171' if row['govt_alert_flag']==1 else '#4ade80'}">{'⚠ YES' if row['govt_alert_flag']==1 else '✓ No'}</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Risk gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={"font": {"color": "#38bdf8", "size": 32}},
            gauge={
                "axis": {"range": [0, 1], "tickcolor": "#334155"},
                "bar": {"color": "#38bdf8"},
                "bgcolor": "#0a1020",
                "borderwidth": 0,
                "steps": [
                    {"range": [0.00, 0.25], "color": "rgba(34,197,94,0.2)"},
                    {"range": [0.25, 0.50], "color": "rgba(234,179,8,0.2)"},
                    {"range": [0.50, 0.75], "color": "rgba(249,115,22,0.2)"},
                    {"range": [0.75, 1.00], "color": "rgba(239,68,68,0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#f87171", "width": 3},
                    "thickness": 0.75,
                    "value": threshold,
                },
            },
            domain={"x": [0, 1], "y": [0, 1]},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0c1828",
            font_color="#8ba8d4",
            margin=dict(t=10, b=10, l=20, r=20),
            height=200,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with d2:
        st.markdown(
            "<div class='detail-card' style='height:100%'>",
            unsafe_allow_html=True,
        )
        st.markdown("**📊 Feature Values**", help="Raw feature values for this account")
        feat_vals = pd.DataFrame({
            "Feature": list(FEATURE_LABELS.values()),
            "Value":   [row[f] for f in FEATURES],
        })
        fig_bar = px.bar(
            feat_vals,
            x="Value",
            y="Feature",
            orientation="h",
            color="Value",
            color_continuous_scale=["#22c55e", "#eab308", "#f97316", "#ef4444"],
            range_color=[0, 1],
        )
        fig_bar.update_layout(
            **CHART_THEME,
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=220,
        )
        fig_bar.update_xaxes(range=[0, 1], **GRID_STYLE)
        fig_bar.update_yaxes(**GRID_STYLE)
        fig_bar.update_traces(marker_line_width=0)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("**🔍 Reason Codes**")
        chips = "".join(f'<span class="reason-chip">⚡ {r}</span>' for r in reasons)
        st.markdown(chips, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── STR Report ────────────────────────────────────────────
    st.markdown('<div class="section-title">STR Report Generator</div>', unsafe_allow_html=True)

    str_report = generate_str_report(row, risk_score, category, prediction, reasons)

    st.markdown(f'<div class="str-box">{str_report}</div>', unsafe_allow_html=True)

    st.download_button(
        label="⬇️  Download STR Report (.txt)",
        data=str_report.encode("utf-8"),
        file_name=f"STR_{selected_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )

    # ── Bulk STR export ───────────────────────────────────────
    st.markdown('<div class="section-title">Bulk Export – All Flagged Accounts</div>', unsafe_allow_html=True)
    flagged = df[df["prediction"] == 1].copy()

    bulk_rows = []
    for _, frow in flagged.iterrows():
        rc  = get_reason_codes(frow, model)
        act_map = {
            "Critical": "IMMEDIATE FREEZE & ESCALATE",
            "High":     "FLAG FOR INVESTIGATION",
            "Medium":   "SCHEDULE REVIEW",
            "Low":      "MONITOR",
        }
        bulk_rows.append({
            "account_id":      frow["account_id"],
            "risk_score":      round(float(frow["risk_score"]), 4),
            "risk_category":   frow["risk_category"],
            "prediction":      frow["pred_label"],
            "top_reason_codes": " | ".join(rc),
            "recommended_action": act_map.get(frow["risk_category"], "MONITOR"),
        })
    bulk_df = pd.DataFrame(bulk_rows)

    st.dataframe(bulk_df, use_container_width=True, height=250)

    csv_bytes = bulk_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Export All Flagged Accounts (.csv)",
        data=csv_bytes,
        file_name=f"FinShieldAI_STR_Bulk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
