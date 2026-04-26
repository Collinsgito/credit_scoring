"""
dashboard/app.py
----------------
Streamlit frontend for the Credit Risk Scoring API.

Run: streamlit run dashboard/app.py
     (Make sure the FastAPI server is running on port 8000 first)
"""

import streamlit as st
import requests
import plotly.graph_objects as go
from math import sqrt

# ── Config ─────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Credit Risk Scorer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg: #080C14;
        --bg-soft: #0F1524;
        --card: #121A2B;
        --card-2: #18233A;
        --text: #E6EDF8;
        --muted: #9AA9C3;
        --accent: #24B8C8;
        --accent-2: #67E4B0;
        --danger: #FF6B6B;
        --success: #6EE7B7;
        --line: #243149;
    }

    .stApp {
        background:
            radial-gradient(900px 360px at 12% -8%, rgba(36, 184, 200, 0.22) 0%, transparent 55%),
            radial-gradient(720px 320px at 95% 8%, rgba(103, 228, 176, 0.18) 0%, transparent 56%),
            linear-gradient(180deg, #060A12 0%, var(--bg) 100%);
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text);
    }

    .block-container {
        padding-top: 1.35rem;
    }

    h1, h2, h3, h4, h5, h6, p, label, span {
        color: var(--text) !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B1220 0%, #0F1A30 100%);
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] * {
        color: #E6EDF8 !important;
        font-family: 'Space Grotesk', sans-serif;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #2A4464;
        background: linear-gradient(120deg, #1B3552 0%, #1A5970 100%);
        color: #ECF8FF;
        font-weight: 700;
    }

    .stButton > button:hover {
        border-color: #3E6B95;
        filter: brightness(1.08);
    }

    .hero-wrap {
        background: linear-gradient(120deg, #0E1A2D 0%, #132B45 50%, #145A78 100%);
        border: 1px solid #27517A;
        border-radius: 18px;
        padding: 1.2rem 1.3rem;
        color: #F6FBFF;
        box-shadow: 0 24px 46px rgba(2, 8, 20, 0.48);
        animation: riseIn .45s ease-out;
    }

    .main-header {
        font-size: clamp(1.65rem, 2.3vw, 2.35rem);
        font-weight: 700;
        letter-spacing: 0.2px;
        margin: 0;
        line-height: 1.15;
        color: #F7FCFF;
    }

    .sub-header {
        margin-top: .4rem;
        font-size: 0.96rem;
        color: #B9D4EA;
        max-width: 60ch;
    }

    .chip-row {
        margin-top: 0.8rem;
        display: flex;
        gap: .5rem;
        flex-wrap: wrap;
    }

    .chip {
        font-size: .78rem;
        border-radius: 999px;
        background: rgba(188, 227, 245, 0.12);
        border: 1px solid rgba(188, 227, 245, 0.28);
        padding: .27rem .64rem;
        color: #E5F8FF;
    }

    .risk-badge {
        padding: 0.5rem 1.1rem;
        border-radius: 999px;
        font-size: 1rem;
        font-weight: 700;
        text-align: center;
        display: inline-block;
        margin: 0.35rem 0 0.7rem;
        letter-spacing: .2px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(160deg, var(--card) 0%, var(--card-2) 100%);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: .6rem .8rem;
        box-shadow: 0 16px 28px rgba(2, 8, 20, 0.3);
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div {
        color: var(--text) !important;
    }

    div[data-testid="stAlert"] {
        background: rgba(18, 26, 43, 0.82);
        border: 1px solid var(--line);
        border-radius: 12px;
    }

    .factor-card {
        background: var(--card);
        border-radius: 12px;
        padding: 0.8rem 0.95rem;
        margin: 0.45rem 0;
        border-left: 4px solid;
        border-top: 1px solid var(--line);
        border-right: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        color: var(--text);
        box-shadow: 0 12px 22px rgba(2, 8, 20, 0.32);
    }

    .mono {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.86rem;
    }

    .footer {
        font-size: 0.78rem;
        color: var(--muted);
        text-align: center;
        margin-top: 2.4rem;
    }

    @keyframes riseIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0px); }
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ───────────────────────────────────────────
def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200 and r.json().get("model_loaded", False)
    except Exception:
        return False


def score_applicant(payload: dict):
    try:
        r = requests.post(f"{API_URL}/score", json=payload, timeout=10)
        if r.status_code == 200:
            return r.json(), None
        return None, r.json().get("detail", "API error")
    except requests.ConnectionError:
        return None, "Cannot connect to API. Is the server running? (uvicorn api.main:app --reload)"
    except Exception as e:
        return None, str(e)


def gauge_chart(score: int, prob: float):
    tier_color = "#6EE7B7"
    if prob < 0.05:
        tier_color = "#6EE7B7"
    elif prob < 0.12:
        tier_color = "#F6C177"
    elif prob < 0.25:
        tier_color = "#F59E7A"
    else:
        tier_color = "#FF6B6B"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 48, "color": tier_color}},
        gauge={
            "axis": {"range": [0, 1000], "tickwidth": 1, "tickcolor": "#8EA3C2"},
            "bar":  {"color": tier_color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 1,
            "bordercolor": "#2A3850",
            "steps": [
                {"range": [0,   200], "color": "#3B1F2B"},   # Very High Risk
                {"range": [200, 400], "color": "#3A2C24"},   # High Risk
                {"range": [400, 700], "color": "#2F2E23"},   # Medium Risk
                {"range": [700, 1000],"color": "#20382D"},   # Low Risk
            ],
            "threshold": {
                "line": {"color": tier_color, "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
        title={"text": "Risk Score", "font": {"size": 16, "color": "#E6EDF8"}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=295,
        margin=dict(l=12, r=12, t=45, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[{
            "text": "0 = highest risk, 1000 = safest",
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": -0.1,
            "showarrow": False,
            "font": {"size": 11, "color": "#93A4BE"}
        }]
    )
    return fig


def default_donut(prob: float):
    non_default = max(0.0, 1.0 - prob)
    fig = go.Figure(
        data=[
            go.Pie(
                values=[prob, non_default],
                labels=["Default", "Repay"],
                hole=0.68,
                marker={"colors": ["#FF6B6B", "#6EE7B7"]},
                textinfo="none",
                sort=False,
            )
        ]
    )
    fig.update_layout(
        height=295,
        margin=dict(l=10, r=10, t=32, b=10),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1, x=0.2),
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[
            {
                "text": f"<b>{prob * 100:.1f}%</b><br><span style='font-size:11px;'>Default risk</span>",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 20, "color": "#E6EDF8"},
            }
        ],
    )
    return fig


def shap_bar_chart(risk_factors, protective_factors):
    labels, values, colors = [], [], []

    for f in protective_factors[::-1]:
        labels.append(f["feature"])
        values.append(f["shap_value"])
        colors.append("#6EE7B7")

    for f in risk_factors:
        labels.append(f["feature"])
        values.append(f["shap_value"])
        colors.append("#FF8474")

    if not labels:
        return None

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Decision Drivers (SHAP)",
        xaxis_title="Impact on default probability",
        height=max(300, len(labels) * 45),
        margin=dict(l=10, r=40, t=46, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EDF8"),
        xaxis=dict(zeroline=True, zerolinecolor="#344866", zerolinewidth=1.5),
    )
    return fig


def profile_radar(payload: dict):
    # Normalize key profile inputs to 0..1 for a compact visual fingerprint.
    age_norm = min(1.0, payload["age"] / 90.0)
    fico_norm = min(1.0, max(0.0, (payload["fico_score"] - 300.0) / 550.0))
    income_norm = min(1.0, payload["monthly_income"] / 15000.0)
    debt_norm = min(1.0, payload["debt_ratio"] / 1.5)
    util_norm = min(1.0, payload["revolving_utilization"] / 1.5)
    late_total = payload["num_30_59_days_late"] + payload["num_60_89_days_late"] + payload["num_90_days_late"]
    late_norm = min(1.0, late_total / 15.0)

    labels = ["Age", "FICO", "Income", "Debt", "Utilization", "Late Payments"]
    values = [age_norm, fico_norm, income_norm, debt_norm, util_norm, late_norm]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name="Applicant Profile",
        line={"color": "#24B8C8", "width": 2.5},
        fillcolor="rgba(36,184,200,0.22)",
    ))
    fig.update_layout(
        title="Applicant Profile Fingerprint",
        polar={
            "bgcolor": "rgba(0,0,0,0)",
            "radialaxis": {
                "visible": True,
                "range": [0, 1],
                "gridcolor": "#314661",
                "linecolor": "#314661",
                "tickfont": {"color": "#B7C6DE"},
            },
            "angularaxis": {
                "gridcolor": "#314661",
                "tickfont": {"color": "#D2DDF0"},
            },
        },
        showlegend=False,
        height=350,
        margin=dict(l=10, r=10, t=40, b=16),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EDF8"),
    )
    return fig


# ── Sidebar inputs ─────────────────────────────────────────────
st.sidebar.markdown("## 💳 Applicant Details")
st.sidebar.markdown("---")

fico_score = st.sidebar.slider("FICO Credit Score", 300, 850, 685, step=5,
    help="Credit score range: 300 (poor) to 850 (exceptional)")

st.sidebar.markdown("**Income & Debt**")
monthly_income = st.sidebar.number_input("Monthly Income (USD)", 0, 50_000, 4_500, step=100)
debt_ratio     = st.sidebar.slider("Debt Ratio", 0.0, 1.5, 0.35, step=0.01,
    help="Monthly debt payments / gross monthly income")
revolving_util = st.sidebar.slider("Revolving Credit Utilization", 0.0, 1.5, 0.30, step=0.01,
    help="Balance / Credit limit across all cards")

st.sidebar.markdown("**Personal**")
age            = st.sidebar.slider("Age", 21, 90, 42)
num_dependents = st.sidebar.slider("Number of Dependents", 0, 10, 2)

st.sidebar.markdown("**Credit History**")
open_lines         = st.sidebar.slider("Open Credit Lines", 0, 30, 7)
real_estate_loans  = st.sidebar.slider("Real Estate Loans", 0, 10, 1)
late_30_59         = st.sidebar.slider("Times 30–59 days late", 0, 15, 0)
late_60_89         = st.sidebar.slider("Times 60–89 days late", 0, 15, 0)
late_90plus        = st.sidebar.slider("Times 90+ days late",   0, 15, 0)

score_btn = st.sidebar.button("🔍  Score This Applicant", use_container_width=True, type="primary")


# ── Main page ─────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-wrap">
        <div class="main-header">Credit Risk Decision Studio</div>
        <div class="sub-header">Production scoring with XGBoost and decision-level explainability using SHAP.</div>
        <div class="chip-row">
            <span class="chip">Real-time Scoring</span>
            <span class="chip">Decision Transparency</span>
            <span class="chip">Risk-Tier Classification</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# API status
api_ok = check_api_health()
if api_ok:
    st.success("✅ API connected and model loaded", icon="✅")
else:
    st.error("⚠️ API not available. Start it: `uvicorn api.main:app --reload --port 8000`")

st.divider()

# ── Score on button click ──────────────────────────────────────
if score_btn:
    payload = {
        "revolving_utilization": revolving_util,
        "age":                   age,
        "num_30_59_days_late":   late_30_59,
        "debt_ratio":            debt_ratio,
        "monthly_income":        float(monthly_income),
        "num_open_credit_lines": open_lines,
        "num_90_days_late":      late_90plus,
        "num_real_estate_loans": real_estate_loans,
        "num_60_89_days_late":   late_60_89,
        "num_dependents":        num_dependents,
        "fico_score":            fico_score,
    }

    with st.spinner("Scoring applicant..."):
        result, error = score_applicant(payload)

    if error:
        st.error(f"Error: {error}")
    elif result:
        # ── Row 1: Key metrics ─────────────────────────────
        col1, col2, col3, col4 = st.columns(4)

        tier_colors = {
            "Low Risk": "#1A7A4A", "Medium Risk": "#C9A84C",
            "High Risk": "#D05538", "Very High Risk": "#8B1A1A"
        }
        tier_color = tier_colors.get(result["risk_tier"], "#555")

        with col1:
            st.metric("Probability of Default",
                      f"{result['probability_of_default']*100:.1f}%")
        with col2:
            st.metric("Risk Score", f"{result['risk_score']}/1000")
        with col3:
            st.metric("FICO Band",
                      f"{result['fico_band']} ({result['fico_label']})")
        with col4:
            st.metric("Decision", result["decision"])

        # Risk tier badge
        st.markdown(
            f'<div class="risk-badge" style="background:{tier_color}1F;color:{tier_color};'
            f'border:1px solid {tier_color}66">{result["risk_tier"]}</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Row 2: Gauge + Probability donut + SHAP ───────
        col_g, col_d, col_s = st.columns([1.05, 1.05, 1.7])

        with col_g:
            st.plotly_chart(
                gauge_chart(result["risk_score"], result["probability_of_default"]),
                use_container_width=True
            )

        with col_d:
            st.plotly_chart(default_donut(result["probability_of_default"]), use_container_width=True)

        with col_s:
            shap_fig = shap_bar_chart(
                result["top_risk_factors"],
                result["top_protective_factors"]
            )
            if shap_fig:
                st.plotly_chart(shap_fig, use_container_width=True)
            else:
                st.info("SHAP explainer not available. Run train_model.py to enable explanations.")

        st.divider()

        # ── Row 3: Profile fingerprint ─────────────────────
        st.plotly_chart(profile_radar(payload), use_container_width=True)

        st.divider()

        # ── Row 4: Factor cards ────────────────────────────
        col_r, col_p = st.columns(2)

        with col_r:
            st.markdown("### 🔴 Top Risk Factors")
            for f in result.get("top_risk_factors", []):
                st.markdown(
                    f'<div class="factor-card" style="border-left-color:#D05538">'
                    f'<strong>{f["feature"]}</strong><br/>'
                    f'Value: <span class="mono">{f["value"]:.3f}</span> &nbsp;|&nbsp; '
                    f'Impact: <span style="color:#D05538">+{f["shap_value"]:.3f}</span></div>',
                    unsafe_allow_html=True
                )
            if not result.get("top_risk_factors"):
                st.info("No significant risk factors identified.")

        with col_p:
            st.markdown("### 🟢 Top Protective Factors")
            for f in result.get("top_protective_factors", []):
                st.markdown(
                    f'<div class="factor-card" style="border-left-color:#1A7A4A">'
                    f'<strong>{f["feature"]}</strong><br/>'
                    f'Value: <span class="mono">{f["value"]:.3f}</span> &nbsp;|&nbsp; '
                    f'Impact: <span style="color:#1A7A4A">{f["shap_value"]:.3f}</span></div>',
                    unsafe_allow_html=True
                )
            if not result.get("top_protective_factors"):
                st.info("No significant protective factors identified.")

        st.divider()

        # ── Raw JSON (expander) ────────────────────────────
        with st.expander("📄 View raw API response (JSON)"):
            st.json(result)

else:
    # Landing state
    st.markdown("### Fill in applicant details on the left, then click **Score This Applicant**")
    st.markdown("""
    **What this tool does:**
    - Predicts probability of loan default using XGBoost
    - Classifies applicants into risk tiers (Low / Medium / High / Very High)
    - Provides SHAP-based explanation of each decision
    - Classifies FICO score into standard credit bands

    **API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
    """)

st.markdown(
    '<div class="footer">Built by Collins Gitonga Mutembei &middot; '
    'XGBoost + SHAP + FastAPI + Streamlit &middot; v1.0.0</div>',
    unsafe_allow_html=True
)
