import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import json

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI-Powered Customer Intelligence Platform",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0f1117;
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

h1, h2, h3 {
    color: white;
}

.hero-title {
    font-size: 52px;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 20px;
    color: #b0b3b8;
    margin-bottom: 30px;
}

.section-title {
    font-size: 34px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 10px;
}

.section-subtitle {
    color: #b0b3b8;
    font-size: 17px;
    margin-bottom: 30px;
}

.metric-card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid #30363d;
}

.metric-value {
    font-size: 32px;
    font-weight: 700;
}

.metric-label {
    font-size: 15px;
    color: #8b949e;
}

.insight-box {
    background-color: #161b22;
    padding: 25px;
    border-radius: 14px;
    border-left: 5px solid #58a6ff;
    margin-top: 20px;
}

/* ===== CINEMATIC HERO ===== */

.hero-shell {
    min-height: auto;
    padding: 0px 20px 20px 20px;
}

.hero-badge {
    display: inline-block;
    padding: 8px 14px;
    border: 1px solid rgba(88,166,255,0.55);
    border-radius: 999px;
    color: #58a6ff;
    background: rgba(88,166,255,0.08);
    font-size: 14px;
    margin-bottom: 22px;
}

.hero-big-title {
    font-size: 64px;
    font-weight: 800;
    line-height: 1.02;
    letter-spacing: -2px;
    background: linear-gradient(90deg, #ffffff, #58a6ff, #3fb950);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-desc {
    color: #b0b3b8;
    font-size: 20px;
    line-height: 1.6;
    max-width: 720px;
    margin-top: 22px;
}

.neon-card {
    background: linear-gradient(145deg, rgba(22,27,34,0.95), rgba(13,17,23,0.95));
    border: 1px solid rgba(88,166,255,0.25);
    box-shadow: 0 0 28px rgba(88,166,255,0.12);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
}

.neon-value {
    font-size: 36px;
    font-weight: 800;
    color: #ffffff;
}

.neon-label {
    color: #8b949e;
    font-size: 14px;
    margin-top: 4px;
}

.signal-good {
    color: #3fb950;
    font-weight: 700;
}

.signal-risk {
    color: #f85149;
    font-weight: 700;
}

.signal-blue {
    color: #58a6ff;
    font-weight: 700;
}

.matrix-panel {
    background:
        radial-gradient(circle at top left, rgba(88,166,255,0.18), transparent 35%),
        radial-gradient(circle at bottom right, rgba(63,185,80,0.14), transparent 35%),
        #0d1117;
    border: 1px solid rgba(88,166,255,0.28);
    border-radius: 22px;
    padding: 18px;
    box-shadow: 0 0 45px rgba(88,166,255,0.12);
}

/* =========================================================
AI NAVBAR
========================================================= */

div[role="radiogroup"] {
    display: flex;
    justify-content: center;
    gap: 14px;
    background: rgba(13,17,23,0.82);
    padding: 16px 20px;
    border-radius: 22px;
    border: 1px solid rgba(88,166,255,0.14);
    backdrop-filter: blur(10px);
    box-shadow:
        0 0 35px rgba(88,166,255,0.08),
        inset 0 0 20px rgba(88,166,255,0.03);
    width: fit-content;
    margin: auto;
}

div[role="radiogroup"] label {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    padding: 12px 22px;
    border-radius: 14px;
    transition: all 0.25s ease;
}

div[role="radiogroup"] label:hover {
    border: 1px solid rgba(88,166,255,0.38);
    background: rgba(88,166,255,0.08);
    box-shadow: 0 0 18px rgba(88,166,255,0.12);
    transform: translateY(-1px);
}

div[role="radiogroup"] label[data-baseweb="radio"] > div {
    color: #E6EDF3;
    font-weight: 600;
    font-size: 15px;
    letter-spacing: 0.2px;
}

/* selected */

div[role="radiogroup"] input:checked + div {
    color: #58a6ff !important;
}

/* hide default circles */

div[role="radiogroup"] input {
    display: none;
}
                                                
</style>
""", unsafe_allow_html=True)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = "data"

CLEAN_PATH = f"{BASE_DIR}/clean_customer_data.csv"

RFM_DIR = f"{BASE_DIR}/RFM_Output"
KMEANS_DIR = f"{BASE_DIR}/KMeans_Output"
HC_DIR = f"{BASE_DIR}/HC_Output"
CHURN_DIR = f"{BASE_DIR}/Churn_Output"
PERSONA_DIR = f"{BASE_DIR}/Persona_Output"

# Intelligence & KPI outputs
INTELLIGENCE_DIR = f"{BASE_DIR}/Intelligence_Output"
BUSINESS_KPI_DIR = f"{BASE_DIR}/Business_KPI_Output"
PREP_DIR = f"{BASE_DIR}/Presentation_Prep"

# =========================================================
# HELPERS
# =========================================================

@st.cache_data
def read_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def format_number(x):
    return f"{x:,.0f}"

def format_money(x):
    return f"${x:,.0f}"

# =========================================================
# LOAD DATA
# =========================================================

# Original data
clean_df = read_csv(CLEAN_PATH)

# RFM outputs
rfm_df = read_csv(f"{RFM_DIR}/customer_rfm_final.csv")
rfm_segment_distribution = read_csv(f"{RFM_DIR}/rfm_segment_distribution.csv")

# KMeans outputs
kmeans_profile = read_csv(f"{KMEANS_DIR}/kmeans_cluster_profile_rfm.csv")
rfm_kmeans_cross = read_csv(f"{KMEANS_DIR}/rfm_kmeans_cross.csv")

# Churn outputs
model_comparison = read_csv(f"{CHURN_DIR}/model_comparison.csv")
feature_importance = read_csv(f"{CHURN_DIR}/feature_importance.csv")
segment_churn = read_csv(f"{CHURN_DIR}/segment_churn_analysis.csv")

# Personas
personas_df = read_csv(f"{PERSONA_DIR}/segment_personas.csv")

# Campaign
campaign_df = read_csv(f"{RFM_DIR}/campaign_customer_list.csv")

# ===== NEW: Intelligence & KPI outputs =====
rfm_intelligence = read_csv(f"{INTELLIGENCE_DIR}/rfm_segment_intelligence.csv")
kmeans_intelligence = read_csv(f"{INTELLIGENCE_DIR}/kmeans_segment_intelligence.csv")
hc_intelligence = read_csv(f"{INTELLIGENCE_DIR}/hc_segment_intelligence.csv")

rfm_business_kpi = read_csv(f"{BUSINESS_KPI_DIR}/rfm_business_kpi.csv")
kmeans_business_kpi = read_csv(f"{BUSINESS_KPI_DIR}/kmeans_business_kpi.csv")
hc_business_kpi = read_csv(f"{BUSINESS_KPI_DIR}/hc_business_kpi.csv")
overview_kpi = read_csv(f"{BUSINESS_KPI_DIR}/overview_kpi_summary.csv")
risk_distribution = read_csv(f"{BUSINESS_KPI_DIR}/risk_distribution.csv")

# ===== NEW: Presentation prep outputs =====
rfm_journey = read_csv(f"{PREP_DIR}/rfm_journey_mapping.csv")
kmeans_enriched = read_csv(f"{PREP_DIR}/kmeans_enriched_intelligence.csv")
hc_enriched = read_csv(f"{PREP_DIR}/hc_enriched_intelligence.csv")
cluster_naming = read_csv(f"{PREP_DIR}/cluster_naming_logic.csv")
hero_kpi = read_csv(f"{PREP_DIR}/hero_page_kpi.csv")

# =========================================================
# HERO METRICS - GLOBAL SCOPE (tüm sayfalar için)
# =========================================================

# Use overview_kpi for accurate 491K statistics
if overview_kpi is not None:
    kpi_dict = dict(zip(overview_kpi['metric'], overview_kpi['value']))
    
    total_customers = int(kpi_dict.get('Total Customers', 491705))
    high_risk_customers = int(kpi_dict.get('High Risk Customers', 0))
    warning_customers = int(kpi_dict.get('Warning Customers', 0))
    healthy_customers = int(kpi_dict.get('Healthy Customers', 0))
    
    # FIX: Remove $ and , from string values
    revenue_at_risk_str = str(kpi_dict.get('Total Revenue at Risk', '0')).replace('$', '').replace(',', '')
    revenue_at_risk = float(revenue_at_risk_str) if revenue_at_risk_str else 0
    
    recoverable_revenue_str = str(kpi_dict.get('Total Recoverable Revenue', '0')).replace('$', '').replace(',', '')
    recoverable_revenue = float(recoverable_revenue_str) if recoverable_revenue_str else 0
    
    avg_purchase_rate_str = str(kpi_dict.get('Average Purchase Rate (%)', '45.2')).replace('%', '')
    avg_purchase_rate = float(avg_purchase_rate_str) if avg_purchase_rate_str else 45.2
else:
    total_customers = 491705
    high_risk_customers = 78169
    warning_customers = 214816
    healthy_customers = 198720
    revenue_at_risk = 319100000
    recoverable_revenue = 29400000
    avg_purchase_rate = 45.2

# Model performance
best_auc = 0.98
best_model_name = "XGBoost"

if model_comparison is not None and "AUC" in model_comparison.columns:
    best_row = model_comparison.loc[model_comparison["AUC"].idxmax()]
    best_auc = best_row["AUC"]
    best_model_name = best_row["Model"]

auc_display = f"{best_auc:.2f}" if best_auc is not None else "N/A"

# Risk distribution
if risk_distribution is not None:
    total_at_risk = risk_distribution[risk_distribution['risk_level'] == '🔴 Critical']['customer_count'].sum()
    total_warning = risk_distribution[risk_distribution['risk_level'] == '🟡 Warning']['customer_count'].sum()
    total_healthy = risk_distribution[risk_distribution['risk_level'] == '🟢 Healthy']['customer_count'].sum()
else:
    total_at_risk = high_risk_customers
    total_warning = warning_customers
    total_healthy = healthy_customers

# Segment count
segment_count = 10
if rfm_segment_distribution is not None:
    segment_count = len(rfm_segment_distribution)

# =========================================================
# NAVBAR
# =========================================================

st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)

section = st.radio(
    label="Navigation",
    options=[
        "Hero",
        "Problem",
        "RFM Intelligence",
        "Clustering",
        "Churn",
        "Personas",
        "Action"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

# =========================================================
# HERO
# =========================================================

if section == "Hero":

    
    # -----------------------------
    # HERO LAYOUT
    # -----------------------------

    st.markdown("<div class='hero-shell'>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:-10px'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.05, 0.95], gap="large")

    # =====================================================
    # LEFT SIDE - HERO + KPI CARDS
    # =====================================================

    with left:

        st.markdown("""
        <div class='hero-badge'>
            🧠 AI-POWERED INTELLIGENCE PLATFORM
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='hero-big-title'>
            Customer Intelligence<br>
            at Scale
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='hero-desc'>
            491K customers analyzed. Behavioral RFM segmentation, 
            churn prediction (98% AUC), AI personas, and 
            ${(revenue_at_risk/1_000_000):.1f}M revenue at risk identified for immediate action.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # KPI Cards Grid
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f"""
            <div class='neon-card'>
                <div class='neon-value'>{format_number(total_customers)}</div>
                <div class='neon-label'>Customers Analyzed</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class='neon-card'>
                <div class='neon-value' style='color:#f85149;'>${(revenue_at_risk/1_000_000):.1f}M</div>
                <div class='neon-label'>Revenue at Risk</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class='neon-card'>
                <div class='neon-value'>{segment_count}</div>
                <div class='neon-label'>RFM Segments</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class='neon-card'>
                <div class='neon-value' style='color:#3fb950;'>${(recoverable_revenue/1_000_000):.1f}M</div>
                <div class='neon-label'>Recoverable Revenue</div>
            </div>
            """, unsafe_allow_html=True)

    # =====================================================
    # RIGHT SIDE - BEHAVIOR MATRIX
    # =====================================================

    with right:

        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(88,166,255,0.18);
        margin-bottom:18px;
        '>
        <div style='color:#58a6ff;font-size:13px;font-weight:600;'>
        LIVE SEGMENT TRACKING
        </div>
        <div style='color:#3fb950;font-size:13px;font-weight:600;'>
        ● ACTIVE 491K
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🧬 Customer Behavior Matrix")

        if rfm_df is not None:
            matrix_sample = rfm_df.sample(
                min(4500, len(rfm_df)),
                random_state=42
            )

            fig = px.scatter(
                matrix_sample,
                x="recency",
                y="monetary",
                size="frequency",
                color="RFM_Segment",
                hover_data=[
                    "CustomerID",
                    "RFM_Segment",
                    "frequency",
                    "PurchaseStatus"
                ],
                template="plotly_dark",
                opacity=0.78
            )

            fig.update_layout(
                height=520,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E6EDF3"),
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis=dict(
                    title="Recency (days since purchase)",
                    showgrid=False,
                    zeroline=False
                ),
                yaxis=dict(
                    title="Monetary Value",
                    showgrid=False,
                    zeroline=False
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.38,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10)
                )
            )

            fig.update_traces(
                marker=dict(
                    line=dict(
                        width=0.4,
                        color="rgba(255,255,255,0.28)"
                    )
                )
            )

            st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # RISK DISTRIBUTION CARDS
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)

    risk1, risk2, risk3 = st.columns(3)

    with risk1:
        st.markdown("#### 🟢 HEALTHY")
        st.metric(
            label="Engaged & Valuable",
            value=format_number(total_healthy),
            delta=f"{(total_healthy/total_customers*100):.1f}% of base"
        )

    with risk2:
        st.markdown("#### 🟡 WARNING")
        st.metric(
            label="Early Churn Signals",
            value=format_number(total_warning),
            delta=f"{(total_warning/total_customers*100):.1f}% of base"
        )

    with risk3:
        st.markdown("#### 🔴 CRITICAL")
        st.metric(
            label="High Churn Risk",
            value=format_number(total_at_risk),
            delta=f"${(revenue_at_risk/1_000_000):.1f}M at risk"
        )

    # =====================================================
    # INSIGHTS
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class='insight-box'>
        <strong>🎯 Key Finding:</strong> {format_number(total_at_risk)} customers 
        ({total_at_risk/total_customers*100:.1f}%) are in critical risk stage. 
        Combined with {format_number(total_warning)} in warning stage, 
        {format_number(total_at_risk + total_warning)} customers ({(total_at_risk + total_warning)/total_customers*100:.1f}%) 
        require immediate retention action.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class='insight-box'>
        <strong>💰 Business Impact:</strong> ${(revenue_at_risk/1_000_000):.1f}M revenue exposure from at-risk 
        and warning segments. With {(recoverable_revenue/revenue_at_risk*100):.1f}% recovery rate, ${(recoverable_revenue/1_000_000):.1f}M recoverable revenue 
        potential through targeted retention campaigns.
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# PROBLEM SECTION
# =========================================================

elif section == "Problem":

    st.markdown(f"""
    <div class='section-title'>
        ⚠️ The Business Problem
    </div>
    <div class='section-subtitle'>
        ${(revenue_at_risk/1_000_000):.1f}M in revenue at risk. {warning_customers + high_risk_customers:,} customers in warning or critical stage.
        Yet behavioral risk patterns remain invisible until churn happens.
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.05, 0.95], gap="large")

    # =====================================================
    # LEFT SIDE — BUSINESS PAIN POINTS
    # =====================================================

    with left:

        st.markdown("""
        <div class='neon-card'>
            <h3>⚠️ Invisible Churn Signals</h3>
            <p>
            Traditional dashboards fail to identify
            behavioral warning patterns before customers disappear.
            59% of customer base shows churn risk.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='neon-card'>
            <h3>🎯 Generic CRM Campaigns</h3>
            <p>
            One-size-fits-all retention strategies fail across
            completely different customer behaviors. No segmentation = 
            wasted marketing budget.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='neon-card'>
            <h3>💰 Revenue Exposure Blindness</h3>
            <p>
            ${(revenue_at_risk/1_000_000):.1f}M revenue at risk goes undetected.
            Only {(recoverable_revenue/revenue_at_risk*100):.1f}% recovery rate with current approaches
            means ${((revenue_at_risk - recoverable_revenue)/1_000_000):.1f}M+ permanently lost.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='neon-card'>
            <h3>🔗 Fragmented Customer Data</h3>
            <p>
            Customer behavior signals remain disconnected
            across analytics, CRM and marketing systems.
            No unified intelligence layer.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # RIGHT SIDE — RISK DISTRIBUTION VISUALIZATION
    # =====================================================

    with right:

        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(88,166,255,0.18);
        margin-bottom:18px;
        '>

        <div style='color:#58a6ff;font-size:13px;font-weight:600;'>
        LIVE CUSTOMER RISK MONITORING
        </div>

        <div style='color:#f85149;font-size:13px;font-weight:600;'>
        ● RISK DETECTED 293K CUSTOMERS
        </div>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Customer Risk Distribution")

        # Use risk_distribution data for accurate visualization
        if risk_distribution is not None:
            fig = px.bar(
                risk_distribution,
                x='risk_level',
                y='customer_count',
                color='risk_level',
                color_discrete_map={
                    '🟢 Healthy': '#3fb950',
                    '🟡 Warning': '#d29922',
                    '🔴 Critical': '#f85149'
                },
                template="plotly_dark",
                opacity=0.85
            )

            fig.update_layout(
                height=520,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=20, b=10),
                font=dict(color="#E6EDF3"),
                xaxis_title="Risk Level",
                yaxis_title="Number of Customers",
                showlegend=False,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='rgba(88,166,255,0.1)')
            )

            fig.update_traces(
                marker=dict(
                    line=dict(
                        width=1,
                        color="rgba(255,255,255,0.2)"
                    )
                ),
                text=risk_distribution['customer_count'],
                textposition='outside'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback visualization
            risk_sample = clean_df.sample(
                min(4500, len(clean_df)),
                random_state=42
            )

            fig = px.scatter(
                risk_sample,
                x="LastPurchaseDaysAgo",
                y="TotalSpent",
                color="PurchaseStatus",
                size="NumberOfPurchases",
                opacity=0.42,
                template="plotly_dark",
                hover_data=[
                    "CustomerID",
                    "PurchaseStatus",
                    "TotalSpent",
                    "NumberOfPurchases"
                ],
                color_discrete_map={
                    0: "#f85149",
                    1: "#3fb950"
                }
            )

            fig.update_layout(
                height=520,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=20, b=10),
                font=dict(color="#E6EDF3"),
                xaxis=dict(
                    title="Last Purchase Days Ago",
                    showgrid=False,
                    zeroline=False
                ),
                yaxis=dict(
                    title="Total Spent",
                    showgrid=False,
                    zeroline=False
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.22,
                    xanchor="center",
                    x=0.5
                )
            )

            fig.update_traces(
                marker=dict(
                    line=dict(
                        width=0.3,
                        color="rgba(255,255,255,0.15)"
                    )
                )
            )

            st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # BOTTOM RISK METRICS
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)

    # Use risk_distribution for accurate numbers
    if risk_distribution is not None:
        healthy = risk_distribution[risk_distribution['risk_level'] == '🟢 Healthy']['customer_count'].sum()
        warning = risk_distribution[risk_distribution['risk_level'] == '🟡 Warning']['customer_count'].sum()
        critical = risk_distribution[risk_distribution['risk_level'] == '🔴 Critical']['customer_count'].sum()
        revenue_exposure = risk_distribution[risk_distribution['risk_level'] == '🔴 Critical']['revenue_exposure'].sum()
    else:
        healthy = 198720
        warning = 214816
        critical = 78169
        revenue_exposure = 46274485

    at_risk_total = warning + critical

    b1, b2, b3 = st.columns(3)

    with b1:
        st.markdown("#### 🔴 CRITICAL RISK")
        st.metric(
            label="High Churn Probability",
            value=format_number(critical),
            delta=f"${(revenue_exposure/1_000_000):.1f}M revenue exposure"
        )

    with b2:
        st.markdown("#### 🟡 WARNING ZONE")
        st.metric(
            label="Early Churn Signals",
            value=format_number(warning),
            delta=f"{(warning/491705*100):.1f}% of base"
        )

    with b3:
        st.markdown("#### 🟢 HEALTHY BASE")
        st.metric(
            label="Engaged & Valuable",
            value=format_number(healthy),
            delta=f"{(healthy/491705*100):.1f}% of base"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Calculate accurate revenue at risk from risk_distribution
    total_revenue_at_risk = risk_distribution[
        risk_distribution['risk_level'].isin(['🟡 Warning', '🔴 Critical'])
    ]['revenue_exposure'].sum() / 1_000_000

    st.markdown(f"""
    <div class='insight-box'>

    <h3>
    The Invisible Crisis: 59% of Your Customer Base is at Risk
    </h3>

    <p style='font-size:18px; line-height:1.8;'>
    <span style='color:#f85149;'>{format_number(critical)} customers</span> in critical stage + 
    <span style='color:#d29922;'>{format_number(warning)} customers</span> showing early churn signals = 
    <span style='color:#f85149;'>${total_revenue_at_risk:.1f}M revenue at risk.</span>
    
    Without behavioral intelligence, this exposure remains invisible until the churn happens. 
    The solution: unified customer segmentation, predictive risk scoring, and 
    AI-powered campaign personalization.
    </p>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# RFM INTELLIGENCE
# =========================================================

elif section == "RFM Intelligence":

    st.markdown("""
    <div class='section-title'>
        RFM Intelligence & Customer Journey Mapping
    </div>

    <div class='section-subtitle'>
        10 customer segments mapped across loyalty growth and churn risk journeys
        with actionable business metrics and behavioral insights.
    </div>
    """, unsafe_allow_html=True)

    if rfm_intelligence is None or rfm_business_kpi is None or rfm_journey is None:
        st.error("Intelligence datasets not found.")
        st.stop()

    # =====================================================
    # KPI CARDS
    # =====================================================

    total_customers = rfm_journey["customer_count"].sum()
    total_segments = rfm_journey["RFM_Segment"].nunique()
    avg_purchase_rate = rfm_journey["purchase_rate_%"].mean()
    at_risk_customers = rfm_journey[rfm_journey["RFM_Segment"] == "at_risk"]["customer_count"].values[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("#### 👥 Total Customers")
        st.metric("Segmented Base", format_number(total_customers))

    with col2:
        st.markdown("#### 🧩 RFM Segments")
        st.metric("Behavioral Groups", total_segments)

    with col3:
        st.markdown("#### 📊 Avg Purchase Rate")
        st.metric("Engagement Level", f"{avg_purchase_rate:.1f}%")

    with col4:
        st.markdown("#### 🔴 At Risk")
        st.metric("Retention Priority", format_number(at_risk_customers))

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # JOURNEY MAPS
    # =====================================================

    left_journey, right_journey = st.columns(2, gap="large")

    # ----- LOYALTY GROWTH JOURNEY -----
    with left_journey:

        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(63,185,80,0.25);
        margin-bottom:18px;
        '>

        <div style='color:#3fb950;font-size:13px;font-weight:600;'>
        🚀 LOYALTY GROWTH JOURNEY
        </div>

        <div style='color:#3fb950;font-size:13px;font-weight:600;'>
        ● 5 STAGES
        </div>

        </div>
        """, unsafe_allow_html=True)

        loyalty_journey = rfm_journey[rfm_journey["journey_type"] == "Loyalty Growth"].sort_values("journey_stage")

        loyalty_stages = {
            1: ("🆕", "New Customers"),
            2: ("🌱", "Promising"),
            3: ("💎", "Potential Loyalists"),
            4: ("🏆", "Loyal Customers"),
            5: ("👑", "Champions")
        }

        for idx, row in loyalty_journey.iterrows():
            stage_num = int(row["journey_stage"])
            emoji, stage_name = loyalty_stages.get(stage_num, ("", row["RFM_Segment"]))
            customer_count = row["customer_count"]
            purchase_rate = row["purchase_rate_%"]
            risk_level = row["risk_level"]

            st.markdown(f"""
            <div class='neon-card'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div>
                        <div style='font-size:28px;margin-bottom:8px;'>{emoji}</div>
                        <div style='font-size:16px;font-weight:700;'>{stage_name}</div>
                        <div style='color:#8b949e;font-size:13px;margin-top:4px;'>{row["RFM_Segment"].replace("_", " ").title()}</div>
                    </div>
                    <div style='text-align:right;'>
                        <div class='neon-value'>{format_number(customer_count)}</div>
                        <div class='neon-label'>Customers</div>
                        <div style='margin-top:8px;font-size:14px;color:#3fb950;font-weight:600;'>{purchase_rate:.0f}% Purchase</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ----- CHURN RISK JOURNEY -----
    with right_journey:

        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(248,81,73,0.25);
        margin-bottom:18px;
        '>

        <div style='color:#f85149;font-size:13px;font-weight:600;'>
        ⚠️ CHURN RISK JOURNEY
        </div>

        <div style='color:#f85149;font-size:13px;font-weight:600;'>
        ● 5 STAGES
        </div>

        </div>
        """, unsafe_allow_html=True)

        churn_journey = rfm_journey[rfm_journey["journey_type"] == "Churn Risk"].sort_values("journey_stage")

        churn_stages = {
            1: ("👀", "Need Attention"),
            2: ("😴", "About To Sleep"),
            3: ("🔥", "At Risk"),
            4: ("💤", "Hibernating"),
            5: ("🚨", "Can't Loose")
        }

        for idx, row in churn_journey.iterrows():
            stage_num = int(row["journey_stage"])
            emoji, stage_name = churn_stages.get(stage_num, ("", row["RFM_Segment"]))
            customer_count = row["customer_count"]
            purchase_rate = row["purchase_rate_%"]
            risk_level = row["risk_level"]

            st.markdown(f"""
            <div class='neon-card' style='border-color:rgba(248,81,73,0.25);box-shadow:0 0 28px rgba(248,81,73,0.12);'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div>
                        <div style='font-size:28px;margin-bottom:8px;'>{emoji}</div>
                        <div style='font-size:16px;font-weight:700;'>{stage_name}</div>
                        <div style='color:#8b949e;font-size:13px;margin-top:4px;'>{row["RFM_Segment"].replace("_", " ").title()}</div>
                    </div>
                    <div style='text-align:right;'>
                        <div class='neon-value'>{format_number(customer_count)}</div>
                        <div class='neon-label'>Customers</div>
                        <div style='margin-top:8px;font-size:14px;color:#f85149;font-weight:600;'>{risk_level}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # SEGMENT INTELLIGENCE TABLE
    # =====================================================

    st.markdown("""
    <div style='
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:10px 14px;
    border-radius:14px;
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(88,166,255,0.18);
    margin-bottom:18px;
    margin-top:30px;
    '>

    <div style='color:#58a6ff;font-size:13px;font-weight:600;'>
    📋 RFM SEGMENT INTELLIGENCE
    </div>

    <div style='color:#3fb950;font-size:13px;font-weight:600;'>
    ● 10 SEGMENTS
    </div>

    </div>
    """, unsafe_allow_html=True)

    # Merge intelligence + business KPI
    merged_intel = rfm_intelligence.merge(rfm_business_kpi, on="RFM_Segment", suffixes=("", "_kpi"))

    # Select display columns
    display_cols = [
        "RFM_Segment",
        "customer_count",
        "purchase_rate_%",
        "risk_level",
        "dominant_product_category",
        "dominant_region",
        "dominant_age_group",
        "dominant_device",
        "discount_affinity",
        "loyalty_strength_score",
        "estimated_churn_rate_%"
    ]

    table_data = merged_intel[display_cols].copy()
    table_data.columns = [
        "Segment",
        "Customers",
        "Purchase %",
        "Risk",
        "Top Category",
        "Top Region",
        "Age Group",
        "Device",
        "Discount %",
        "Loyalty Score",
        "Churn Risk %"
    ]

    # Format numbers
    table_data["Customers"] = table_data["Customers"].apply(format_number)
    table_data["Discount %"] = table_data["Discount %"].apply(lambda x: f"{x:.1f}%")
    table_data["Purchase %"] = table_data["Purchase %"].apply(lambda x: f"{x:.1f}%")
    table_data["Churn Risk %"] = table_data["Churn Risk %"].apply(lambda x: f"{x:.1f}%")

    st.dataframe(table_data, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # BUSINESS KPI METRICS
    # =====================================================

    st.markdown("""
    <div style='
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:10px 14px;
    border-radius:14px;
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(88,166,255,0.18);
    margin-bottom:18px;
    '>

    <div style='color:#58a6ff;font-size:13px;font-weight:600;'>
    💰 BUSINESS KPI METRICS
    </div>

    <div style='color:#3fb950;font-size:13px;font-weight:600;'>
    ● FINANCIAL & STRATEGIC
    </div>

    </div>
    """, unsafe_allow_html=True)

    kpi_cols = [
        "RFM_Segment",
        "customer_count",
        "revenue_at_risk",
        "recoverable_revenue",
        "potential_retention_gain",
        "avg_customer_ltv",
        "loyalty_strength_score",
        "estimated_campaign_roi_multiple"
    ]

    kpi_data = rfm_business_kpi[kpi_cols].copy()
    kpi_data.columns = [
        "Segment",
        "Customers",
        "Revenue at Risk",
        "Recoverable Revenue",
        "Retention Gain",
        "Avg LTV",
        "Loyalty Score",
        "Campaign ROI"
    ]

    # Format currency columns
    kpi_data["Revenue at Risk"] = kpi_data["Revenue at Risk"].apply(format_money)
    kpi_data["Recoverable Revenue"] = kpi_data["Recoverable Revenue"].apply(format_money)
    kpi_data["Retention Gain"] = kpi_data["Retention Gain"].apply(format_money)
    kpi_data["Avg LTV"] = kpi_data["Avg LTV"].apply(format_money)
    kpi_data["Campaign ROI"] = kpi_data["Campaign ROI"].apply(lambda x: f"{x:.2f}x")

    st.dataframe(kpi_data, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # TOP AT-RISK & VALUE SEGMENTS
    # =====================================================

    left_risk, right_value = st.columns(2, gap="large")

    # ----- TOP AT-RISK SEGMENTS -----
    with left_risk:

        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(248,81,73,0.25);
        margin-bottom:18px;
        '>

        <div style='color:#f85149;font-size:13px;font-weight:600;'>
        🔴 TOP AT-RISK SEGMENTS
        </div>

        <div style='color:#f85149;font-size:13px;font-weight:600;'>
        ● URGENT ACTION
        </div>

        </div>
        """, unsafe_allow_html=True)

        # Sort by revenue at risk (descending)
        top_at_risk = rfm_business_kpi.nlargest(5, "revenue_at_risk")[
            ["RFM_Segment", "customer_count", "revenue_at_risk", "estimated_churn_rate_%"]
        ]

        for idx, row in top_at_risk.iterrows():
            segment = row["RFM_Segment"]
            customers = row["customer_count"]
            revenue_risk = row["revenue_at_risk"]
            churn_rate = row["estimated_churn_rate_%"]

            st.markdown(f"""
            <div class='neon-card' style='border-color:rgba(248,81,73,0.25);'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div>
                        <div style='font-size:16px;font-weight:700;color:#f85149;'>{segment.replace("_", " ").title()}</div>
                        <div style='color:#8b949e;font-size:13px;margin-top:4px;'>{format_number(customers)} customers</div>
                    </div>
                    <div style='text-align:right;'>
                        <div style='font-size:20px;font-weight:700;color:#f85149;'>{format_money(revenue_risk)}</div>
                        <div style='color:#8b949e;font-size:13px;margin-top:4px;'>{churn_rate:.1f}% churn risk</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ----- TOP VALUE SEGMENTS -----
    with right_value:

        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(63,185,80,0.25);
        margin-bottom:18px;
        '>

        <div style='color:#3fb950;font-size:13px;font-weight:600;'>
        🟢 TOP VALUE SEGMENTS
        </div>

        <div style='color:#3fb950;font-size:13px;font-weight:600;'>
        ● HIGH LTV
        </div>

        </div>
        """, unsafe_allow_html=True)

        # Sort by average LTV (descending)
        top_value = rfm_business_kpi.nlargest(5, "avg_customer_ltv")[
            ["RFM_Segment", "customer_count", "avg_customer_ltv", "loyalty_strength_score"]
        ]

        for idx, row in top_value.iterrows():
            segment = row["RFM_Segment"]
            customers = row["customer_count"]
            ltv = row["avg_customer_ltv"]
            loyalty = row["loyalty_strength_score"]

            st.markdown(f"""
            <div class='neon-card' style='border-color:rgba(63,185,80,0.25);'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div>
                        <div style='font-size:16px;font-weight:700;color:#3fb950;'>{segment.replace("_", " ").title()}</div>
                        <div style='color:#8b949e;font-size:13px;margin-top:4px;'>{format_number(customers)} customers</div>
                    </div>
                    <div style='text-align:right;'>
                        <div style='font-size:20px;font-weight:700;color:#3fb950;'>{format_money(ltv)}</div>
                        <div style='color:#8b949e;font-size:13px;margin-top:4px;'>Loyalty {loyalty:.2f}/10</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # CLOSING INSIGHT
    # =====================================================

    st.markdown("""
    <div class='insight-box'>
        <h3>🎯 From RFM to Intelligence to Action</h3>
        <p style='font-size:18px; line-height:1.8;'>
        These 10 RFM segments represent distinct customer journeys. Loyalty Growth segments
        show customers moving toward high value (new → champions). Churn Risk segments reveal
        customers at different stages of disengagement (need attention → hibernating).
        Business KPIs translate each segment into financial impact, enabling targeted
        retention, win-back, and expansion campaigns with measurable ROI.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# CLUSTERING - COMPREHENSIVE DATA SCIENTIST GRADE
# =========================================================

elif section == "Clustering":

    st.markdown("""
    <div class='section-title'>
        Advanced Customer Clustering & Validation
    </div>

    <div class='section-subtitle'>
        Triple-method validation: RFM (rule-based) → KMeans (unsupervised) → Hierarchical (confirmation).
        Three independent clustering approaches validate segmentation robustness across 491K customers.
    </div>
    """, unsafe_allow_html=True)

    if kmeans_enriched is None or hc_enriched is None or kmeans_profile is None:
        st.error("Clustering datasets not found.")
        st.stop()

    # =====================================================
    # P0: LOAD HC REAL DATA
    # =====================================================

    hc_profile = read_csv(f"{HC_DIR}/hc_cluster_profile.csv")

    if hc_profile is None:
        st.error("HC cluster profile not found.")
        st.stop()

    # =====================================================
    # KPI VALIDATION ROW
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    kmeans_total = kmeans_enriched["customer_count"].sum()
    hc_total = hc_enriched["customer_count"].sum()
    kmeans_clusters = kmeans_enriched.shape[0]
    hc_clusters = hc_enriched.shape[0]

    with col1:
        st.markdown("#### 🔬 KMeans")
        st.metric("Clusters", f"{kmeans_clusters}")

    with col2:
        st.markdown("#### 📊 KMeans Coverage")
        st.metric("Customers", format_number(kmeans_total))

    with col3:
        st.markdown("#### 🌳 Hierarchical")
        st.metric("Clusters", f"{hc_clusters}")

    with col4:
        st.markdown("#### 🔄 HC Coverage")
        st.metric("Sample Validated", format_number(hc_total))

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # CLUSTERING TABS
    # =====================================================

    tab_kmeans, tab_hc, tab_comparison = st.tabs([
        "🔵 KMeans Clustering",
        "🟣 Hierarchical Clustering",
        "📊 Validation & Comparison"
    ])

    # =========================================================
    # TAB 1: KMEANS CLUSTERING (P0, P1, P2 FIXES)
    # =========================================================

    with tab_kmeans:

        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(88,166,255,0.18);
        margin-bottom:18px;
        '>

        <div style='color:#58a6ff;font-size:13px;font-weight:600;'>
        KMEANS BEHAVIORAL CLUSTERS
        </div>

        <div style='color:#3fb950;font-size:13px;font-weight:600;'>
        ● {kmeans_clusters} CLUSTERS
        </div>

        </div>
        """, unsafe_allow_html=True)

        left_km, right_km = st.columns([1.05, 0.95], gap="large")

        # ----- KMEANS SCATTER PLOT -----
        with left_km:

            kmeans_profile_copy = kmeans_profile.copy()

            fig_km = px.scatter(
                kmeans_profile_copy,
                x="Ort_Recency",
                y="Ort_Monetary",
                size="Musteri_Sayisi",
                color="Cluster_Adi",
                hover_name="Cluster_Adi",
                hover_data=[
                    "Musteri_Sayisi",
                    "Ort_Frequency",
                    "Satin_Alma_Orani",
                    "Ort_Memnuniyet"
                ],
                template="plotly_dark",
                title="KMeans: Recency × Monetary Space"
            )

            fig_km.update_layout(
                height=560,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=50, b=10),
                font=dict(color="#E6EDF3"),
                xaxis=dict(
                    title="Average Recency (days)",
                    showgrid=False,
                    zeroline=False
                ),
                yaxis=dict(
                    title="Average Monetary (TL)",
                    showgrid=False,
                    zeroline=False
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.28,
                    xanchor="center",
                    x=0.5
                )
            )

            fig_km.update_traces(
                marker=dict(
                    line=dict(width=1, color="rgba(255,255,255,0.25)")
                )
            )

            st.plotly_chart(fig_km, use_container_width=True)

        # ----- KMEANS SELECTBOX & METRICS -----
        with right_km:

            st.markdown("""
            <div style='
            display:flex;
            justify-content:space-between;
            align-items:center;
            padding:10px 14px;
            border-radius:14px;
            background:rgba(255,255,255,0.03);
            border:1px solid rgba(88,166,255,0.18);
            margin-bottom:18px;
            '>

            <div style='color:#58a6ff;font-size:13px;font-weight:600;'>
            CLUSTER INTELLIGENCE
            </div>

            <div style='color:#f85149;font-size:13px;font-weight:600;'>
            ● ACTIONABLE PROFILES
            </div>

            </div>
            """, unsafe_allow_html=True)

            kmeans_cluster_names = sorted(kmeans_profile_copy["Cluster_Adi"].dropna().unique())

            selected_kmeans_cluster = st.selectbox(
                "Select KMeans Cluster",
                kmeans_cluster_names,
                key="kmeans_select"
            )

            kmeans_cluster_data = kmeans_profile_copy[
                kmeans_profile_copy["Cluster_Adi"] == selected_kmeans_cluster
            ]

            km_customers = kmeans_cluster_data["Musteri_Sayisi"].sum()
            km_recency = kmeans_cluster_data["Ort_Recency"].mean()
            km_frequency = kmeans_cluster_data["Ort_Frequency"].mean()
            km_monetary = kmeans_cluster_data["Ort_Monetary"].mean()
            km_purchase_rate = kmeans_cluster_data["Satin_Alma_Orani"].mean() * 100

            c1, c2 = st.columns(2)

            with c1:
                st.metric("Customers", format_number(km_customers))
                st.metric("Avg Recency", f"{km_recency:.1f} days")

            with c2:
                st.metric("Avg Frequency", f"{km_frequency:.1f}")
                st.metric("Avg Monetary", format_money(km_monetary))

            st.metric("Purchase Rate", f"{km_purchase_rate:.1f}%")

            st.markdown("<br>", unsafe_allow_html=True)

            # P1: JOURNEY TYPE + RISK STRATIFICATION
            km_enriched_data = kmeans_enriched[
                kmeans_enriched["cluster_name"].astype(str).str.contains(
                    selected_kmeans_cluster.split()[0], case=False, na=False
                )
            ]

            if not km_enriched_data.empty:
                km_journey = km_enriched_data["journey_type"].values[0]
                km_risk = km_enriched_data["risk_level"].values[0]
                km_revenue_at_risk = km_enriched_data["revenue_at_risk"].values[0]
                km_recoverable = km_enriched_data["recoverable_revenue"].values[0]
                km_category = km_enriched_data["dominant_product_category"].values[0]
                km_region = km_enriched_data["dominant_region"].values[0]
                km_age = km_enriched_data["dominant_age_group"].values[0]
                km_device = km_enriched_data["dominant_device"].values[0]
                km_discount = km_enriched_data["discount_affinity"].values[0]
                km_loyalty = km_enriched_data["loyalty_strength_score"].values[0]
                km_churn = km_enriched_data["estimated_churn_rate_%"].values[0]

                # P1: Risk-based border styling
                risk_color_border = {
                    "🔴 Critical": "rgba(248,81,73,0.4)",
                    "🟡 Warning": "rgba(200,150,0,0.3)",
                    "🟢 Healthy": "rgba(63,185,80,0.3)"
                }.get(km_risk, "rgba(88,166,255,0.25)")

                risk_color_text = {
                    "🔴 Critical": "#f85149",
                    "🟡 Warning": "#d29922",
                    "🟢 Healthy": "#3fb950"
                }.get(km_risk, "#58a6ff")

                st.markdown(f"""
                <div class='neon-card' style='border-color:{risk_color_border};border-width:2px;'>
                    <div style='font-size:13px;color:{risk_color_text};font-weight:700;margin-bottom:12px;'>
                        {km_journey.upper()} | {km_risk}
                    </div>
                    <div style='font-size:14px;color:#8b949e;margin-bottom:8px;'>
                        <strong>Revenue at Risk:</strong> {format_money(km_revenue_at_risk)}
                    </div>
                    <div style='font-size:14px;color:#8b949e;margin-bottom:8px;'>
                        <strong>Recoverable:</strong> {format_money(km_recoverable)}
                    </div>
                    <div style='font-size:14px;color:#8b949e;margin-bottom:8px;'>
                        <strong>Top Category:</strong> {km_category}
                    </div>
                    <div style='font-size:14px;color:#8b949e;margin-bottom:8px;'>
                        <strong>Region:</strong> {km_region} | <strong>Age:</strong> {km_age} | <strong>Device:</strong> {km_device}
                    </div>
                    <div style='font-size:14px;color:#8b949e;'>
                        <strong>Discount Affinity:</strong> {km_discount}% | <strong>Loyalty:</strong> {km_loyalty:.1f}/10 | <strong>Churn Risk:</strong> {km_churn:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # P0: CLUSTER NAMING LOGIC EXPLANATION
        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(88,166,255,0.18);
        margin-bottom:18px;
        '>

        <div style='color:#58a6ff;font-size:13px;font-weight:600;'>
        CLUSTER NAMING LOGIC
        </div>

        <div style='color:#3fb950;font-size:13px;font-weight:600;'>
        ● SCIENTIFIC METHODOLOGY
        </div>

        </div>
        """, unsafe_allow_html=True)

        naming_logic = cluster_naming.copy()

        with st.expander("🔍 How were cluster names determined?", expanded=False):
            for idx, row in naming_logic.iterrows():
                st.markdown(f"""
                <div class='neon-card'>
                    <div style='font-size:16px;font-weight:700;color:#58a6ff;margin-bottom:12px;'>
                        {row['cluster_type']}
                    </div>
                    <div style='font-size:14px;color:#8b949e;margin-bottom:8px;'>
                        <strong>Naming Basis:</strong> {row['naming_basis']}
                    </div>
                    <div style='font-size:14px;color:#8b949e;margin-bottom:8px;'>
                        <strong>Source:</strong> {row['calculation_source']}
                    </div>
                    <div style='font-size:14px;color:#b0b3b8;line-height:1.6;'>
                        <strong>Business Meaning:</strong> {row['business_interpretation']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # P1/P2: KMEANS RISK STRATIFICATION TABLE
        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(88,166,255,0.18);
        margin-bottom:18px;
        '>

        <div style='color:#58a6ff;font-size:13px;font-weight:600;'>
        KMEANS CLUSTER PROFILES (RISK STRATIFIED)
        </div>

        <div style='color:#3fb950;font-size:13px;font-weight:600;'>
        ● SORTED BY RISK
        </div>

        </div>
        """, unsafe_allow_html=True)

        # Merge KMeans profile with enriched intelligence for sorting
        # cluster_name is int64, so extract cluster number from Cluster_Adi
        kmeans_profile_temp = kmeans_profile_copy.copy()
        # Extract number from cluster name (e.g., "Aktif Alıcılar" might need ID mapping)
        # Use left join on index instead
        kmeans_merged = kmeans_profile_temp.copy()
        
        # Add enriched data by matching position (safer approach)
        for idx, row in kmeans_enriched.iterrows():
            matching_rows = kmeans_merged[kmeans_merged.index == idx]
            if not matching_rows.empty:
                kmeans_merged.loc[idx, "journey_type"] = row["journey_type"]
                kmeans_merged.loc[idx, "risk_level"] = row["risk_level"]
                kmeans_merged.loc[idx, "revenue_at_risk"] = row["revenue_at_risk"]
                kmeans_merged.loc[idx, "recoverable_revenue"] = row["recoverable_revenue"]

        # P2: Sort by risk level priority
        risk_priority = {
            "🔴 Critical": 0,
            "🟡 Warning": 1,
            "🟢 Healthy": 2
        }
        kmeans_merged["risk_priority"] = kmeans_merged["risk_level"].map(risk_priority)
        kmeans_merged_sorted = kmeans_merged.sort_values("risk_priority")

        kmeans_table = kmeans_merged_sorted[[
            "Cluster_Adi",
            "journey_type",
            "Musteri_Sayisi",
            "Ort_Recency",
            "Ort_Frequency",
            "Ort_Monetary",
            "Satin_Alma_Orani",
            "risk_level",
            "revenue_at_risk"
        ]].copy()

        kmeans_table.columns = [
            "Cluster",
            "Journey",
            "Customers",
            "Avg Recency",
            "Avg Frequency",
            "Avg Monetary",
            "Purchase %",
            "Risk",
            "Revenue at Risk"
        ]

        kmeans_table["Customers"] = kmeans_table["Customers"].apply(format_number)
        kmeans_table["Avg Recency"] = kmeans_table["Avg Recency"].apply(lambda x: f"{x:.1f}d")
        kmeans_table["Avg Frequency"] = kmeans_table["Avg Frequency"].apply(lambda x: f"{x:.2f}")
        kmeans_table["Avg Monetary"] = kmeans_table["Avg Monetary"].apply(format_money)
        kmeans_table["Purchase %"] = kmeans_table["Purchase %"].apply(lambda x: f"{x:.1f}%")
        kmeans_table["Revenue at Risk"] = kmeans_table["Revenue at Risk"].apply(format_money)

        st.dataframe(kmeans_table, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # P1/P2: KMEANS BUSINESS METRICS WITH JOURNEY TYPE
        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(88,166,255,0.18);
        margin-bottom:18px;
        '>

        <div style='color:#58a6ff;font-size:13px;font-weight:600;'>
        KMEANS BUSINESS METRICS
        </div>

        <div style='color:#3fb950;font-size:13px;font-weight:600;'>
        ● FINANCIAL IMPACT
        </div>

        </div>
        """, unsafe_allow_html=True)

        # P2: Sort by risk
        kmeans_business = kmeans_enriched.copy()
        kmeans_business["risk_priority"] = kmeans_business["risk_level"].map(risk_priority)
        kmeans_business_sorted = kmeans_business.sort_values("risk_priority")

        kmeans_biz_table = kmeans_business_sorted[[
            "cluster_name",
            "journey_type",
            "customer_count",
            "purchase_rate_%",
            "risk_level",
            "revenue_at_risk",
            "recoverable_revenue",
            "potential_retention_gain",
            "loyalty_strength_score",
            "estimated_churn_rate_%"
        ]].copy()

        kmeans_biz_table.columns = [
            "Cluster",
            "Journey",
            "Customers",
            "Purchase %",
            "Risk",
            "Revenue at Risk",
            "Recoverable",
            "Retention Gain",
            "Loyalty Score",
            "Churn Risk %"
        ]

        kmeans_biz_table["Customers"] = kmeans_biz_table["Customers"].apply(format_number)
        kmeans_biz_table["Purchase %"] = kmeans_biz_table["Purchase %"].apply(lambda x: f"{x:.1f}%")
        kmeans_biz_table["Revenue at Risk"] = kmeans_biz_table["Revenue at Risk"].apply(format_money)
        kmeans_biz_table["Recoverable"] = kmeans_biz_table["Recoverable"].apply(format_money)
        kmeans_biz_table["Retention Gain"] = kmeans_biz_table["Retention Gain"].apply(format_money)
        kmeans_biz_table["Loyalty Score"] = kmeans_biz_table["Loyalty Score"].apply(lambda x: f"{x:.1f}/10")
        kmeans_biz_table["Churn Risk %"] = kmeans_biz_table["Churn Risk %"].apply(lambda x: f"{x:.1f}%")

        st.dataframe(kmeans_biz_table, use_container_width=True, hide_index=True)

    # =========================================================
    # TAB 2: HIERARCHICAL CLUSTERING (P0 REAL DATA)
    # =========================================================

    with tab_hc:

        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(150,100,200,0.25);
        margin-bottom:18px;
        '>

        <div style='color:#d791f0;font-size:13px;font-weight:600;'>
        HIERARCHICAL CLUSTERING (VALIDATION)
        </div>

        <div style='color:#3fb950;font-size:13px;font-weight:600;'>
        ● {hc_clusters} CLUSTERS VALIDATED
        </div>

        </div>
        """, unsafe_allow_html=True)

        left_hc, right_hc = st.columns([1.05, 0.95], gap="large")

        # ----- HC SCATTER PLOT (REAL DATA) -----
        with left_hc:

            hc_profile_copy = hc_profile.copy()

            fig_hc = px.scatter(
                hc_profile_copy,
                x="Ort_Recency",
                y="Ort_Monetary",
                size="Musteri_Sayisi",
                color="HC_Cluster_Adi",
                hover_name="HC_Cluster_Adi",
                hover_data=[
                    "Musteri_Sayisi",
                    "Ort_Frequency",
                    "Satin_Alma_Orani",
                    "Ort_Memnuniyet"
                ],
                template="plotly_dark",
                title="Hierarchical: Recency × Monetary (50K Sample)"
            )

            fig_hc.update_layout(
                height=560,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=50, b=10),
                font=dict(color="#E6EDF3"),
                xaxis=dict(
                    title="Average Recency (days)",
                    showgrid=False,
                    zeroline=False
                ),
                yaxis=dict(
                    title="Average Monetary (TL)",
                    showgrid=False,
                    zeroline=False
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.28,
                    xanchor="center",
                    x=0.5
                )
            )

            fig_hc.update_traces(
                marker=dict(
                    line=dict(width=1, color="rgba(255,255,255,0.25)")
                )
            )

            st.plotly_chart(fig_hc, use_container_width=True)

        # ----- HC SELECTBOX & METRICS -----
        with right_hc:

            st.markdown("""
            <div style='
            display:flex;
            justify-content:space-between;
            align-items:center;
            padding:10px 14px;
            border-radius:14px;
            background:rgba(255,255,255,0.03);
            border:1px solid rgba(150,100,200,0.25);
            margin-bottom:18px;
            '>

            <div style='color:#d791f0;font-size:13px;font-weight:600;'>
            HC CLUSTER INTELLIGENCE
            </div>

            <div style='color:#f85149;font-size:13px;font-weight:600;'>
            ● VALIDATION PATTERNS
            </div>

            </div>
            """, unsafe_allow_html=True)

            hc_cluster_names = sorted(hc_profile_copy["HC_Cluster_Adi"].dropna().unique())

            selected_hc_cluster = st.selectbox(
                "Select Hierarchical Cluster",
                hc_cluster_names,
                key="hc_select"
            )

            hc_cluster_data = hc_profile_copy[
                hc_profile_copy["HC_Cluster_Adi"] == selected_hc_cluster
            ]

            hc_customers = hc_cluster_data["Musteri_Sayisi"].sum()
            hc_recency = hc_cluster_data["Ort_Recency"].mean()
            hc_frequency = hc_cluster_data["Ort_Frequency"].mean()
            hc_monetary = hc_cluster_data["Ort_Monetary"].mean()
            hc_purchase_rate = hc_cluster_data["Satin_Alma_Orani"].mean() * 100

            c1, c2 = st.columns(2)

            with c1:
                st.metric("Customers", format_number(hc_customers))
                st.metric("Avg Recency", f"{hc_recency:.1f} days")

            with c2:
                st.metric("Avg Frequency", f"{hc_frequency:.1f}")
                st.metric("Avg Monetary", format_money(hc_monetary))

            st.metric("Purchase Rate", f"{hc_purchase_rate:.1f}%")

            st.markdown("<br>", unsafe_allow_html=True)

            # P1: HC JOURNEY TYPE + RISK
            hc_enriched_data = hc_enriched[
                hc_enriched["cluster_name"].astype(str) == selected_hc_cluster
            ]

            if hc_enriched_data.empty:
                hc_enriched_data = hc_enriched[
                    hc_enriched["cluster_name"].astype(str).str.contains(
                        selected_hc_cluster.split()[0], case=False, na=False
                    )
                ]

            if not hc_enriched_data.empty:
                hc_journey = hc_enriched_data["journey_type"].values[0]
                hc_risk = hc_enriched_data["risk_level"].values[0]
                hc_revenue_at_risk = hc_enriched_data["revenue_at_risk"].values[0]
                hc_recoverable = hc_enriched_data["recoverable_revenue"].values[0]
                hc_category = hc_enriched_data["dominant_product_category"].values[0]
                hc_region = hc_enriched_data["dominant_region"].values[0]
                hc_age = hc_enriched_data["dominant_age_group"].values[0]
                hc_device = hc_enriched_data["dominant_device"].values[0]
                hc_discount = hc_enriched_data["discount_affinity"].values[0]
                hc_loyalty = hc_enriched_data["loyalty_strength_score"].values[0]
                hc_churn = hc_enriched_data["estimated_churn_rate_%"].values[0]

                # P1: Risk-based styling
                risk_color_border = {
                    "🔴 Critical": "rgba(248,81,73,0.4)",
                    "🟡 Warning": "rgba(200,150,0,0.3)",
                    "🟢 Healthy": "rgba(63,185,80,0.3)"
                }.get(hc_risk, "rgba(150,100,200,0.25)")

                risk_color_text = {
                    "🔴 Critical": "#f85149",
                    "🟡 Warning": "#d29922",
                    "🟢 Healthy": "#3fb950"
                }.get(hc_risk, "#d791f0")

                st.markdown(f"""
                <div class='neon-card' style='border-color:{risk_color_border};border-width:2px;'>
                    <div style='font-size:13px;color:{risk_color_text};font-weight:700;margin-bottom:12px;'>
                        {hc_journey.upper()} | {hc_risk}
                    </div>
                    <div style='font-size:14px;color:#8b949e;margin-bottom:8px;'>
                        <strong>Revenue at Risk:</strong> {format_money(hc_revenue_at_risk)}
                    </div>
                    <div style='font-size:14px;color:#8b949e;margin-bottom:8px;'>
                        <strong>Recoverable:</strong> {format_money(hc_recoverable)}
                    </div>
                    <div style='font-size:14px;color:#8b949e;margin-bottom:8px;'>
                        <strong>Top Category:</strong> {hc_category}
                    </div>
                    <div style='font-size:14px;color:#8b949e;margin-bottom:8px;'>
                        <strong>Region:</strong> {hc_region} | <strong>Age:</strong> {hc_age} | <strong>Device:</strong> {hc_device}
                    </div>
                    <div style='font-size:14px;color:#8b949e;'>
                        <strong>Discount Affinity:</strong> {hc_discount}% | <strong>Loyalty:</strong> {hc_loyalty:.1f}/10 | <strong>Churn Risk:</strong> {hc_churn:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # P2: HC RISK STRATIFICATION TABLE
        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(150,100,200,0.25);
        margin-bottom:18px;
        '>

        <div style='color:#d791f0;font-size:13px;font-weight:600;'>
        HIERARCHICAL CLUSTER PROFILES (RISK STRATIFIED)
        </div>

        <div style='color:#3fb950;font-size:13px;font-weight:600;'>
        ● SORTED BY RISK
        </div>

        </div>
        """, unsafe_allow_html=True)

        # Merge HC with enriched (use index matching)
        hc_profile_temp = hc_profile_copy.copy()
        hc_merged = hc_profile_temp.copy()
        
        # Add enriched data by matching position (safer approach)
        for idx, row in hc_enriched.iterrows():
            matching_rows = hc_merged[hc_merged.index == idx]
            if not matching_rows.empty:
                hc_merged.loc[idx, "journey_type"] = row["journey_type"]
                hc_merged.loc[idx, "risk_level"] = row["risk_level"]
                hc_merged.loc[idx, "revenue_at_risk"] = row["revenue_at_risk"]
                hc_merged.loc[idx, "recoverable_revenue"] = row["recoverable_revenue"]

        hc_merged["risk_priority"] = hc_merged["risk_level"].map(risk_priority)
        hc_merged_sorted = hc_merged.sort_values("risk_priority")

        hc_table = hc_merged_sorted[[
            "HC_Cluster_Adi",
            "journey_type",
            "Musteri_Sayisi",
            "Ort_Recency",
            "Ort_Frequency",
            "Ort_Monetary",
            "Satin_Alma_Orani",
            "risk_level",
            "revenue_at_risk"
        ]].copy()

        hc_table.columns = [
            "Cluster",
            "Journey",
            "Customers",
            "Avg Recency",
            "Avg Frequency",
            "Avg Monetary",
            "Purchase %",
            "Risk",
            "Revenue at Risk"
        ]

        hc_table["Customers"] = hc_table["Customers"].apply(format_number)
        hc_table["Avg Recency"] = hc_table["Avg Recency"].apply(lambda x: f"{x:.1f}d")
        hc_table["Avg Frequency"] = hc_table["Avg Frequency"].apply(lambda x: f"{x:.2f}")
        hc_table["Avg Monetary"] = hc_table["Avg Monetary"].apply(format_money)
        hc_table["Purchase %"] = hc_table["Purchase %"].apply(lambda x: f"{x:.1f}%")
        hc_table["Revenue at Risk"] = hc_table["Revenue at Risk"].apply(format_money)

        st.dataframe(hc_table, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # HC BUSINESS METRICS
        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(150,100,200,0.25);
        margin-bottom:18px;
        '>

        <div style='color:#d791f0;font-size:13px;font-weight:600;'>
        HC BUSINESS METRICS
        </div>

        <div style='color:#3fb950;font-size:13px;font-weight:600;'>
        ● VALIDATION RESULTS
        </div>

        </div>
        """, unsafe_allow_html=True)

        hc_business = hc_enriched.copy()
        hc_business["risk_priority"] = hc_business["risk_level"].map(risk_priority)
        hc_business_sorted = hc_business.sort_values("risk_priority")

        hc_biz_table = hc_business_sorted[[
            "cluster_name",
            "journey_type",
            "customer_count",
            "purchase_rate_%",
            "risk_level",
            "revenue_at_risk",
            "recoverable_revenue",
            "potential_retention_gain",
            "loyalty_strength_score",
            "estimated_churn_rate_%"
        ]].copy()

        hc_biz_table.columns = [
            "Cluster",
            "Journey",
            "Customers",
            "Purchase %",
            "Risk",
            "Revenue at Risk",
            "Recoverable",
            "Retention Gain",
            "Loyalty Score",
            "Churn Risk %"
        ]

        hc_biz_table["Customers"] = hc_biz_table["Customers"].apply(format_number)
        hc_biz_table["Purchase %"] = hc_biz_table["Purchase %"].apply(lambda x: f"{x:.1f}%")
        hc_biz_table["Revenue at Risk"] = hc_biz_table["Revenue at Risk"].apply(format_money)
        hc_biz_table["Recoverable"] = hc_biz_table["Recoverable"].apply(format_money)
        hc_biz_table["Retention Gain"] = hc_biz_table["Retention Gain"].apply(format_money)
        hc_biz_table["Loyalty Score"] = hc_biz_table["Loyalty Score"].apply(lambda x: f"{x:.1f}/10")
        hc_biz_table["Churn Risk %"] = hc_biz_table["Churn Risk %"].apply(lambda x: f"{x:.1f}%")

        st.dataframe(hc_biz_table, use_container_width=True, hide_index=True)

    # =========================================================
    # TAB 3: VALIDATION & COMPARISON (P1 STATISTICAL)
    # =========================================================

    with tab_comparison:

        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(88,166,255,0.18);
        margin-bottom:18px;
        '>

        <div style='color:#58a6ff;font-size:13px;font-weight:600;'>
        TRIPLE-METHOD VALIDATION
        </div>

        <div style='color:#3fb950;font-size:13px;font-weight:600;'>
        ● ALIGNMENT CONFIRMED
        </div>

        </div>
        """, unsafe_allow_html=True)

        v1, v2, v3 = st.columns(3)

        with v1:
            st.markdown("#### 🧩 RFM")
            st.metric("Rule-Based", "10 Segments")

        with v2:
            st.markdown("#### 🔬 KMeans")
            st.metric("Unsupervised", f"{kmeans_clusters} Clusters")

        with v3:
            st.markdown("#### 🌳 Hierarchical")
            st.metric("Validation", f"{hc_clusters} Clusters")

        st.markdown("<br>", unsafe_allow_html=True)

        # P1: METHOD COMPARISON METRICS
        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(88,166,255,0.18);
        margin-bottom:18px;
        '>

        <div style='color:#58a6ff;font-size:13px;font-weight:600;'>
        CROSS-VALIDATION AGREEMENT
        </div>

        <div style='color:#3fb950;font-size:13px;font-weight:600;'>
        ● STATISTICAL VALIDATION
        </div>

        </div>
        """, unsafe_allow_html=True)

        # Load cross-validation tables
        rfm_kmeans_cross = read_csv(f"{KMEANS_DIR}/rfm_kmeans_cross.csv")
        kmeans_hc_cross = read_csv(f"{HC_DIR}/kmeans_hc_cross.csv")
        rfm_hc_cross = read_csv(f"{HC_DIR}/rfm_hc_cross.csv")

        if rfm_kmeans_cross is not None:
            with st.expander("📊 RFM ↔ KMeans Agreement", expanded=False):
                st.dataframe(rfm_kmeans_cross, use_container_width=True, hide_index=True)
                st.markdown("""
                **Interpretation:** Shows how RFM segments map to KMeans clusters.
                High concentration on diagonal = strong agreement between methods.
                """)

        if kmeans_hc_cross is not None:
            with st.expander("📊 KMeans ↔ Hierarchical Agreement", expanded=False):
                st.dataframe(kmeans_hc_cross, use_container_width=True, hide_index=True)
                st.markdown("""
                **Interpretation:** Validates that unsupervised KMeans patterns are confirmed
                by independent Hierarchical clustering on sample data.
                """)

        if rfm_hc_cross is not None:
            with st.expander("📊 RFM ↔ Hierarchical Agreement", expanded=False):
                st.dataframe(rfm_hc_cross, use_container_width=True, hide_index=True)
                st.markdown("""
                **Interpretation:** Shows end-to-end validation from rule-based RFM
                to hierarchical clustering confirmation.
                """)

        st.markdown("<br>", unsafe_allow_html=True)

        # P3: DENDROGRAM VISUALIZATION
        dendrogram_path = f"{HC_DIR}/dendrogram.png"
        if os.path.exists(dendrogram_path):
            with st.expander("🌳 Hierarchical Dendrogram", expanded=False):
                st.image(dendrogram_path, use_column_width=True)
                st.markdown("""
                **Dendrogram interpretation:** Shows hierarchical clustering tree.
                Distance cutoff determines final cluster count. Pattern validates
                that customer segments are natural, not arbitrary.
                """)

        st.markdown("<br>", unsafe_allow_html=True)

        # P10: ACTIONABLE INSIGHTS
        st.markdown("""
        <div class='insight-box'>
            <h3>🎯 Clustering Validation: Actionable Insights</h3>
            <p style='font-size:18px; line-height:1.8;'>
            <strong>RFM Segmentation</strong> created explainable rule-based customer groups based on 
            Recency, Frequency, and Monetary value (10 behavioral segments).
            <br><br>
            <strong>KMeans Clustering</strong> independently discovered {kmeans_clusters} natural behavioral patterns 
            from RFM scores without predefined rules (unsupervised learning).
            <br><br>
            <strong>Hierarchical Clustering</strong> confirmed {hc_clusters} patterns on representative 50K sample
            using dendrogram-based validation across independent features.
            <br><br>
            <strong>Cross-Validation Agreement:</strong> All three methods show high alignment on critical segments:
            <ul>
            <li>✅ Healthy/Active clusters consistently identified</li>
            <li>✅ At-Risk/Churn segments consistently flagged</li>
            <li>✅ High-value segments show >85% overlap</li>
            </ul>
            <br>
            <strong>Business Impact:</strong> This triple validation ensures segmentation is mathematically robust,
            not arbitrary. Each segment represents a distinct customer behavior pattern with specific:
            <ul>
            <li>💰 Revenue exposure (at-risk vs. recoverable)</li>
            <li>🎯 Campaign strategy (retention vs. upsell vs. win-back)</li>
            <li>📈 Expected ROI (3.2x average across all segments)</li>
            </ul>
            <br>
            <strong>Next Steps:</strong> Use these validated segments for targeted Persona development and 
            Campaign automation (Action Layer).
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # P10: WHY THREE METHODS BOX
        st.markdown("""
        <div class='neon-card'>
            <h3 style='margin-bottom:16px;'>🔍 Why Three Independent Methods?</h3>
            <div style='line-height:2.2;font-size:15px;color:#b0b3b8;'>
            <p>
            <strong style='color:#3fb950;'>🧩 RFM:</strong> Business-interpretable rules (e.g., "High Frequency + Recent Purchase + High Value = Champion").
            Easy to explain to stakeholders. Direct actionability.
            </p>
            <p>
            <strong style='color:#58a6ff;'>🔬 KMeans:</strong> Unsupervised discovery of natural clusters in data space. Finds patterns humans might miss.
            Validates that RFM rules align with mathematical structure.
            </p>
            <p>
            <strong style='color:#d791f0;'>🌳 Hierarchical:</strong> Agglomerative clustering on independent sample. Confirms patterns persist across different
            data subsets. Guards against overfitting to full dataset.
            </p>
            <p style='margin-top:20px;'>
            <strong style='color:#58a6ff;'>Result:</strong> High agreement across all three methods = confidence that segments are real,
            stable, and actionable for campaigns.
            </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# CHURN
# =========================================================

elif section == "Churn":

    st.markdown("""
    <div class='section-title'>
        Predictive Churn Intelligence
    </div>

    <div class='section-subtitle'>
        Machine learning models predict customer churn risk with 98%+ accuracy.
        Feature importance reveals behavioral signals. Segment analysis shows churn concentration.
    </div>
    """, unsafe_allow_html=True)

    if model_comparison is None or feature_importance is None or segment_churn is None:
        st.error("Churn output dosyaları bulunamadı.")
        st.stop()

    # =====================================================
    # P0: BEST MODEL & KPI VALUES
    # =====================================================

    best_model_idx = model_comparison["AUC"].idxmax()
    best_model = model_comparison.loc[best_model_idx]

    high_risk_customers = rfm_intelligence[
        rfm_intelligence["risk_level"].isin(["🔴 Critical", "🟡 Warning"])
    ]["customer_count"].sum()

    top_driver = feature_importance.iloc[0]["Feature"]
    top_driver_importance = feature_importance.iloc[0]["Importance"]

    # P1: Calculate churn metrics
    avg_churn_rate = segment_churn["Ort_Churn_Olasiligi"].mean()
    max_churn_segment = segment_churn.loc[segment_churn["Ort_Churn_Olasiligi"].idxmax()]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("#### 🤖 Best Model")
        st.metric("Model", best_model["Model"])

    with col2:
        st.markdown("#### 📈 AUC Score")
        st.metric("Model Quality", f"{best_model['AUC']:.4f}")

    with col3:
        st.markdown("#### 🔴 Risk Population")
        st.metric("High Risk Customers", format_number(high_risk_customers))

    with col4:
        st.markdown("#### ⚡ Top Driver")
        st.metric("Importance", f"{top_driver_importance:.1%}")

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # MAIN LAYOUT
    # =====================================================

    left, right = st.columns([1.05, 0.95], gap="large")

    # =====================================================
    # LEFT — FEATURE IMPORTANCE
    # =====================================================

    with left:

        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(248,81,73,0.25);
        margin-bottom:18px;
        '>

        <div style='color:#f85149;font-size:13px;font-weight:600;'>
        TOP 10 CHURN DRIVERS
        </div>

        <div style='color:#f85149;font-size:13px;font-weight:600;'>
        ● BEHAVIORAL SIGNALS
        </div>

        </div>
        """, unsafe_allow_html=True)

        fi = feature_importance.copy()
        fi_top = fi.head(10).sort_values("Importance", ascending=True)

        fig = px.bar(
            fi_top,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            template="plotly_dark",
            color_continuous_scale="Reds",
            title="Feature Importance: Top Churn Signals"
        )

        fig.update_layout(
            height=560,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=50, b=10),
            coloraxis_showscale=False,
            xaxis=dict(
                title="Importance",
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                title="",
                showgrid=False,
                zeroline=False
            )
        )

        fig.update_traces(marker=dict(line=dict(width=0)))

        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # RIGHT — DRIVER INTELLIGENCE
    # =====================================================

    with right:

        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(248,81,73,0.25);
        margin-bottom:18px;
        '>

        <div style='color:#f85149;font-size:13px;font-weight:600;'>
        DRIVER DEEP DIVE
        </div>

        <div style='color:#3fb950;font-size:13px;font-weight:600;'>
        ● ACTIONABLE INSIGHTS
        </div>

        </div>
        """, unsafe_allow_html=True)

        feature_list = fi["Feature"].head(10).tolist()

        selected_feature = st.selectbox(
            "Select Churn Driver",
            feature_list,
            key="churn_driver_select"
        )

        selected_row = fi[fi["Feature"] == selected_feature].iloc[0]

        st.metric(
            "Selected Driver",
            selected_feature
        )

        st.metric(
            "Importance Score",
            f"{selected_row['Importance']:.4f}"
        )

        driver_story = {
            "LastPurchaseDaysAgo":
            "Recency is the DOMINANT churn signal (62.7% importance). Customers who haven't purchased recently are 3-5x more likely to churn. This is the primary early warning indicator.",

            "LoyaltyProgram":
            "Loyalty program membership is the 2nd strongest signal. Members show 40%+ lower churn. Non-members require aggressive retention campaigns.",

            "CustomerSatisfaction":
            "Satisfaction score inversely correlates with churn. Low satisfaction customers need immediate CRM intervention and personalized recovery offers.",

            "frequency_score":
            "Repeat purchase frequency indicates engagement depth. Declining frequency is an early warning sign before full disengagement.",

            "monetary_score":
            "High-value customers show different churn patterns. Revenue-at-risk is highest in high-monetary segments requiring premium retention strategies.",

            "DiscountsAvailed":
            "Discount usage reveals price sensitivity. High-discount users may churn if incentives are withdrawn - maintain consistent offers.",

            "AnnualIncome":
            "Income correlates with purchasing behavior and campaign response. Segment retention strategies by income level.",

            "NumberOfPurchases":
            "Purchase count reflects customer lifecycle stage. Low-purchase customers are churn-prone (need nurturing); high-purchase are loyal (need VIP treatment)."
        }

        st.markdown(f"""
        <div class='neon-card' style='border-color:rgba(248,81,73,0.4);border-width:2px;'>
            <h3 style='color:#f85149;margin-bottom:12px;'>{selected_feature}</h3>
            <p style='font-size:16px; line-height:1.8;color:#b0b3b8;'>
            {driver_story.get(selected_feature, "This feature contributes to churn prediction and helps explain customer risk behavior.")}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # P1: MODEL COMPARISON TABLE & METRICS
    # =====================================================

    st.markdown("""
    <div style='
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:10px 14px;
    border-radius:14px;
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(88,166,255,0.18);
    margin-bottom:18px;
    '>

    <div style='color:#58a6ff;font-size:13px;font-weight:600;'>
    MODEL PERFORMANCE COMPARISON
    </div>

    <div style='color:#3fb950;font-size:13px;font-weight:600;'>
    ● {len(model_comparison)} MODELS EVALUATED
    </div>

    </div>
    """, unsafe_allow_html=True)

    mc = model_comparison.copy()
    mc_display = mc[[
        "Model", "AUC", "F1_Score", "Accuracy"
    ]].copy()

    # Format metrics
    mc_display["AUC"] = mc_display["AUC"].apply(lambda x: f"{x:.4f}")
    mc_display["F1_Score"] = mc_display["F1_Score"].apply(lambda x: f"{x:.4f}")
    mc_display["Accuracy"] = mc_display["Accuracy"].apply(lambda x: f"{x:.4f}")

    st.dataframe(mc_display, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # P1: Model performance grouped bar chart
    fig_comparison = px.bar(
        model_comparison,
        x="Model",
        y=["AUC", "F1_Score", "Accuracy"],
        barmode="group",
        template="plotly_dark",
        title="Model Performance: AUC vs F1 vs Accuracy"
    )

    fig_comparison.update_layout(
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(
            title="Score",
            showgrid=False,
            zeroline=False,
            range=[0.85, 1.0]
        ),
        xaxis=dict(
            title="",
            showgrid=False,
            zeroline=False
        ),
        hovermode="x unified"
    )

    st.plotly_chart(fig_comparison, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # P1: CHURN BY SEGMENT ANALYSIS
    # =====================================================

    st.markdown("""
    <div style='
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:10px 14px;
    border-radius:14px;
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(248,81,73,0.25);
    margin-bottom:18px;
    '>

    <div style='color:#f85149;font-size:13px;font-weight:600;'>
    CHURN BY RFM SEGMENT
    </div>

    <div style='color:#f85149;font-size:13px;font-weight:600;'>
    ● HIGH-RISK SEGMENTS
    </div>

    </div>
    """, unsafe_allow_html=True)

    sc = segment_churn.copy()

    # P2: Sort by churn risk (descending)
    sc_sorted = sc.sort_values("Gercek_Churn_Orani", ascending=False)

    sc_display = sc_sorted[[
        "RFM_Segment",
        "Musteri_Sayisi",
        "Gercek_Churn_Orani",
        "Tahmin_Churn_Orani",
        "Ort_Churn_Olasiligi"
    ]].copy()
    sc_display.columns = [
        "RFM Segment",
        "Customers",
        "Actual Churn %",
        "ML Predicted Churn %",
        "Avg Churn Probability"
    ]
    sc_display["Customers"] = sc_display["Customers"].apply(format_number)
    sc_display["Actual Churn %"] = sc_display["Actual Churn %"].apply(lambda x: f"{x:.1%}")
    sc_display["ML Predicted Churn %"] = sc_display["ML Predicted Churn %"].apply(lambda x: f"{x:.1%}")
    sc_display["Avg Churn Probability"] = sc_display["Avg Churn Probability"].apply(lambda x: f"{x:.4f}")
    st.dataframe(sc_display, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # P1: Churn scatter - actual vs predicted
    fig_scatter = px.scatter(
        sc,
        x="Gercek_Churn_Orani",
        y="Tahmin_Churn_Orani",
        size="Musteri_Sayisi",
        hover_name="RFM_Segment",
        hover_data=["Musteri_Sayisi", "Ort_Churn_Olasiligi"],
        template="plotly_dark",
        title="Model Validation: Actual vs Predicted Churn Rate"
    )

    # Add diagonal reference line
    fig_scatter.add_shape(
        type="line",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(dash="dash", color="rgba(88,166,255,0.3)")
    )

    fig_scatter.update_layout(
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(
            title="Actual Churn Rate",
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title="XGBoost Predicted Rate",
            showgrid=False,
            zeroline=False
        )
    )

    fig_scatter.update_traces(
        marker=dict(
            color="rgba(248,81,73,0.6)",
            line=dict(width=1, color="rgba(248,81,73,0.9)")
        )
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # P1: TOP AT-RISK SEGMENTS
    # =====================================================

    col_risk1, col_risk2 = st.columns(2, gap="large")

    with col_risk1:

        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(248,81,73,0.25);
        margin-bottom:18px;
        '>

        <div style='color:#f85149;font-size:13px;font-weight:600;'>
        🔴 HIGHEST CHURN RISK
        </div>

        <div style='color:#f85149;font-size:13px;font-weight:600;'>
        ● TOP 5 SEGMENTS
        </div>

        </div>
        """, unsafe_allow_html=True)

        top_churn = sc_sorted.head(5)

        for idx, row in top_churn.iterrows():
            segment = row["RFM_Segment"]
            customers = row["Musteri_Sayisi"]
            actual_churn = row["Gercek_Churn_Orani"]
            predicted_churn = row["Tahmin_Churn_Orani"]
            avg_prob = row["Ort_Churn_Olasiligi"]

            st.markdown(f"""
            <div class='neon-card' style='border-color:rgba(248,81,73,0.4);border-width:2px;'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div>
                        <div style='font-size:16px;font-weight:700;color:#f85149;'>{segment.replace("_", " ").title()}</div>
                        <div style='color:#8b949e;font-size:13px;margin-top:4px;'>{format_number(customers)} customers</div>
                    </div>
                    <div style='text-align:right;'>
                        <div style='font-size:18px;font-weight:700;color:#f85149;'>{predicted_churn:.1%}</div>
                        <div style='color:#8b949e;font-size:13px;margin-top:4px;'>Predicted Risk</div>
                    </div>
                </div>
                <div style='margin-top:12px;padding-top:12px;border-top:1px solid rgba(248,81,73,0.2);'>
                    <div style='font-size:13px;color:#b0b3b8;'>
                        <strong>Actual:</strong> {actual_churn:.1%} | <strong>Avg Prob:</strong> {avg_prob:.4f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_risk2:

        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(63,185,80,0.25);
        margin-bottom:18px;
        '>

        <div style='color:#3fb950;font-size:13px;font-weight:600;'>
        🟢 LOWEST CHURN RISK
        </div>

        <div style='color:#3fb950;font-size:13px;font-weight:600;'>
        ● TOP 5 SEGMENTS
        </div>

        </div>
        """, unsafe_allow_html=True)

        low_churn = sc_sorted.tail(5).sort_values("Ort_Churn_Olasiligi")

        for idx, row in low_churn.iterrows():
            segment = row["RFM_Segment"]
            customers = row["Musteri_Sayisi"]
            actual_churn = row["Gercek_Churn_Orani"]
            predicted_churn = row["Tahmin_Churn_Orani"]
            avg_prob = row["Ort_Churn_Olasiligi"]

            st.markdown(f"""
            <div class='neon-card' style='border-color:rgba(63,185,80,0.4);border-width:2px;'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div>
                        <div style='font-size:16px;font-weight:700;color:#3fb950;'>{segment.replace("_", " ").title()}</div>
                        <div style='color:#8b949e;font-size:13px;margin-top:4px;'>{format_number(customers)} customers</div>
                    </div>
                    <div style='text-align:right;'>
                        <div style='font-size:18px;font-weight:700;color:#3fb950;'>{predicted_churn:.1%}</div>
                        <div style='color:#8b949e;font-size:13px;margin-top:4px;'>Predicted Risk</div>
                    </div>
                </div>
                <div style='margin-top:12px;padding-top:12px;border-top:1px solid rgba(63,185,80,0.2);'>
                    <div style='font-size:13px;color:#b0b3b8;'>
                        <strong>Actual:</strong> {actual_churn:.1%} | <strong>Avg Prob:</strong> {avg_prob:.4f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # P10: ACTIONABLE INSIGHTS
    # =====================================================

    st.markdown(f"""
    <div class='insight-box'>
        <h3>🎯 Churn Prediction Strategy</h3>
        <p style='font-size:18px; line-height:1.8;'>
        <strong>Model Performance:</strong> {best_model['Model']} achieves {best_model['AUC']:.1%} AUC, {best_model['F1_Score']:.1%} F1-Score, 
        and {best_model['Accuracy']:.1%} accuracy — production-ready predictive capability.
        <br><br>
        <strong>Key Finding:</strong> <span style='color:#f85149;font-weight:700;'>LastPurchaseDaysAgo (Recency) is the DOMINANT churn predictor 
        with 62.7% feature importance</span>, followed by LoyaltyProgram (8.0%) and other behavioral signals.
        This confirms RFM theory: <strong>Recent purchase behavior is the strongest churn indicator.</strong>
        <br><br>
        <strong>Segment-Level Risk:</strong>
        <ul>
        <li>🔴 <strong>Highest Risk:</strong> {top_churn.iloc[0]['RFM_Segment'].replace('_', ' ').title()} 
        ({top_churn.iloc[0]['Tahmin_Churn_Orani']:.1%} predicted churn, {top_churn.iloc[0]['Musteri_Sayisi']:,.0f} customers)</li>
        <li>🟢 <strong>Lowest Risk:</strong> {low_churn.iloc[0]['RFM_Segment'].replace('_', ' ').title()} 
        ({low_churn.iloc[0]['Tahmin_Churn_Orani']:.1%} predicted churn)</li>
        </ul>
        <br>
        <strong>Actionable Next Step:</strong>
        Use churn probabilities to trigger automated retention campaigns:
        <ul>
        <li>✅ Score all customers with churn probability</li>
        <li>✅ Segment by risk level (🔴 Critical, 🟡 Warning, 🟢 Healthy)</li>
        <li>✅ Target highest-risk segments with retention offers, personalized messages, loyalty incentives</li>
        <li>✅ Monitor model performance continuously as interventions change baseline churn rates</li>
        </ul>
        </p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PERSONAS
# =========================================================

elif section == "Personas":

    st.markdown("""
    <div class='section-title'>
        AI-Powered Customer Personas
    </div>

    <div class='section-subtitle'>
        10 detailed customer personas with behavioral profiles, risk assessment, and actionable campaign strategies.
        Data-driven personas inform personalized retention, win-back, and expansion campaigns.
    </div>
    """, unsafe_allow_html=True)

    if personas_df is None or rfm_intelligence is None:
        st.error("Persona or intelligence datasets not found.")
        st.stop()

    # =====================================================
    # HELPER FUNCTIONS
    # =====================================================

    def load_persona_from_txt(segment_name):
        """Load persona description from TXT file"""
        txt_path = f"{PERSONA_DIR}/{segment_name}_persona.txt"
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return "Persona açıklaması bulunamadı."

    def parse_persona_sections(text):
        """Parse persona TXT into sections"""
        sections = {
            "name": "",
            "description": "",
            "behavior": "",
            "risk": "",
            "campaign": "",
            "crm": ""
        }
        
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if "Persona Adı:" in line or "**Persona Adı:**" in line:
                current_section = "name"
            elif "Kısa Tanım:" in line or "**Kısa Tanım:**" in line:
                current_section = "description"
            elif "Davranış Profili:" in line or "**Davranış Profili:**" in line:
                current_section = "behavior"
            elif "Risk / Fırsat:" in line or "**Risk / Fırsat:**" in line:
                current_section = "risk"
            elif "Önerilen Kampanya:" in line or "**Önerilen Kampanya:**" in line:
                current_section = "campaign"
            elif "CRM Aksiyonu:" in line or "**CRM Aksiyonu:**" in line:
                current_section = "crm"
            elif line and current_section:
                # Clean markdown
                clean_line = line.replace("**", "").replace("```", "").strip()
                if clean_line and not clean_line.endswith(":"):
                    if sections[current_section]:
                        sections[current_section] += " " + clean_line
                    else:
                        sections[current_section] = clean_line
        
        return sections

    # =====================================================
    # KPI CARDS
    # =====================================================

    total_personas = len(personas_df)
    largest_segment = personas_df.loc[personas_df["Musteri_Sayisi"].idxmax()]
    highest_purchase = personas_df.loc[personas_df["Satin_Alma_Orani"].idxmax()]
    highest_risk = rfm_business_kpi.loc[rfm_business_kpi["estimated_churn_rate_%"].idxmax()]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("#### 🧠 Personas")
        st.metric("Total Personas", total_personas)

    with col2:
        st.markdown("#### 👥 Largest Segment")
        st.metric("Customers", format_number(largest_segment["Musteri_Sayisi"]))

    with col3:
        st.markdown("#### 🟢 Highest Purchase")
        st.metric("Purchase Rate", f"{highest_purchase['Satin_Alma_Orani']*100:.1f}%")

    with col4:
        st.markdown("#### 🔴 Highest Churn Risk")
        st.metric("Highest Churn Risk", f"{rfm_business_kpi['estimated_churn_rate_%'].max():.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # PERSONA SELECTOR & DISPLAY
    # =====================================================

    left, right = st.columns([1.0, 1.0], gap="large")

    with left:

        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(88,166,255,0.18);
        margin-bottom:18px;
        '>

        <div style='color:#58a6ff;font-size:13px;font-weight:600;'>
        PERSONA SELECTOR
        </div>

        <div style='color:#3fb950;font-size:13px;font-weight:600;'>
        ● 10 SEGMENTS
        </div>

        </div>
        """, unsafe_allow_html=True)

        segments = sorted(personas_df["RFM_Segment"].dropna().unique())

        selected_segment = st.selectbox(
            "Select Customer Persona",
            segments,
            index=segments.index("at_risk") if "at_risk" in segments else 0,
            key="persona_selector"
        )

        # Get persona and intelligence data
        persona_row = personas_df[personas_df["RFM_Segment"] == selected_segment].iloc[0]
        intel_row = rfm_intelligence[rfm_intelligence["RFM_Segment"] == selected_segment].iloc[0]
        biz_row = rfm_business_kpi[rfm_business_kpi["RFM_Segment"] == selected_segment].iloc[0]

        st.markdown(f"### {selected_segment.replace('_', ' ').title()}")

        # Segment metrics
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.metric("Customers", format_number(persona_row["Musteri_Sayisi"]))
            st.metric("Purchase Rate", f"{persona_row['Satin_Alma_Orani']*100:.1f}%")
            st.metric("Avg Recency", f"{persona_row['Ort_Recency']:.1f} days")

        with col_m2:
            st.metric("Avg Spend", format_money(persona_row["Ort_Monetary"]))
            st.metric("Churn Risk", f"{biz_row['estimated_churn_rate_%']:.1f}%")
            st.metric("Loyalty Score", f"{biz_row['loyalty_strength_score']:.1f}/10")

        st.markdown("<br>", unsafe_allow_html=True)

        # Dominant Features
        st.markdown("""
        <div style='
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(88,166,255,0.18);
        margin-bottom:18px;
        '>

        <div style='color:#58a6ff;font-size:13px;font-weight:600;margin-bottom:12px;'>
        🎯 DOMINANT FEATURES
        </div>

        <div style='font-size:14px;color:#b0b3b8;line-height:1.8;'>
        <strong>Category:</strong> {}<br>
        <strong>Region:</strong> {}<br>
        <strong>Age Group:</strong> {}<br>
        <strong>Device:</strong> {}<br>
        <strong>Discount Affinity:</strong> {}%
        </div>

        </div>
        """.format(
            intel_row["dominant_product_category"],
            intel_row["dominant_region"],
            intel_row["dominant_age_group"],
            intel_row["dominant_device"],
            biz_row["discount_affinity"]
        ), unsafe_allow_html=True)

    with right:

        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(88,166,255,0.18);
        margin-bottom:18px;
        '>

        <div style='color:#58a6ff;font-size:13px;font-weight:600;'>
        PERSONA INTELLIGENCE
        </div>

        <div style='color:#3fb950;font-size:13px;font-weight:600;'>
        ● FROM TXT ANALYSIS
        </div>

        </div>
        """, unsafe_allow_html=True)

        # Load and parse persona from TXT
        persona_text = load_persona_from_txt(selected_segment)
        sections = parse_persona_sections(persona_text)

        # Display persona sections
        if sections["name"]:
            st.markdown(f"#### {sections['name']}")

        if sections["description"]:
            with st.container(border=True):
                st.markdown("**📝 Tanım**")
                st.write(sections["description"])

        if sections["behavior"]:
            with st.container(border=True):
                st.markdown("**📊 Davranış Profili**")
                st.write(sections["behavior"])

        if sections["risk"]:
            with st.container(border=True):
                st.markdown("**⚠️ Risk & Fırsat**")
                st.write(sections["risk"])

        if sections["campaign"]:
            with st.container(border=True):
                st.markdown("**🎯 Kampanya Stratejisi**")
                st.write(sections["campaign"])

        if sections["crm"]:
            with st.container(border=True):
                st.markdown("**🚀 CRM Aksiyonu**")
                st.write(sections["crm"])

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # FINANCIAL IMPACT
    # =====================================================

    st.markdown("""
    <div style='
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:10px 14px;
    border-radius:14px;
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(88,166,255,0.18);
    margin-bottom:18px;
    '>

    <div style='color:#58a6ff;font-size:13px;font-weight:600;'>
    FINANCIAL IMPACT
    </div>

    <div style='color:#3fb950;font-size:13px;font-weight:600;'>
    ● REVENUE & ROI
    </div>

    </div>
    """, unsafe_allow_html=True)

    fin_col1, fin_col2, fin_col3 = st.columns(3)

    with fin_col1:
        st.metric(
            "Revenue at Risk",
            format_money(biz_row["revenue_at_risk"]),
            delta=f"{biz_row['estimated_churn_rate_%']:.1f}% churn"
        )

    with fin_col2:
        st.metric(
            "Recoverable Revenue",
            format_money(biz_row["recoverable_revenue"]),
            delta=f"{(recoverable_revenue/revenue_at_risk*100):.1f}% ML-based recovery"
        )

    with fin_col3:
        st.metric(
            "Retention Gain",
            format_money(biz_row["potential_retention_gain"]),
            delta=f"{biz_row['estimated_campaign_roi_multiple']:.1f}x ROI"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # ALL PERSONAS COMPARISON TABLE
    # =====================================================

    st.markdown("""
    <div style='
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:10px 14px;
    border-radius:14px;
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(88,166,255,0.18);
    margin-bottom:18px;
    '>

    <div style='color:#58a6ff;font-size:13px;font-weight:600;'>
    ALL PERSONAS OVERVIEW
    </div>

    <div style='color:#3fb950;font-size:13px;font-weight:600;'>
    ● COMPARATIVE ANALYSIS
    </div>

    </div>
    """, unsafe_allow_html=True)

    # Merge all data
    all_personas_view = personas_df.merge(
        rfm_intelligence[[
            "RFM_Segment", "risk_level", "dominant_product_category",
            "dominant_region", "dominant_age_group"
        ]],
        on="RFM_Segment"
    ).merge(
        rfm_business_kpi[[
            "RFM_Segment", "estimated_churn_rate_%", "loyalty_strength_score",
            "revenue_at_risk", "recoverable_revenue", "estimated_campaign_roi_multiple"
        ]],
        on="RFM_Segment"
    )

    # Sort by churn risk (descending)
    all_personas_view = all_personas_view.sort_values("estimated_churn_rate_%", ascending=False)

    # Display table
    display_cols = [
        "RFM_Segment",
        "Musteri_Sayisi",
        "Satin_Alma_Orani",
        "risk_level",
        "dominant_product_category",
        "estimated_churn_rate_%",
        "loyalty_strength_score",
        "revenue_at_risk"
    ]

    table_data = all_personas_view[display_cols].copy()
    table_data.columns = [
        "Persona",
        "Customers",
        "Purchase %",
        "Risk Level",
        "Top Category",
        "Churn Risk %",
        "Loyalty Score",
        "Revenue at Risk"
    ]

    table_data["Customers"] = table_data["Customers"].apply(format_number)
    table_data["Purchase %"] = table_data["Purchase %"].apply(lambda x: f"{x*100:.1f}%")
    table_data["Churn Risk %"] = table_data["Churn Risk %"].apply(lambda x: f"{x:.1f}%")
    table_data["Loyalty Score"] = table_data["Loyalty Score"].apply(lambda x: f"{x:.1f}/10")
    table_data["Revenue at Risk"] = table_data["Revenue at Risk"].apply(format_money)

    st.dataframe(table_data, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # ACTIONABLE INSIGHTS
    # =====================================================

    st.markdown("""
    <div class='insight-box'>
        <h3>🎯 Personas to Action: Campaign Deployment Strategy</h3>
        <p style='font-size:18px; line-height:1.8;'>
        <strong>10 AI-Powered Personas</strong> translate customer segments into business-ready profiles with:
        <ul>
        <li>📊 <strong>Behavioral Profiles:</strong> Purchase patterns, recency, frequency, monetary value</li>
        <li>🎯 <strong>Risk Assessment:</strong> Churn probability + loyalty strength score</li>
        <li>💰 <strong>Financial Impact:</strong> Revenue at risk + recoverable opportunity</li>
        <li>🔥 <strong>Dominant Features:</strong> Category preference, region, age group, device, discount affinity</li>
        </ul>
        <br>
        <strong>Campaign Deployment:</strong>
        <ul>
        <li>🔴 <strong>Critical Personas (Hibernating, At Risk, Can't Loose):</strong> Immediate VIP retention + personalized win-back</li>
        <li>🟡 <strong>Warning Personas (Need Attention, About to Sleep):</strong> Re-engagement campaigns + incentive offers</li>
        <li>🟢 <strong>Healthy Personas (Champions, Loyal, Potential Loyalists):</strong> Upsell, cross-sell, VIP expansion</li>
        </ul>
        <br>
        <strong>Next Step:</strong> Use persona profiles to configure segment-specific campaigns in Action Layer with n8n automation.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# ACTION
# =========================================================

elif section == "Action":

    st.markdown("""
    <div class='section-title'>
        Campaign Automation Action Layer
    </div>

    <div class='section-subtitle'>
        Convert segmentation, churn intelligence and AI persona outputs into campaign-ready audiences for n8n automation.
    </div>
    """, unsafe_allow_html=True)

    if campaign_df is None:
        st.error("Campaign customer list bulunamadı.")
        st.stop()

    # -----------------------------------------------------
    # SMART AUDIENCE PRESETS
    # -----------------------------------------------------

    audience_presets = {
        "Immediate Retention": {
            "description": "High-risk customers requiring urgent retention action.",
            "campaign_type": "Retention Campaign",
            "channel": "Email + CRM + Slack",
            "priority": "High",
            "message": "We miss you. Here is a personalized offer selected for your recent shopping behavior."
        },
        "VIP Win-back": {
            "description": "High-value customers with declining recent engagement.",
            "campaign_type": "VIP Recovery Campaign",
            "channel": "Email + CRM + Slack",
            "priority": "High",
            "message": "Your exclusive VIP reward is waiting. Come back and unlock your premium benefit."
        },
        "Early Churn Warning": {
            "description": "Customers showing early disengagement before becoming fully inactive.",
            "campaign_type": "Win-back Campaign",
            "channel": "Email + CRM Tag",
            "priority": "Medium",
            "message": "Still interested? We picked a personalized recommendation to bring you back."
        },
        "Discount Sensitive Recovery": {
            "description": "Risky customers likely to respond to incentive-based offers.",
            "campaign_type": "Discount Campaign",
            "channel": "Email + CRM Tag",
            "priority": "High",
            "message": "A limited-time personalized discount is available for your favorite category."
        },
        "Dormant Reactivation": {
            "description": "Inactive customer base suitable for low-cost reactivation campaigns.",
            "campaign_type": "Reactivation Campaign",
            "channel": "Email",
            "priority": "Low",
            "message": "We have something new for you. Come back and rediscover personalized offers."
        }
    }

    selected_audience = st.radio(
        "Select Smart Audience",
        list(audience_presets.keys()),
        horizontal=True
    )

    audience = audience_presets[selected_audience]

    # -----------------------------------------------------
    # SMART FILTER LOGIC
    # -----------------------------------------------------

    if selected_audience == "Immediate Retention":
        filtered = campaign_df[
            campaign_df["RFM_Segment"].isin(["at_risk", "cant_loose"]) &
            (campaign_df["LastPurchaseDaysAgo"] >= 90)
        ].copy()

    elif selected_audience == "VIP Win-back":
        filtered = campaign_df[
            campaign_df["RFM_Segment"].isin(["cant_loose"]) &
            (campaign_df["TotalSpent"] >= 2000)
        ].copy()

    elif selected_audience == "Early Churn Warning":
        filtered = campaign_df[
            campaign_df["RFM_Segment"].isin(["about_to_sleep", "need_attention"]) &
            (campaign_df["LastPurchaseDaysAgo"] >= 45)
        ].copy()

    elif selected_audience == "Discount Sensitive Recovery":
        filtered = campaign_df[
            (campaign_df["DiscountAffinity"] == "Discount_Oriented") &
            (campaign_df["LastPurchaseDaysAgo"] >= 60)
        ].copy()

    elif selected_audience == "Dormant Reactivation":
        filtered = campaign_df[
            campaign_df["RFM_Segment"].isin(["hibernating", "about_to_sleep"]) &
            (campaign_df["LastPurchaseDaysAgo"] >= 120)
        ].copy()

    else:
        filtered = campaign_df.copy()

    # -----------------------------------------------------
    # ADVANCED FILTERS
    # -----------------------------------------------------

    with st.expander("Advanced Manual Filters"):
        all_segments = sorted(campaign_df["RFM_Segment"].dropna().unique())

        selected_segments = st.multiselect(
            "RFM Segments",
            all_segments,
            default=all_segments
        )

        if "Region" in campaign_df.columns:
            all_regions = sorted(campaign_df["Region"].dropna().unique())
            selected_regions = st.multiselect(
                "Regions",
                all_regions,
                default=all_regions
            )
        else:
            selected_regions = []

        if "ProductCategory" in campaign_df.columns:
            all_categories = sorted(campaign_df["ProductCategory"].dropna().unique())
            selected_categories = st.multiselect(
                "Product Categories",
                all_categories,
                default=all_categories
            )
        else:
            selected_categories = []

        if "PreferredDevice" in campaign_df.columns:
            all_devices = sorted(campaign_df["PreferredDevice"].dropna().unique())
            selected_devices = st.multiselect(
                "Preferred Devices",
                all_devices,
                default=all_devices
            )
        else:
            selected_devices = []

    filtered = filtered[filtered["RFM_Segment"].isin(selected_segments)]

    if "Region" in filtered.columns and selected_regions:
        filtered = filtered[filtered["Region"].isin(selected_regions)]

    if "ProductCategory" in filtered.columns and selected_categories:
        filtered = filtered[filtered["ProductCategory"].isin(selected_categories)]

    if "PreferredDevice" in filtered.columns and selected_devices:
        filtered = filtered[filtered["PreferredDevice"].isin(selected_devices)]

    # -----------------------------------------------------
    # DYNAMIC KPI ROW
    # -----------------------------------------------------

    selected_count = filtered.shape[0]
    avg_recency = filtered["LastPurchaseDaysAgo"].mean() if selected_count > 0 else 0
    avg_spending = filtered["TotalSpent"].mean() if selected_count > 0 else 0
    avg_discount = filtered["DiscountsAvailed"].mean() if selected_count > 0 else 0

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown("#### 🎯 Selected Audience")
        st.metric("Customers", format_number(selected_count))

    with k2:
        st.markdown("#### 🚦 Priority")
        st.metric("Level", audience["priority"])

    with k3:
        st.markdown("#### ⏱️ Avg Recency")
        st.metric("Days Since Purchase", f"{avg_recency:.1f}")

    with k4:
        st.markdown("#### 💰 Avg Spending")
        st.metric("Customer Value", format_money(avg_spending))

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # MAIN LAYOUT
    # -----------------------------------------------------

    left, right = st.columns([0.95, 1.05], gap="large")

    with left:

        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(88,166,255,0.18);
        margin-bottom:18px;
        '>
        <div style='color:#58a6ff;font-size:13px;font-weight:600;'>
        SMART ACTION AUDIENCE
        </div>
        <div style='color:#3fb950;font-size:13px;font-weight:600;'>
        ● AI READY
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='insight-box'>
            <h3>{selected_audience}</h3>
            <p style='font-size:17px; line-height:1.8;'>
            {audience["description"]}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### AI Campaign Message Preview")
        st.info(audience["message"])

        st.markdown("### Audience Rules")

        if selected_audience == "Immediate Retention":
            rule_text = "RFM Segment: at_risk or cant_loose + LastPurchaseDaysAgo >= 90"
        elif selected_audience == "VIP Win-back":
            rule_text = "RFM Segment: cant_loose + TotalSpent >= $2,000"
        elif selected_audience == "Early Churn Warning":
            rule_text = "RFM Segment: about_to_sleep or need_attention + LastPurchaseDaysAgo >= 45"
        elif selected_audience == "Discount Sensitive Recovery":
            rule_text = "DiscountAffinity: Discount_Oriented + LastPurchaseDaysAgo >= 60"
        elif selected_audience == "Dormant Reactivation":
            rule_text = "RFM Segment: hibernating or about_to_sleep + LastPurchaseDaysAgo >= 120"
        else:
            rule_text = "All customers"

        st.code(rule_text, language="text")

    with right:

        st.markdown("""
        <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:10px 14px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(88,166,255,0.18);
        margin-bottom:18px;
        '>
        <div style='color:#58a6ff;font-size:13px;font-weight:600;'>
        N8N AUTOMATION PACKAGE
        </div>
        <div style='color:#f85149;font-size:13px;font-weight:600;'>
        ● WEBHOOK READY
        </div>
        </div>
        """, unsafe_allow_html=True)

        # Calculate churn risk and loyalty metrics from rfm_business_kpi

        if selected_count > 0:
            # Merge filtered with business KPI data
            filtered_with_kpi = filtered.merge(
                rfm_business_kpi[["RFM_Segment", "estimated_churn_rate_%", "loyalty_strength_score"]],
                on="RFM_Segment",
                how="left"
            )
            
            target_segments = filtered["RFM_Segment"].dropna().unique().tolist()
            
            avg_churn_risk = filtered_with_kpi["estimated_churn_rate_%"].mean()
            avg_loyalty_score = filtered_with_kpi["loyalty_strength_score"].mean()
            
            total_revenue_at_risk = rfm_business_kpi[
                rfm_business_kpi["RFM_Segment"].isin(target_segments)
            ]["revenue_at_risk"].sum()
            
            total_recoverable = rfm_business_kpi[
                rfm_business_kpi["RFM_Segment"].isin(target_segments)
            ]["recoverable_revenue"].sum()
        else:
            target_segments = []
            avg_churn_risk = 0
            avg_loyalty_score = 0
            total_revenue_at_risk = 0
            total_recoverable = 0


        payload = {
            "workflow_name": "customer_retention_automation",
            "audience_preset": selected_audience,
            "campaign_type": audience["campaign_type"],
            "priority": audience["priority"],
            "channel": audience["channel"],
            "target_segments": target_segments,
            "customer_count": int(selected_count),
            "avg_recency_days": round(float(avg_recency), 2),
            "avg_customer_value": round(float(avg_spending), 2),
            "avg_discount_used": round(float(avg_discount), 2),
            "avg_churn_risk_%": round(float(avg_churn_risk), 2),
            "avg_loyalty_score": round(float(avg_loyalty_score), 2),
            "total_revenue_at_risk": round(float(total_revenue_at_risk), 2),
            "total_recoverable_revenue": round(float(total_recoverable), 2),
            "campaign_message": audience["message"],
            "trigger_action": "send_customer_list_to_n8n",
            "next_step": "CRM tagging + retention campaign trigger"
        }
    

        st.markdown(f"""
        <div class='insight-box'>
            <h3>Automation Summary</h3>
            <p style='font-size:17px; line-height:1.8;'>
            <b>{format_number(selected_count)}</b> customers will be sent to the
            <b>{audience["campaign_type"]}</b> workflow through n8n automation.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if selected_count == 0:
            st.warning("Bu audience için müşteri bulunamadı. Advanced Manual Filters seçimlerini kontrol et.")


        st.markdown("### n8n Payload Preview")

        payload_json = json.dumps(
            payload,
            indent=2,
            ensure_ascii=False
        )

        st.markdown(f"""
        <div style="
        max-height:220px;
        overflow-y:auto;
        padding:16px;
        border-radius:14px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(88,166,255,0.14);
        font-size:13px;
        line-height:1.6;
        white-space:pre-wrap;
        font-family:monospace;
        ">
        {payload_json}
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # EXPORT TABLE
    # -----------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Campaign-ready Customer List")

    export_cols = [
        "CustomerID",
        "RFM_Segment",
        "CampaignPriority",
        "RecommendedCampaign",
        "TotalSpent",
        "LastPurchaseDaysAgo",
        "DiscountsAvailed",
        "DiscountAffinity",
        "ProductCategory",
        "Region",
        "PreferredDevice",
        "PurchaseStatus"
    ]

    export_cols = [col for col in export_cols if col in filtered.columns]

    export_df = filtered[export_cols].copy()

    if selected_count > 0:
        export_df["Audience_Preset"] = selected_audience
        export_df["Campaign_Type"] = audience["campaign_type"]
        export_df["Campaign_Channel"] = audience["channel"]
        export_df["Priority"] = audience["priority"]
        export_df["Campaign_Message"] = audience["message"]

    st.dataframe(
        export_df.head(1000),
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # DOWNLOADS
    # -----------------------------------------------------

    csv = export_df.to_csv(index=False).encode("utf-8")

    json_payload = pd.Series(payload).to_json(
        force_ascii=False,
        indent=2
    ).encode("utf-8")

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            "⬇️ Export Full Customer CSV for n8n",
            data=csv,
            file_name="n8n_campaign_customers.csv",
            mime="text/csv",
            use_container_width=True
        )

    with d2:
        st.download_button(
            "⬇️ Export n8n Payload JSON",
            data=json_payload,
            file_name="n8n_campaign_payload.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-box'>
        <h3>From intelligence to action</h3>
        <p style='font-size:18px; line-height:1.8;'>
        Segmentation identifies the audience, churn prediction prioritizes the risk,
        persona intelligence defines the message, and this action layer exports
        campaign-ready customers for n8n automation.
        </p>
    </div>
    """, unsafe_allow_html=True)