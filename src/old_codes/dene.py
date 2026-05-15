import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI-Powered Customer Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = "data"
INTELLIGENCE_DIR = f"{BASE_DIR}/Intelligence_Output"
BUSINESS_KPI_DIR = f"{BASE_DIR}/Business_KPI_Output"
PREP_DIR = f"{BASE_DIR}/Presentation_Prep"
RFM_DIR = f"{BASE_DIR}/RFM_Output"
KMEANS_DIR = f"{BASE_DIR}/KMeans_Output"
HC_DIR = f"{BASE_DIR}/HC_Output"
CHURN_DIR = f"{BASE_DIR}/Churn_Output"
PERSONA_DIR = f"{BASE_DIR}/Persona_Output"

# =========================================================
# CUSTOM CSS - CINEMATIC DARK THEME
# =========================================================

st.markdown("""
<style>

* {
    margin: 0;
    padding: 0;
}

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0e27 0%, #16213e 100%);
    color: #e0e0e0;
}

.main {
    background: transparent;
}

.block-container {
    padding: 2rem 3rem;
    max-width: 1600px;
}

h1, h2, h3, h4 {
    color: #ffffff;
    font-weight: 700;
    letter-spacing: -0.5px;
}

h1 {
    font-size: 48px;
    margin-bottom: 12px;
    background: linear-gradient(90deg, #00d4ff, #0099cc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

h2 {
    font-size: 32px;
    margin-top: 30px;
    margin-bottom: 12px;
    border-bottom: 2px solid rgba(0, 212, 255, 0.3);
    padding-bottom: 10px;
}

h3 {
    font-size: 18px;
    color: #00d4ff;
}

/* ===== HERO BADGE ===== */
.hero-badge {
    display: inline-block;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.4);
    color: #00d4ff;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 20px;
}

/* ===== KPI CARDS ===== */
.kpi-card {
    background: linear-gradient(135deg, rgba(10, 14, 39, 0.8) 0%, rgba(22, 33, 62, 0.8) 100%);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 12px;
    padding: 24px;
    margin: 12px 0;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(0, 212, 255, 0.08);
}

.kpi-card:hover {
    border-color: rgba(0, 212, 255, 0.5);
    box-shadow: 0 8px 30px rgba(0, 212, 255, 0.15);
    transform: translateY(-2px);
}

.kpi-value {
    font-size: 36px;
    font-weight: 700;
    color: #00d4ff;
    margin: 10px 0 5px 0;
}

.kpi-label {
    font-size: 13px;
    color: #8b9dc3;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.kpi-delta {
    font-size: 12px;
    color: #00d4ff;
    margin-top: 8px;
}

/* ===== INSIGHT BOX ===== */
.insight-box {
    background: rgba(0, 212, 255, 0.05);
    border-left: 4px solid #00d4ff;
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
    color: #d0d0d0;
    line-height: 1.6;
}

/* ===== RISK INDICATOR ===== */
.risk-critical {
    color: #ff4444;
    font-weight: 700;
}

.risk-warning {
    color: #ffaa00;
    font-weight: 700;
}

.risk-healthy {
    color: #44ff44;
    font-weight: 700;
}

/* ===== TABLE STYLING ===== */
[data-testid="stDataFrame"] {
    background-color: rgba(22, 33, 62, 0.6) !important;
    border-radius: 8px;
}

/* ===== NAVBAR ===== */
.stRadio > label {
    color: #b0b3b8 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

.stRadio > label > div {
    padding: 10px 16px !important;
    background: rgba(0, 212, 255, 0.05) !important;
    border: 1px solid rgba(0, 212, 255, 0.2) !important;
    border-radius: 8px !important;
    margin: 0 8px 0 0 !important;
}

.stRadio > label > div:hover {
    background: rgba(0, 212, 255, 0.15) !important;
    border-color: rgba(0, 212, 255, 0.5) !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data
def load_csv(path):
    """Load CSV safely"""
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not load {path}: {e}")
    return None

# Load main data
rfm_intel = load_csv(f"{INTELLIGENCE_DIR}/rfm_segment_intelligence.csv")
kmeans_intel = load_csv(f"{INTELLIGENCE_DIR}/kmeans_segment_intelligence.csv")
hc_intel = load_csv(f"{INTELLIGENCE_DIR}/hc_segment_intelligence.csv")

rfm_kpi = load_csv(f"{BUSINESS_KPI_DIR}/rfm_business_kpi.csv")
kmeans_kpi = load_csv(f"{BUSINESS_KPI_DIR}/kmeans_business_kpi.csv")
hc_kpi = load_csv(f"{BUSINESS_KPI_DIR}/hc_business_kpi.csv")
overview_kpi = load_csv(f"{BUSINESS_KPI_DIR}/overview_kpi_summary.csv")
risk_dist = load_csv(f"{BUSINESS_KPI_DIR}/risk_distribution.csv")

# Load prep data
rfm_journey = load_csv(f"{PREP_DIR}/rfm_journey_mapping.csv")
kmeans_enriched = load_csv(f"{PREP_DIR}/kmeans_enriched_intelligence.csv")
hc_enriched = load_csv(f"{PREP_DIR}/hc_enriched_intelligence.csv")
cluster_naming = load_csv(f"{PREP_DIR}/cluster_naming_logic.csv")
hero_kpi = load_csv(f"{PREP_DIR}/hero_page_kpi.csv")

# Load original data
rfm_df = load_csv(f"{RFM_DIR}/customer_rfm_final.csv")
model_comparison = load_csv(f"{CHURN_DIR}/model_comparison.csv")
feature_importance = load_csv(f"{CHURN_DIR}/feature_importance.csv")
personas_df = load_csv(f"{PERSONA_DIR}/segment_personas.csv")

# =========================================================
# NAVBAR
# =========================================================

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

col_nav = st.columns([1, 1, 1, 1, 1, 1, 1])
pages = ["Hero", "Problem", "RFM Intelligence", "Clustering", "Churn", "Personas", "Action"]

section = st.radio(
    label="Navigation",
    options=pages,
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

# =========================================================
# PAGE 1: HERO
# =========================================================

if section == "Hero":
    
    st.markdown('<div class="hero-badge">🧠 AI-POWERED CUSTOMER INTELLIGENCE</div>', unsafe_allow_html=True)
    
    st.markdown("# The Complete Picture")
    st.markdown("### Unified segmentation, predictive intelligence, and actionable insights in a single platform")
    
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    
    # KPI Cards Row 1
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">👥 Total Customers</div>
            <div class="kpi-value">50,000</div>
            <div class="kpi-delta">Analyzed & Segmented</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">💰 Revenue at Risk</div>
            <div class="kpi-value">$19.9M</div>
            <div class="kpi-delta">Immediate action needed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">📈 Recoverable Revenue</div>
            <div class="kpi-value">$3.0M</div>
            <div class="kpi-delta">With retention campaigns</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">✅ Model AUC</div>
            <div class="kpi-value">98%</div>
            <div class="kpi-delta">Churn prediction accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    
    # KPI Cards Row 2
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">🔴 Critical Risk</div>
            <div class="kpi-value">8,009</div>
            <div class="kpi-delta">16% of customer base</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">🟡 Warning Zone</div>
            <div class="kpi-value">21,735</div>
            <div class="kpi-delta">43% at early churn stage</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">🟢 Healthy Base</div>
            <div class="kpi-value">20,256</div>
            <div class="kpi-delta">41% engaged & valuable</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">🎯 Potential Gain</div>
            <div class="kpi-value">$896K</div>
            <div class="kpi-delta">From retention focus</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
    
    # System Overview
    st.markdown("## Platform Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <strong>📊 Segmentation Engine</strong><br>
        RFM-based: 10 segments<br>
        KMeans-based: 4 clusters<br>
        Hierarchical: 7 clusters<br>
        → 100% customer coverage with overlapping validation
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <strong>🤖 Predictive Layer</strong><br>
        Churn prediction: 98% AUC<br>
        Risk scoring: Recency-dominant<br>
        Revenue exposure: Calculated<br>
        → Actionable alerts for each customer
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <strong>🧠 Behavioral Intelligence</strong><br>
        Dominant features per segment<br>
        Product affinity mapping<br>
        Regional & demographic insights<br>
        → Personalized campaign targeting
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <strong>⚙️ Automation Ready</strong><br>
        n8n webhook integration<br>
        Smart audience definitions<br>
        Campaign recommendation engine<br>
        → Operationalized AI workflows
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# PAGE 2: PROBLEM
# =========================================================

elif section == "Problem":
    
    st.markdown('<div class="hero-badge">⚠️ THE CHALLENGE</div>', unsafe_allow_html=True)
    
    st.markdown("# Why Customer Intelligence Matters")
    st.markdown("### The gap between data and action")
    
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    
    # The Problem
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <strong>❌ Invisible Churn Signals</strong><br>
        Customers leave silently. Last purchase is often the only warning signal, and by then it's too late to intervene.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <strong>❌ Generic CRM Campaigns</strong><br>
        One-size-fits-all messaging fails. Without behavioral segmentation, marketing ROI suffers and customer experience degrades.
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <strong>❌ Fragmented Insights</strong><br>
        Data lives in isolation. RFM scores, churn models, personas exist separately—no unified view of each customer.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <strong>❌ Manual Workflows</strong><br>
        Campaign execution is manual and slow. Insights don't translate to immediate action at scale.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    
    # Risk Landscape Visualization
    st.markdown("## Customer Risk Landscape")
    
    if risk_dist is not None:
        fig = go.Figure()
        
        colors_risk = {'🟢 Healthy': '#44ff44', '🟡 Warning': '#ffaa00', '🔴 Critical': '#ff4444'}
        
        for risk_level in ['🟢 Healthy', '🟡 Warning', '🔴 Critical']:
            data = risk_dist[risk_dist['risk_level'] == risk_level]
            if not data.empty:
                fig.add_trace(go.Bar(
                    x=[risk_level],
                    y=data['customer_count'].values,
                    name=risk_level,
                    marker=dict(color=colors_risk[risk_level]),
                    text=data['customer_count'].values,
                    textposition='auto',
                ))
        
        fig.update_layout(
            title="Risk Distribution Across Customer Base",
            xaxis_title="Risk Level",
            yaxis_title="Number of Customers",
            template="plotly_dark",
            height=400,
            showlegend=False,
            hovermode='x unified',
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <strong>The Insight:</strong> 59% of your customer base (29,744 customers) is at risk or in warning stages. 
    This represents <span class="risk-critical">$19.9M in revenue exposure</span>. Without timely intervention, 
    churn will accelerate and revenue will decline.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PAGE 3: RFM INTELLIGENCE
# =========================================================

elif section == "RFM Intelligence":
    
    st.markdown('<div class="hero-badge">📊 BEHAVIORAL SEGMENTATION</div>', unsafe_allow_html=True)
    
    st.markdown("# RFM Segmentation Strategy")
    st.markdown("### Rule-based customer classification: Recency, Frequency, Monetary")
    
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    
    # Journey Selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("## Loyalty Growth Journey")
        if rfm_journey is not None:
            loyalty = rfm_journey[rfm_journey['journey_type'] == 'Loyalty Growth'].sort_values('journey_stage')
            
            for idx, row in loyalty.iterrows():
                emoji = "⭐" if row['RFM_Segment'] == 'champions' else "📈"
                st.markdown(f"""
                <div class="kpi-card">
                    <div>{emoji} <strong>{row['RFM_Segment'].upper()}</strong></div>
                    <div style="font-size:14px; color:#00d4ff;">👥 {row['customer_count']:,} | 💳 {row['purchase_rate_%']:.0f}% Active</div>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("## Churn Risk Journey")
        if rfm_journey is not None:
            churn = rfm_journey[rfm_journey['journey_type'] == 'Churn Risk'].sort_values('journey_stage')
            
            for idx, row in churn.iterrows():
                emoji = "🔴" if row['purchase_rate_%'] == 0.0 else "🟡"
                st.markdown(f"""
                <div class="kpi-card">
                    <div>{emoji} <strong>{row['RFM_Segment'].upper()}</strong></div>
                    <div style="font-size:14px; color:#ff6666;">👥 {row['customer_count']:,} | 💳 {row['purchase_rate_%']:.0f}% Active</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    
    # Full RFM Table
    if rfm_intel is not None:
        st.markdown("## Full RFM Segment Intelligence")
        
        display_cols = ['RFM_Segment', 'customer_count', 'purchase_rate_%', 'risk_level', 
                       'dominant_product_category', 'dominant_region', 'dominant_age_group', 'discount_affinity']
        
        df_display = rfm_intel[[c for c in display_cols if c in rfm_intel.columns]].copy()
        df_display.columns = ['Segment', 'Customers', 'Purchase Rate %', 'Risk Level', 'Top Category', 'Top Region', 'Top Age', 'Discount %']
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)

# =========================================================
# PAGE 4: CLUSTERING
# =========================================================

elif section == "Clustering":
    
    st.markdown('<div class="hero-badge">🔬 ALGORITHMIC VALIDATION</div>', unsafe_allow_html=True)
    
    st.markdown("# Clustering Analysis & Validation")
    st.markdown("### RFM segmentation validated through KMeans & Hierarchical clustering")
    
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    
    # Cluster Naming Logic
    st.markdown("## Cluster Naming Logic (How Are Clusters Named?)")
    
    if cluster_naming is not None:
        logic_display = cluster_naming[['cluster_type', 'naming_basis', 'business_interpretation']].copy()
        logic_display.columns = ['Cluster Type', 'Naming Basis (Score/RFM)', 'Business Interpretation']
        st.dataframe(logic_display, use_container_width=True, hide_index=True)
    
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    
    # KMeans Section
    st.markdown("## KMeans Clustering (4 Clusters)")
    
    if kmeans_enriched is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Risk Distribution")
            kmeans_risk = kmeans_enriched.groupby('journey_type').size()
            fig = go.Figure(data=[go.Pie(labels=kmeans_risk.index, values=kmeans_risk.values, 
                                         marker=dict(colors=['#44ff44', '#ffaa00', '#ff4444']))])
            fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Revenue at Risk per Cluster")
            top_risk = kmeans_enriched.nlargest(3, 'revenue_at_risk')[['cluster_name', 'revenue_at_risk']]
            fig = go.Figure(data=[go.Bar(x=top_risk['cluster_name'], y=top_risk['revenue_at_risk'],
                                         marker=dict(color='#ff6666'))])
            fig.update_layout(template="plotly_dark", height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### KMeans Cluster Profiles")
        display_cols = ['cluster_name', 'customer_count', 'journey_type', 'purchase_rate_%', 
                       'dominant_product_category', 'dominant_region', 'risk_level']
        df_kmeans = kmeans_enriched[[c for c in display_cols if c in kmeans_enriched.columns]].copy()
        df_kmeans.columns = ['Cluster', 'Customers', 'Journey Type', 'Purchase %', 'Top Category', 'Top Region', 'Risk Level']
        st.dataframe(df_kmeans, use_container_width=True, hide_index=True)
    
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    
    # HC Section
    st.markdown("## Hierarchical Clustering (7 Clusters - Validation Layer)")
    
    if hc_enriched is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Risk Distribution")
            hc_risk = hc_enriched.groupby('journey_type').size()
            fig = go.Figure(data=[go.Pie(labels=hc_risk.index, values=hc_risk.values,
                                         marker=dict(colors=['#44ff44', '#ffaa00', '#ff4444']))])
            fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Revenue at Risk per Cluster")
            top_risk_hc = hc_enriched.nlargest(3, 'revenue_at_risk')[['cluster_name', 'revenue_at_risk']]
            fig = go.Figure(data=[go.Bar(x=top_risk_hc['cluster_name'], y=top_risk_hc['revenue_at_risk'],
                                         marker=dict(color='#ff6666'))])
            fig.update_layout(template="plotly_dark", height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### HC Cluster Profiles")
        display_cols = ['cluster_name', 'customer_count', 'journey_type', 'purchase_rate_%',
                       'dominant_product_category', 'dominant_region', 'risk_level']
        df_hc = hc_enriched[[c for c in display_cols if c in hc_enriched.columns]].copy()
        df_hc.columns = ['Cluster', 'Customers', 'Journey Type', 'Purchase %', 'Top Category', 'Top Region', 'Risk Level']
        st.dataframe(df_hc, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="insight-box">
    <strong>Validation Insight:</strong> HC clustering confirms RFM and KMeans patterns, 
    validating our segmentation strategy across three independent methods.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PAGE 5: CHURN PREDICTION
# =========================================================

elif section == "Churn":
    
    st.markdown('<div class="hero-badge">🎯 PREDICTIVE INTELLIGENCE</div>', unsafe_allow_html=True)
    
    st.markdown("# Churn Prediction Engine")
    st.markdown("### Early warning system: Identify at-risk customers before they leave")
    
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    
    # Model Performance
    if model_comparison is not None:
        st.markdown("## Model Performance Comparison")
        
        col1, col2, col3 = st.columns(3)
        
        best_auc = model_comparison['AUC'].max()
        best_model = model_comparison.loc[model_comparison['AUC'].idxmax(), 'Model']
        
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">🏆 Best Model</div>
                <div class="kpi-value" style="color:#00d4ff;">{best_model}</div>
                <div class="kpi-delta">Selected for production</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">📊 AUC Score</div>
                <div class="kpi-value" style="color:#00ff00;">{best_auc:.4f}</div>
                <div class="kpi-delta">Prediction accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">✅ Validation</div>
                <div class="kpi-value" style="color:#00d4ff;">98%</div>
                <div class="kpi-delta">Precision on test set</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        
        # Model Comparison Table
        st.markdown("### All Models Tested")
        st.dataframe(model_comparison, use_container_width=True, hide_index=True)
    
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    
    # Feature Importance
    if feature_importance is not None:
        st.markdown("## Top Churn Drivers (Feature Importance)")
        
        # Sort by importance
        feat_sorted = feature_importance.nlargest(10, 'Importance')
        
        fig = go.Figure(data=[
            go.Bar(x=feat_sorted['Importance'], y=feat_sorted['Feature'], orientation='h',
                  marker=dict(color=feat_sorted['Importance'], colorscale='Reds', showscale=False))
        ])
        
        fig.update_layout(
            title="Features Predicting Customer Churn",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            template="plotly_dark",
            height=400,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <strong>🔑 Key Finding:</strong> Recency (days since last purchase) is the dominant churn predictor. 
    Combined with frequency and monetary value, it creates a powerful early warning system.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PAGE 6: PERSONAS
# =========================================================

elif section == "Personas":
    
    st.markdown('<div class="hero-badge">🧑 CUSTOMER ARCHETYPES</div>', unsafe_allow_html=True)
    
    st.markdown("# Customer Personas")
    st.markdown("### Behavioral profiles enriched with product, regional, and demographic insights")
    
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    
    if personas_df is not None:
        # Segment selector
        segments = sorted(personas_df['Segment'].unique())
        selected_segment = st.selectbox("Select a segment to explore:", segments)
        
        segment_data = personas_df[personas_df['Segment'] == selected_segment]
        
        if not segment_data.empty:
            st.markdown(f"## {selected_segment.upper()}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if 'Persona' in segment_data.columns:
                    persona_text = segment_data['Persona'].values[0]
                    st.markdown(f"""
                    <div class="insight-box">
                    {persona_text}
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if 'Campaign Recommendation' in segment_data.columns:
                    campaign = segment_data['Campaign Recommendation'].values[0]
                    st.markdown(f"""
                    <div class="insight-box">
                    <strong>📧 Campaign Strategy:</strong><br>
                    {campaign}
                    </div>
                    """, unsafe_allow_html=True)

# =========================================================
# PAGE 7: ACTION LAYER
# =========================================================

elif section == "Action":
    
    st.markdown('<div class="hero-badge">⚙️ OPERATIONALIZATION</div>', unsafe_allow_html=True)
    
    st.markdown("# Campaign Action Layer")
    st.markdown("### From insight to action: Smart audience definitions & automation-ready payloads")
    
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    
    # Smart Actions
    smart_actions = {
        "🔴 Immediate Retention": "at_risk / cant_loose segments with high recent activity",
        "⭐ VIP Win-back": "cant_loose with high lifetime value - premium incentives",
        "⚠️ Early Warning": "about_to_sleep / need_attention with medium-high frequency",
        "💰 Discount Recovery": "discount-affinity segments with recent churn signals",
        "🔄 Dormant Reactivation": "hibernating segments - reengagement campaigns"
    }
    
    st.markdown("## Smart Audience Definitions")
    
    cols = st.columns(len(smart_actions))
    for i, (action, description) in enumerate(smart_actions.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{action}</div>
                <div style="font-size:12px; color:#b0b3b8; margin-top:10px;">{description}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    
    # Automation Status
    st.markdown("## Automation Integration")
    
    st.markdown("""
    <div class="insight-box">
    <strong>✅ n8n Webhook Ready</strong><br>
    All segments can trigger automated email campaigns to marketing@fufighters.com<br>
    Payload includes: Customer IDs | Segment info | AI-generated campaign message | Recommended action
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    
    # Export Options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="kpi-card">
            <div style="text-align:center;">
                <div style="font-size:24px;">📊</div>
                <div class="kpi-label">CSV Export</div>
                <div style="font-size:12px; color:#b0b3b8; margin-top:8px;">Customer lists by segment</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <div style="text-align:center;">
                <div style="font-size:24px;">📋</div>
                <div class="kpi-label">JSON Payload</div>
                <div style="font-size:12px; color:#b0b3b8; margin-top:8px;">For API/webhook integration</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="kpi-card">
            <div style="text-align:center;">
                <div style="font-size:24px;">🤖</div>
                <div class="kpi-label">n8n Workflow</div>
                <div style="font-size:12px; color:#b0b3b8; margin-top:8px;">Automated email campaigns</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("<div style='height:50px'></div>", unsafe_allow_html=True)
st.markdown("""
---
<div style='text-align:center; color:#8b9dc3; font-size:12px;'>
AI-Powered Customer Intelligence Platform | Presented: """ + datetime.now().strftime("%B %d, %Y") + """
</div>
""", unsafe_allow_html=True)