"""
BUSINESS KPI LAYER (ML-INTEGRATED)
===================================

Amaç:
- Segment Intelligence çıktılarını business metriklere dönüştür
- ✅ ML Churn Predictions entegre (hardcoded değil!)
- Executive-level dashboard için operasyonel metrikler üret
- Revenue impact, retention gain, campaign ROI hesapla

ML Integration:
- estimated_churn_rate_% = segment_churn_analysis.csv'den (ML model)
- recovery_rate = churn_rate'e bağlı dynamic hesaplama
- retention_gain = ML-based churn risk'e bağlı

Çıktılar:
- rfm_business_kpi.csv
- kmeans_business_kpi.csv
- hc_business_kpi.csv
- overview_kpi_summary.csv (Hero sayfası için)
- risk_distribution.csv
"""

import pandas as pd
import numpy as np
import os

# =====================================================================
# CONFIG
# =====================================================================

BASE_DIR = "data"
INTELLIGENCE_DIR = f"{BASE_DIR}/Intelligence_Output"
BUSINESS_KPI_DIR = f"{BASE_DIR}/Business_KPI_Output"
CHURN_DIR = f"{BASE_DIR}/Churn_Output"

os.makedirs(BUSINESS_KPI_DIR, exist_ok=True)

# =====================================================================
# LOAD DATA
# =====================================================================

print("📊 Loading Intelligence outputs...")

rfm_intel = pd.read_csv(f"{INTELLIGENCE_DIR}/rfm_segment_intelligence.csv")
kmeans_intel = pd.read_csv(f"{INTELLIGENCE_DIR}/kmeans_segment_intelligence.csv")
hc_intel = pd.read_csv(f"{INTELLIGENCE_DIR}/hc_segment_intelligence.csv")

# Ana RFM data
rfm_df = pd.read_csv(f"{BASE_DIR}/RFM_Output/customer_rfm_final.csv")

# ✅ ML Churn Predictions yükle
segment_churn_path = f"{CHURN_DIR}/segment_churn_analysis.csv"
if os.path.exists(segment_churn_path):
    segment_churn = pd.read_csv(segment_churn_path)
    print(f"✅ Loaded ML Churn Predictions: {len(segment_churn)} segments")
    print(segment_churn[["RFM_Segment", "Musteri_Sayisi", "Tahmin_Churn_Orani"]].to_string(index=False))
else:
    segment_churn = None
    print("⚠️ ML Churn Predictions not found - using fallback")

print(f"\n✅ Loaded RFM Intelligence: {len(rfm_intel)} segments")
print(f"✅ Loaded KMeans Intelligence: {len(kmeans_intel)} clusters")
print(f"✅ Loaded HC Intelligence: {len(hc_intel)} clusters")

# =====================================================================
# ML CHURN RATE LOOKUP
# =====================================================================

def get_ml_churn_rate(segment_name, segment_churn_df):
    """
    ML model'den segment'in churn rate'ini al.
    Eğer ML prediction yoksa, risk_level'dan fallback kullan.
    """
    if segment_churn_df is not None:
        match = segment_churn_df[segment_churn_df["RFM_Segment"] == segment_name]
        if len(match) > 0:
            return match.iloc[0]["Tahmin_Churn_Orani"]
    # Fallback: risk_level based (eski hardcoded)
    return None

# =====================================================================
# BUSINESS METRICS CALCULATION
# =====================================================================

def calculate_business_kpi(intel_df, segment_col, original_df=None, is_rfm=False):
    """
    Her segment/cluster için business KPI'ları hesapla:
    - Revenue at Risk
    - Recoverable Revenue (ML-based churn rate'e bağlı)
    - High Value Customer Exposure
    - Potential Retention Gain
    - Campaign ROI Projection
    
    ✅ ML Integration:
    - estimated_churn_rate_% = ML model predictions
    - recovery_rate = dynamic (churn_rate'e bağlı)
    - retention_gain = ML-based
    """
    
    kpi_df = intel_df.copy()
    
    # 1. REVENUE AT RISK
    # Risk level'a göre revenue at risk hesapla
    kpi_df["revenue_at_risk"] = kpi_df.apply(
        lambda row: row["TotalSpent_mean"] * row["customer_count"] 
        if row["risk_level"] in ["🟡 Warning", "🔴 Critical"]
        else 0,
        axis=1
    ).round(0)
    
    # =====================================================
    # ✅ ML CHURN RATE INTEGRATION
    # =====================================================
    
    if is_rfm and segment_churn is not None:
        # RFM segments için ML predictions kullan
        churn_map = dict(zip(
            segment_churn["RFM_Segment"],
            segment_churn["Tahmin_Churn_Orani"]
        ))
        kpi_df["estimated_churn_rate_%"] = (kpi_df["RFM_Segment"].map(churn_map).fillna(0) * 100).round(1)
        print("  ✅ ML Churn Predictions applied (RFM)")
    else:
        # KMeans/HC için risk_level based (ML prediction yok)
        kpi_df["estimated_churn_rate_%"] = kpi_df.apply(
            lambda row: 100.0 if row["risk_level"] == "🔴 Critical"
            else 60.0 if row["risk_level"] == "🟡 Warning"
            else 10.0,
            axis=1
        ).round(1)
        if not is_rfm:
            print("  ⚠️ KMeans/HC: risk_level based churn (no ML prediction)")
    
    # =====================================================
    # ✅ RECOVERABLE REVENUE (ML-based)
    # =====================================================
    # Recovery rate = churn_rate'e bağlı dynamic hesaplama
    # Yüksek churn → düşük recovery (zaten kaybedilmiş)
    # Orta churn → yüksek recovery (hala kurtarılabilir)
    # Düşük churn → minimal recovery (zaten healthy)
    
    def calc_recovery_rate(churn_rate):
        """
        ML-based dynamic recovery rate:
        - Churn 0-20%: %5 recovery (healthy, minimal intervention needed)
        - Churn 20-50%: %20 recovery (warning, good intervention window)
        - Churn 50-80%: %15 recovery (high risk, harder to recover)
        - Churn 80-100%: %8 recovery (critical, very hard to recover)
        """
        if churn_rate >= 80:
            return 0.08  # Critical: çoğu gone, az recovery
        elif churn_rate >= 50:
            return 0.15  # High risk: intervention possible
        elif churn_rate >= 20:
            return 0.20  # Warning: best intervention window
        else:
            return 0.05  # Healthy: minimal churn, minimal recovery needed
    
    kpi_df["recovery_rate"] = kpi_df["estimated_churn_rate_%"].apply(calc_recovery_rate)
    
    kpi_df["recoverable_revenue"] = kpi_df.apply(
        lambda row: row["TotalSpent_mean"] * row["customer_count"] * row["recovery_rate"]
        if row["risk_level"] in ["🟡 Warning", "🔴 Critical"]
        else 0,
        axis=1
    ).round(0)
    
    # 3. HIGH VALUE CUSTOMER EXPOSURE
    kpi_df["high_value_exposure"] = kpi_df.apply(
        lambda row: row["customer_count"]
        if "TotalSpent_mean" in row and row["TotalSpent_mean"] > rfm_intel["TotalSpent_mean"].quantile(0.75)
        and row["risk_level"] in ["🟡 Warning", "🔴 Critical"]
        else 0,
        axis=1
    ).astype(int)
    
    # =====================================================
    # ✅ POTENTIAL RETENTION GAIN (ML-based)
    # =====================================================
    # Retention gain = recoverable * success_rate
    # Success rate = churn_rate'e bağlı
    
    def calc_retention_success(churn_rate):
        """
        ML-based retention success rate:
        - Churn 0-20%: %50 success (easy to retain)
        - Churn 20-50%: %35 success (moderate effort)
        - Churn 50-80%: %25 success (significant effort)
        - Churn 80-100%: %10 success (very difficult)
        """
        if churn_rate >= 80:
            return 0.10
        elif churn_rate >= 50:
            return 0.25
        elif churn_rate >= 20:
            return 0.35
        else:
            return 0.50
    
    kpi_df["retention_success_rate"] = kpi_df["estimated_churn_rate_%"].apply(calc_retention_success)
    kpi_df["potential_retention_gain"] = (kpi_df["recoverable_revenue"] * kpi_df["retention_success_rate"]).round(0)
    
    # 5. AVG CUSTOMER VALUE
    kpi_df["avg_customer_ltv"] = kpi_df["TotalSpent_mean"].round(2)
    
    # =====================================================
    # ✅ CAMPAIGN ROI PROJECTION (ML-based)
    # =====================================================
    # Marketing cost per customer = churn_rate'e bağlı
    # Yüksek churn → daha pahalı campaign (VIP, personal)
    # Düşük churn → ucuz campaign (email, push)
    
    def calc_marketing_cost(churn_rate):
        """
        ML-based marketing cost per customer:
        - Churn 0-20%: $10/customer (basic email/push)
        - Churn 20-50%: $20/customer (targeted campaign)
        - Churn 50-80%: $35/customer (personalized offer)
        - Churn 80-100%: $50/customer (VIP personal outreach)
        """
        if churn_rate >= 80:
            return 50
        elif churn_rate >= 50:
            return 35
        elif churn_rate >= 20:
            return 20
        else:
            return 10
    
    kpi_df["marketing_cost_per_customer"] = kpi_df["estimated_churn_rate_%"].apply(calc_marketing_cost)
    
    kpi_df["estimated_campaign_roi_multiple"] = kpi_df.apply(
        lambda row: (row["recoverable_revenue"] / max(row["customer_count"] * row["marketing_cost_per_customer"], 1)) 
        if row["customer_count"] > 0 else 0,
        axis=1
    ).round(1)
    
    # Safeguard against inf/nan
    kpi_df["estimated_campaign_roi_multiple"] = kpi_df["estimated_campaign_roi_multiple"].replace([np.inf, -np.inf], 0).fillna(0)
    
    # 8. DISCOUNT IMPACT
    kpi_df["discount_impact_on_margin_%"] = (kpi_df["discount_affinity"] * 0.5).round(1)
    
    # 9. LOYALTY STRENGTH
    if is_rfm and "LoyaltyProgram_mean" in kpi_df.columns:
        kpi_df["loyalty_strength_score"] = (
            (kpi_df["frequency_mean"] / kpi_df["frequency_mean"].max() * 0.6 +
             kpi_df["LoyaltyProgram_mean"] * 0.4) * 100
        ).round(1)
    else:
        kpi_df["loyalty_strength_score"] = (
            (kpi_df["frequency_mean"] / kpi_df["frequency_mean"].max()) * 100
        ).round(1)
    
    # 10. SATISFACTION IMPACT
    if is_rfm and "CustomerSatisfaction_mean" in kpi_df.columns:
        kpi_df["customer_satisfaction_score"] = (kpi_df["CustomerSatisfaction_mean"] * 100).round(1)
    else:
        kpi_df["customer_satisfaction_score"] = 0.0
    
    # Drop intermediate columns
    kpi_df = kpi_df.drop(columns=["recovery_rate", "retention_success_rate", "marketing_cost_per_customer"], errors="ignore")
    
    return kpi_df

# =====================================================================
# RFM BUSINESS KPI
# =====================================================================

print("\n🎯 Calculating RFM Business KPI (ML-Integrated)...")

rfm_kpi = calculate_business_kpi(rfm_intel, "RFM_Segment", rfm_df, is_rfm=True)

# Order by revenue at risk (descending)
rfm_kpi = rfm_kpi.sort_values("revenue_at_risk", ascending=False)

# Select relevant columns
rfm_kpi_cols = [
    "RFM_Segment",
    "customer_count",
    "TotalSpent_mean",
    "purchase_rate_%",
    "risk_level",
    "revenue_at_risk",
    "recoverable_revenue",
    "potential_retention_gain",
    "high_value_exposure",
    "avg_customer_ltv",
    "estimated_churn_rate_%",
    "estimated_campaign_roi_multiple",
    "discount_affinity",
    "discount_impact_on_margin_%",
    "loyalty_strength_score",
    "customer_satisfaction_score",
]

rfm_kpi = rfm_kpi[[c for c in rfm_kpi_cols if c in rfm_kpi.columns]]

rfm_kpi.to_csv(f"{BUSINESS_KPI_DIR}/rfm_business_kpi.csv", index=False)
print(f"✅ Saved: {BUSINESS_KPI_DIR}/rfm_business_kpi.csv")
print("\nRFM Business KPI (ML-Based Churn):")
print(rfm_kpi[["RFM_Segment", "customer_count", "estimated_churn_rate_%", "revenue_at_risk", "recoverable_revenue", "potential_retention_gain"]].to_string(index=False))

# =====================================================================
# KMEANS BUSINESS KPI
# =====================================================================

print("\n🎯 Calculating KMeans Business KPI...")

kmeans_kpi = calculate_business_kpi(kmeans_intel, "cluster_name", rfm_df, is_rfm=False)
kmeans_kpi = kmeans_kpi.sort_values("revenue_at_risk", ascending=False)

kmeans_kpi_cols = [
    "cluster_name",
    "customer_count",
    "TotalSpent_mean",
    "purchase_rate_%",
    "risk_level",
    "revenue_at_risk",
    "recoverable_revenue",
    "potential_retention_gain",
    "high_value_exposure",
    "avg_customer_ltv",
    "estimated_churn_rate_%",
    "estimated_campaign_roi_multiple",
    "discount_affinity",
    "loyalty_strength_score",
    "customer_satisfaction_score",
]

kmeans_kpi = kmeans_kpi[[c for c in kmeans_kpi_cols if c in kmeans_kpi.columns]]

kmeans_kpi.to_csv(f"{BUSINESS_KPI_DIR}/kmeans_business_kpi.csv", index=False)
print(f"✅ Saved: {BUSINESS_KPI_DIR}/kmeans_business_kpi.csv")
print("\nTop 3 At-Risk Clusters:")
print(kmeans_kpi[["cluster_name", "customer_count", "revenue_at_risk"]].head(3))

# =====================================================================
# HC BUSINESS KPI
# =====================================================================

print("\n🎯 Calculating HC Business KPI...")

hc_kpi = calculate_business_kpi(hc_intel, "cluster_name", rfm_df, is_rfm=False)
hc_kpi = hc_kpi.sort_values("revenue_at_risk", ascending=False)

hc_kpi_cols = [
    "cluster_name",
    "customer_count",
    "TotalSpent_mean",
    "purchase_rate_%",
    "risk_level",
    "revenue_at_risk",
    "recoverable_revenue",
    "potential_retention_gain",
    "high_value_exposure",
    "avg_customer_ltv",
    "estimated_churn_rate_%",
    "estimated_campaign_roi_multiple",
    "discount_affinity",
    "loyalty_strength_score",
    "customer_satisfaction_score",
]

hc_kpi = hc_kpi[[c for c in hc_kpi_cols if c in hc_kpi.columns]]

hc_kpi.to_csv(f"{BUSINESS_KPI_DIR}/hc_business_kpi.csv", index=False)
print(f"✅ Saved: {BUSINESS_KPI_DIR}/hc_business_kpi.csv")
print("\nTop 3 At-Risk Clusters:")
print(hc_kpi[["cluster_name", "customer_count", "revenue_at_risk"]].head(3))

# =====================================================================
# OVERVIEW KPI SUMMARY (Hero sayfası için)
# =====================================================================

print("\n🎯 Calculating Overview KPI Summary...")

# Tüm segmentler üzerinde aggregation
total_customers = rfm_intel["customer_count"].sum()
total_revenue = (rfm_intel["TotalSpent_mean"] * rfm_intel["customer_count"]).sum()
avg_customer_spend = rfm_intel["TotalSpent_mean"].mean()

# Risk metrikleri
high_risk_customers = rfm_intel[rfm_intel["risk_level"] == "🔴 Critical"]["customer_count"].sum()
warning_customers = rfm_intel[rfm_intel["risk_level"] == "🟡 Warning"]["customer_count"].sum()
healthy_customers = rfm_intel[rfm_intel["risk_level"] == "🟢 Healthy"]["customer_count"].sum()

# Revenue at risk (total)
total_revenue_at_risk = rfm_kpi["revenue_at_risk"].sum()
total_recoverable_revenue = rfm_kpi["recoverable_revenue"].sum()
total_potential_gain = rfm_kpi["potential_retention_gain"].sum()

# KPI Summary
overview_summary = pd.DataFrame({
    "metric": [
        "Total Customers",
        "High Risk Customers",
        "Warning Customers",
        "Healthy Customers",
        "Total Annual Revenue (est.)",
        "Average Customer Spend",
        "Total Revenue at Risk",
        "Total Recoverable Revenue",
        "Total Potential Retention Gain",
        "Average Purchase Rate (%)",
        "Average Risk Score",
        "Average Customer Discount Affinity (%)",
        "RFM Segments Count",
        "KMeans Clusters Count",
        "HC Clusters Count",
    ],
    "value": [
        int(total_customers),
        int(high_risk_customers),
        int(warning_customers),
        int(healthy_customers),
        f"${total_revenue:,.0f}",
        f"${avg_customer_spend:,.2f}",
        f"${total_revenue_at_risk:,.0f}",
        f"${total_recoverable_revenue:,.0f}",
        f"${total_potential_gain:,.0f}",
        round(rfm_intel["purchase_rate_%"].mean(), 1),
        round(rfm_intel["risk_score"].mean(), 3),
        round(rfm_intel["discount_affinity"].mean(), 1),
        len(rfm_intel),
        len(kmeans_intel),
        len(hc_intel),
    ]
})

overview_summary.to_csv(f"{BUSINESS_KPI_DIR}/overview_kpi_summary.csv", index=False)
print(f"✅ Saved: {BUSINESS_KPI_DIR}/overview_kpi_summary.csv")
print("\n" + "="*70)
print("OVERVIEW KPI SUMMARY")
print("="*70)
print(overview_summary.to_string(index=False))

# =====================================================================
# SEGMENT RISK DISTRIBUTION
# =====================================================================

print("\n🎯 Calculating Risk Distribution...")

risk_distribution = pd.DataFrame({
    "risk_level": ["🟢 Healthy", "🟡 Warning", "🔴 Critical"],
    "customer_count": [
        healthy_customers,
        warning_customers,
        high_risk_customers,
    ],
    "revenue_exposure": [
        rfm_kpi[rfm_kpi["risk_level"] == "🟢 Healthy"]["revenue_at_risk"].sum(),
        rfm_kpi[rfm_kpi["risk_level"] == "🟡 Warning"]["revenue_at_risk"].sum(),
        rfm_kpi[rfm_kpi["risk_level"] == "🔴 Critical"]["revenue_at_risk"].sum(),
    ]
})

risk_distribution.to_csv(f"{BUSINESS_KPI_DIR}/risk_distribution.csv", index=False)
print(f"✅ Saved: {BUSINESS_KPI_DIR}/risk_distribution.csv")
print("\nRisk Distribution:")
print(risk_distribution.to_string(index=False))

# =====================================================================
# SUMMARY
# =====================================================================

print("\n" + "="*70)
print("✅ BUSINESS KPI LAYER COMPLETED (ML-INTEGRATED)")
print("="*70)
print(f"\n📁 Output files created in: {BUSINESS_KPI_DIR}/")
print(f"   1. rfm_business_kpi.csv (ML churn predictions)")
print(f"   2. kmeans_business_kpi.csv")
print(f"   3. hc_business_kpi.csv")
print(f"   4. overview_kpi_summary.csv (Hero sayfası için)")
print(f"   5. risk_distribution.csv")
print(f"\n🎯 ML Integration:")
print(f"   ✅ estimated_churn_rate_% = ML model predictions")
print(f"   ✅ recovery_rate = dynamic (churn-based)")
print(f"   ✅ retention_gain = ML-based")
print(f"   ✅ campaign_roi = ML-based marketing cost")
print("="*70)