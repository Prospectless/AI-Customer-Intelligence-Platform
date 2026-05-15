"""
MASTER PREP SCRIPT - FIXED
==========================

Amaç:
- Journey tables (RFM Loyalty Growth + Churn Risk)
- Enriched KMeans/HC intelligence
- Cluster naming logic explanation
- Presentation app için ready CSV'ler

KMeans cluster_name is INTEGER - convert to string!
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = "data"
INTELLIGENCE_DIR = f"{BASE_DIR}/Intelligence_Output"
BUSINESS_KPI_DIR = f"{BASE_DIR}/Business_KPI_Output"
PREP_DIR = f"{BASE_DIR}/Presentation_Prep"

os.makedirs(PREP_DIR, exist_ok=True)

print("📊 Loading all intelligence outputs...")

rfm_intel = pd.read_csv(f"{INTELLIGENCE_DIR}/rfm_segment_intelligence.csv")
kmeans_intel = pd.read_csv(f"{INTELLIGENCE_DIR}/kmeans_segment_intelligence.csv")
hc_intel = pd.read_csv(f"{INTELLIGENCE_DIR}/hc_segment_intelligence.csv")

rfm_kpi = pd.read_csv(f"{BUSINESS_KPI_DIR}/rfm_business_kpi.csv")
kmeans_kpi = pd.read_csv(f"{BUSINESS_KPI_DIR}/kmeans_business_kpi.csv")
hc_kpi = pd.read_csv(f"{BUSINESS_KPI_DIR}/hc_business_kpi.csv")
overview_kpi = pd.read_csv(f"{BUSINESS_KPI_DIR}/overview_kpi_summary.csv")
risk_dist = pd.read_csv(f"{BUSINESS_KPI_DIR}/risk_distribution.csv")

print(f"✅ Loaded all intelligence & KPI data")

# =====================================================================
# 1. RFM JOURNEY MAPPING
# =====================================================================

print("\n🎯 Creating RFM Journey Mapping...")

loyalty_growth_journey = [
    "new_customers",
    "promising",
    "potential_loyalists",
    "loyal_customers",
    "champions",
]

churn_risk_journey = [
    "need_attention",
    "about_to_sleep",
    "at_risk",
    "hibernating",
    "cant_loose"
]

rfm_journey = rfm_intel[["RFM_Segment", "customer_count", "purchase_rate_%", "risk_level"]].copy()

# Assign journey type
rfm_journey["journey_type"] = rfm_journey["RFM_Segment"].apply(
    lambda x: "Loyalty Growth" if x in loyalty_growth_journey else "Churn Risk"
)

# Assign journey stage
def get_loyalty_stage(segment):
    if segment == "new_customers": return 1
    elif segment == "promising": return 2
    elif segment == "potential_loyalists": return 3
    elif segment == "loyal_customers": return 4
    elif segment == "champions": return 5
    else: return 0

def get_churn_stage(segment):
    if segment == "need_attention": return 1
    elif segment == "about_to_sleep": return 2
    elif segment == "at_risk": return 3
    elif segment == "hibernating": return 4
    elif segment == "cant_loose": return 5
    else: return 0

rfm_journey["journey_stage"] = rfm_journey.apply(
    lambda row: (get_loyalty_stage(row["RFM_Segment"]) 
                 if row["journey_type"] == "Loyalty Growth" 
                 else get_churn_stage(row["RFM_Segment"])),
    axis=1
)

# Sort by journey
rfm_journey = rfm_journey.sort_values(["journey_type", "journey_stage"])

rfm_journey.to_csv(f"{PREP_DIR}/rfm_journey_mapping.csv", index=False)
print(f"✅ Saved: {PREP_DIR}/rfm_journey_mapping.csv")
print("\nLoyalty Growth Journey:")
loyalty = rfm_journey[rfm_journey["journey_type"] == "Loyalty Growth"][["RFM_Segment", "customer_count", "purchase_rate_%"]]
print(loyalty.to_string(index=False))
print("\nChurn Risk Journey:")
churn = rfm_journey[rfm_journey["journey_type"] == "Churn Risk"][["RFM_Segment", "customer_count", "purchase_rate_%"]]
print(churn.to_string(index=False))

# =====================================================================
# 2. KMEANS ENRICHED INTELLIGENCE
# =====================================================================

print("\n🎯 Creating KMeans Enriched Intelligence...")

kmeans_enriched = kmeans_intel.copy()

# Add business metrics from KPI
kmeans_enriched = kmeans_enriched.merge(
    kmeans_kpi[["cluster_name", "revenue_at_risk", "recoverable_revenue", 
                "potential_retention_gain", "loyalty_strength_score", "estimated_churn_rate_%"]],
    on="cluster_name",
    how="left"
)

# Add journey mapping - FIX: Convert cluster_name to string first!
def assign_kmeans_journey(name):
    name = str(name)  # IMPORTANT: Convert integer to string
    if "Pasif" in name or "Dormant" in name or name in ["4", "5"]:
        return "At Risk"
    elif "Kaybolmakta" in name or "Low" in name or name == "1":
        return "Warning"
    elif "Aktif" in name or "High Value" in name or name in ["0", "2", "3"]:
        return "Healthy"
    else:
        return "Active"

kmeans_enriched["journey_type"] = kmeans_enriched["cluster_name"].apply(assign_kmeans_journey)

# Reorder columns
cols_order = [
    "cluster_name",
    "journey_type",
    "customer_count",
    "TotalSpent_mean",
    "purchase_rate_%",
    "risk_level",
    "dominant_product_category",
    "dominant_region",
    "dominant_age_group",
    "dominant_device",
    "discount_affinity",
    "revenue_at_risk",
    "recoverable_revenue",
    "potential_retention_gain",
    "loyalty_strength_score",
    "estimated_churn_rate_%"
]

kmeans_enriched = kmeans_enriched[[c for c in cols_order if c in kmeans_enriched.columns]]

kmeans_enriched.to_csv(f"{PREP_DIR}/kmeans_enriched_intelligence.csv", index=False)
print(f"✅ Saved: {PREP_DIR}/kmeans_enriched_intelligence.csv")
print("\nKMeans Enriched Summary:")
print(kmeans_enriched[["cluster_name", "journey_type", "customer_count", "revenue_at_risk"]].to_string(index=False))

# =====================================================================
# 3. HC ENRICHED INTELLIGENCE
# =====================================================================

print("\n🎯 Creating HC Enriched Intelligence...")

hc_enriched = hc_intel.copy()

# Add business metrics from KPI
hc_enriched = hc_enriched.merge(
    hc_kpi[["cluster_name", "revenue_at_risk", "recoverable_revenue", 
             "potential_retention_gain", "loyalty_strength_score", "estimated_churn_rate_%"]],
    on="cluster_name",
    how="left"
)

# Add journey mapping - FIX: Convert cluster_name to string first!
def assign_hc_journey(name):
    name = str(name)  # IMPORTANT: Convert to string (though HC names are already strings)
    if "Pasif" in name or "Dormant" in name:
        return "At Risk"
    elif "Kaybolmakta" in name or "Low" in name:
        return "Warning"
    elif "Aktif" in name or "Yüksek" in name:
        return "Healthy"
    else:
        return "Active"

hc_enriched["journey_type"] = hc_enriched["cluster_name"].apply(assign_hc_journey)

# Reorder columns
cols_order_hc = [
    "cluster_name",
    "journey_type",
    "customer_count",
    "TotalSpent_mean",
    "purchase_rate_%",
    "risk_level",
    "dominant_product_category",
    "dominant_region",
    "dominant_age_group",
    "dominant_device",
    "discount_affinity",
    "revenue_at_risk",
    "recoverable_revenue",
    "potential_retention_gain",
    "loyalty_strength_score",
    "estimated_churn_rate_%"
]

hc_enriched = hc_enriched[[c for c in cols_order_hc if c in hc_enriched.columns]]

hc_enriched.to_csv(f"{PREP_DIR}/hc_enriched_intelligence.csv", index=False)
print(f"✅ Saved: {PREP_DIR}/hc_enriched_intelligence.csv")
print("\nHC Enriched Summary:")
print(hc_enriched[["cluster_name", "journey_type", "customer_count", "revenue_at_risk"]].to_string(index=False))

# =====================================================================
# 4. CLUSTER NAMING LOGIC EXPLANATION
# =====================================================================

print("\n🎯 Creating Cluster Naming Logic...")

naming_logic = pd.DataFrame({
    "cluster_type": [
        "RFM_champions", 
        "RFM_hibernating", 
        "KMeans_Healthy", 
        "KMeans_At_Risk",
        "HC_VIP Active", 
        "HC_Dormant"
    ],
    "naming_basis": [
        "Low Recency + High Frequency + High Monetary (active, engaged, valuable)",
        "High Recency + Low Frequency + Low Monetary (lost customers, high churn risk)",
        "Risk Score > 0.6: Good recency, high frequency/monetary",
        "Risk Score < 0.3: Poor recency, low frequency/monetary",
        "Score > 0.75: Very recent, frequent, high-value purchases",
        "Score < 0.2: Dormant, minimal engagement across all metrics"
    ],
    "calculation_source": [
        "RFM Scores", 
        "RFM Scores", 
        "KMeans Clustering + RFM Normalization",
        "KMeans Clustering + RFM Normalization", 
        "HC Clustering + RFM Normalization", 
        "HC Clustering + RFM Normalization"
    ],
    "business_interpretation": [
        "VIP customers, core revenue drivers, high retention priority",
        "Churned customers, recovery campaign needed",
        "Active, high-value segment worth engagement",
        "At-risk segment, needs intervention",
        "Highest value, most engaged, best ROI",
        "Lost segment, lowest engagement and value"
    ]
})

naming_logic.to_csv(f"{PREP_DIR}/cluster_naming_logic.csv", index=False)
print(f"✅ Saved: {PREP_DIR}/cluster_naming_logic.csv")
print("\nCluster Naming Logic:")
print(naming_logic.to_string(index=False))

# =====================================================================
# 5. PRESENTATION READY SUMMARY (Hero Page Quick Stats)
# =====================================================================

print("\n🎯 Creating Presentation Ready Summary...")

hero_stats = pd.DataFrame({
    "kpi": [
        "Total Customers",
        "Revenue at Risk",
        "Recoverable Revenue",
        "Potential Retention Gain",
        "High Risk Customers",
        "Warning Customers",
        "Healthy Customers",
        "Average Purchase Rate",
        "Model AUC Score"
    ],
    "value": [
        "491,705",
        "$197.7M",
        "$29.7M",
        "$8.9M",
        "78,169",
        "214,816",
        "198,720",
        "45.2%",
        "98%"
    ],
    "emoji": ["👥", "⚠️", "💰", "📈", "🔴", "🟡", "🟢", "📊", "✅"]
})

hero_stats.to_csv(f"{PREP_DIR}/hero_page_kpi.csv", index=False)
print(f"✅ Saved: {PREP_DIR}/hero_page_kpi.csv")
print("\nHero Page Stats:")
print(hero_stats.to_string(index=False))

# =====================================================================
# 6. SEGMENT FEATURES QUICK REFERENCE
# =====================================================================

print("\n🎯 Creating Segment Features Quick Reference...")

# RFM features
rfm_features = rfm_intel[[
    "RFM_Segment",
    "customer_count",
    "dominant_product_category",
    "dominant_region",
    "dominant_age_group",
    "dominant_device",
    "discount_affinity"
]].copy()

rfm_features.to_csv(f"{PREP_DIR}/rfm_segment_features.csv", index=False)
print(f"✅ Saved: {PREP_DIR}/rfm_segment_features.csv")

# KMeans features
kmeans_features = kmeans_intel[[
    "cluster_name",
    "customer_count",
    "dominant_product_category",
    "dominant_region",
    "dominant_age_group",
    "dominant_device",
    "discount_affinity"
]].copy()

kmeans_features.to_csv(f"{PREP_DIR}/kmeans_segment_features.csv", index=False)
print(f"✅ Saved: {PREP_DIR}/kmeans_segment_features.csv")

# HC features
hc_features = hc_intel[[
    "cluster_name",
    "customer_count",
    "dominant_product_category",
    "dominant_region",
    "dominant_age_group",
    "dominant_device",
    "discount_affinity"
]].copy()

hc_features.to_csv(f"{PREP_DIR}/hc_segment_features.csv", index=False)
print(f"✅ Saved: {PREP_DIR}/hc_segment_features.csv")

# =====================================================================
# SUMMARY
# =====================================================================

print("\n" + "="*70)
print("✅ MASTER PREP SCRIPT COMPLETED")
print("="*70)
print(f"\n📁 All files ready in: {PREP_DIR}/")
print(f"\n📋 Generated Files:")
print(f"   1. rfm_journey_mapping.csv (Loyalty Growth + Churn Risk)")
print(f"   2. kmeans_enriched_intelligence.csv (with business metrics)")
print(f"   3. hc_enriched_intelligence.csv (with business metrics)")
print(f"   4. cluster_naming_logic.csv (explanation table)")
print(f"   5. hero_page_kpi.csv (quick stats for hero page)")
print(f"   6. rfm_segment_features.csv (dominant features)")
print(f"   7. kmeans_segment_features.csv (dominant features)")
print(f"   8. hc_segment_features.csv (dominant features)")
print(f"\n🚀 Ready for: presentation_app.py integration!")
print("="*70)