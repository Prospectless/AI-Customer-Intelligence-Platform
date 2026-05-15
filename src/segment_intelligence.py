"""
SEGMENT INTELLIGENCE ENGINE - FULL 491K DATASET (WORKING)
"""
import pandas as pd
import numpy as np
import os

BASE_DIR = "data"
RFM_DIR = f"{BASE_DIR}/RFM_Output"
KMEANS_DIR = f"{BASE_DIR}/KMeans_Output"
HC_DIR = f"{BASE_DIR}/HC_Output"
INTELLIGENCE_DIR = f"{BASE_DIR}/Intelligence_Output"

os.makedirs(INTELLIGENCE_DIR, exist_ok=True)

print("📊 Loading datasets...")

rfm_df = pd.read_csv(f"{RFM_DIR}/customer_rfm_final.csv")
kmeans_df = pd.read_csv(f"{KMEANS_DIR}/customer_kmeans_rfm_final.csv")
three_method = pd.read_csv(f"{HC_DIR}/three_method_comparison.csv")

print(f"✅ Loaded {len(rfm_df):,} customers from RFM (FULL 491K)")
print(f"✅ Loaded {len(kmeans_df):,} customers from KMeans (FULL 491K)")
print(f"✅ Loaded {len(three_method):,} customers from three_method (HC sample 50K)")

def get_age_group(age):
    if pd.isna(age):
        return "Unknown"
    if age < 25: return "18-24"
    elif age < 35: return "25-34"
    elif age < 45: return "35-44"
    elif age < 55: return "45-54"
    elif age < 65: return "55-64"
    else: return "65+"

def assign_risk_level(score):
    if score < 0.33:
        return "🟢 Healthy"
    elif score < 0.66:
        return "🟡 Warning"
    else:
        return "🔴 Critical"

# =====================================================================
# RFM SEGMENT INTELLIGENCE - FULL 491K
# =====================================================================

print("\n🎯 Processing RFM Segment Intelligence (FULL 491K)...")

rfm_intelligence = rfm_df.groupby("RFM_Segment").agg({
    "recency": ["mean", "median"],
    "frequency": ["mean", "median"],
    "monetary": ["mean", "median"],
    "LastPurchaseDaysAgo": "mean",
    "TotalSpent": ["mean", "sum"],
    "DiscountsAvailed": "mean",
    "CustomerTenureYears": "mean",
    "AnnualIncome": "mean",
    "Age": "mean",
    "PurchaseStatus": ["mean", "sum"],
    "LoyaltyProgram": "mean",
    "CustomerSatisfaction": "mean",
    "NumberOfPurchases": "mean",
    "SessionCount": "mean",
}).round(2)

rfm_intelligence.columns = ["_".join(col).strip() for col in rfm_intelligence.columns.values]
rfm_intelligence["customer_count"] = rfm_df.groupby("RFM_Segment").size()

rfm_intelligence["dominant_product_category"] = rfm_df.groupby("RFM_Segment")["ProductCategory"].apply(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown"
)
rfm_intelligence["dominant_region"] = rfm_df.groupby("RFM_Segment")["Region"].apply(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown"
)
rfm_intelligence["dominant_device"] = rfm_df.groupby("RFM_Segment")["PreferredDevice"].apply(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown"
)
rfm_intelligence["dominant_gender"] = rfm_df.groupby("RFM_Segment")["Gender"].apply(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown"
)
rfm_intelligence["dominant_referral_source"] = rfm_df.groupby("RFM_Segment")["ReferralSource"].apply(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown"
)

rfm_intelligence["discount_affinity"] = (
    rfm_df.groupby("RFM_Segment")["DiscountsAvailed"].apply(lambda x: (x > 0).sum()) / 
    rfm_df.groupby("RFM_Segment").size() * 100
).round(1)

rfm_df["age_group"] = rfm_df["Age"].apply(get_age_group)
rfm_intelligence["dominant_age_group"] = rfm_df.groupby("RFM_Segment")["age_group"].apply(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown"
)

rfm_intelligence["risk_score"] = (
    (rfm_intelligence["recency_mean"] / rfm_intelligence["recency_mean"].max()) * 0.5 +
    (1 - rfm_intelligence["frequency_mean"] / rfm_intelligence["frequency_mean"].max()) * 0.3 +
    (1 - rfm_intelligence["monetary_mean"] / rfm_intelligence["monetary_mean"].max()) * 0.2
).round(3)

rfm_intelligence["risk_level"] = rfm_intelligence["risk_score"].apply(assign_risk_level)
rfm_intelligence["purchase_rate_%"] = (rfm_intelligence["PurchaseStatus_mean"] * 100).round(1)

rfm_intelligence = rfm_intelligence.reset_index()

cols_order = [
    "RFM_Segment", "customer_count", "recency_mean", "frequency_mean", "monetary_mean",
    "LastPurchaseDaysAgo_mean", "TotalSpent_mean", "purchase_rate_%", "PurchaseStatus_sum",
    "risk_score", "risk_level", "dominant_product_category", "dominant_region", "dominant_age_group",
    "dominant_device", "dominant_gender", "dominant_referral_source", "discount_affinity",
    "LoyaltyProgram_mean", "CustomerSatisfaction_mean", "AnnualIncome_mean", "Age_mean",
]

rfm_intelligence = rfm_intelligence[[c for c in cols_order if c in rfm_intelligence.columns]]
rfm_intelligence.to_csv(f"{INTELLIGENCE_DIR}/rfm_segment_intelligence.csv", index=False)

print(f"✅ Saved: rfm_segment_intelligence.csv ({len(rfm_intelligence)} segments)")
print(f"\n{rfm_intelligence[['RFM_Segment', 'customer_count', 'purchase_rate_%', 'risk_level']].to_string()}")

# =====================================================================
# KMEANS SEGMENT INTELLIGENCE - FULL 491K
# =====================================================================

print("\n🎯 Processing KMeans Segment Intelligence (FULL 491K)...")

# Merge - DON'T include recency, frequency, monetary (already in kmeans_df)
kmeans_full = kmeans_df 

# recency, frequency, monetary already in kmeans_full from kmeans_df!
kmeans_intelligence = kmeans_full.groupby("KMeans_Cluster").agg({
    "recency": ["mean", "median"],
    "frequency": ["mean", "median"],
    "monetary": ["mean", "median"],
    "TotalSpent": ["mean", "sum"],
    "PurchaseStatus": "mean",
}).round(2)

kmeans_intelligence.columns = ["_".join(col).strip() for col in kmeans_intelligence.columns.values]
kmeans_intelligence["customer_count"] = kmeans_full.groupby("KMeans_Cluster").size()

kmeans_intelligence["dominant_product_category"] = kmeans_full.groupby("KMeans_Cluster")["ProductCategory"].apply(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown"
)
kmeans_intelligence["dominant_region"] = kmeans_full.groupby("KMeans_Cluster")["Region"].apply(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown"
)
kmeans_intelligence["dominant_device"] = kmeans_full.groupby("KMeans_Cluster")["PreferredDevice"].apply(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown"
)
kmeans_intelligence["dominant_gender"] = kmeans_full.groupby("KMeans_Cluster")["Gender"].apply(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown"
)

kmeans_intelligence["discount_affinity"] = (
    kmeans_full.groupby("KMeans_Cluster")["DiscountsAvailed"].apply(lambda x: (x > 0).sum()) / 
    kmeans_full.groupby("KMeans_Cluster").size() * 100
).round(1)

kmeans_full["age_group"] = kmeans_full["Age"].apply(get_age_group)
kmeans_intelligence["dominant_age_group"] = kmeans_full.groupby("KMeans_Cluster")["age_group"].apply(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown"
)

kmeans_intelligence["risk_score"] = (
    (kmeans_intelligence["recency_mean"] / kmeans_intelligence["recency_mean"].max()) * 0.5 +
    (1 - kmeans_intelligence["frequency_mean"] / kmeans_intelligence["frequency_mean"].max()) * 0.3 +
    (1 - kmeans_intelligence["monetary_mean"] / kmeans_intelligence["monetary_mean"].max()) * 0.2
).round(3)

kmeans_intelligence["risk_level"] = kmeans_intelligence["risk_score"].apply(assign_risk_level)
kmeans_intelligence["purchase_rate_%"] = (kmeans_intelligence["PurchaseStatus_mean"] * 100).round(1)

kmeans_intelligence = kmeans_intelligence.reset_index()
kmeans_intelligence.rename(columns={"KMeans_Cluster": "cluster_name"}, inplace=True)

cols_kmeans = ["cluster_name", "customer_count", "recency_mean", "frequency_mean", "monetary_mean",
               "TotalSpent_mean", "purchase_rate_%", "risk_score", "risk_level",
               "dominant_product_category", "dominant_region", "dominant_age_group",
               "dominant_device", "discount_affinity"]

kmeans_intelligence = kmeans_intelligence[[c for c in cols_kmeans if c in kmeans_intelligence.columns]]
kmeans_intelligence.to_csv(f"{INTELLIGENCE_DIR}/kmeans_segment_intelligence.csv", index=False)

print(f"✅ Saved: kmeans_segment_intelligence.csv ({len(kmeans_intelligence)} clusters)")
print(f"\n{kmeans_intelligence[['cluster_name', 'customer_count', 'purchase_rate_%', 'risk_level']].to_string()}")

# =====================================================================
# HC SEGMENT INTELLIGENCE - 50K VALIDATION
# =====================================================================

print("\n🎯 Processing HC Segment Intelligence (50K validation)...")

hc_full = three_method.merge(
    rfm_df[["CustomerID", "ProductCategory", "Region", "PreferredDevice", "Gender", "ReferralSource",
             "Age", "DiscountsAvailed", "LoyaltyProgram", "CustomerSatisfaction", "AnnualIncome",
             "NumberOfPurchases", "SessionCount", "CustomerTenureYears"]],
    on="CustomerID", how="left"
)

hc_intelligence = hc_full.groupby("HC_Cluster_Adi").agg({
    "recency": ["mean", "median"],
    "frequency": ["mean", "median"],
    "monetary": ["mean", "median"],
    "TotalSpent": ["mean", "sum"],
    "PurchaseStatus": "mean",
}).round(2)

hc_intelligence.columns = ["_".join(col).strip() for col in hc_intelligence.columns.values]
hc_intelligence["customer_count"] = hc_full.groupby("HC_Cluster_Adi").size()

hc_intelligence["dominant_product_category"] = hc_full.groupby("HC_Cluster_Adi")["ProductCategory"].apply(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown"
)
hc_intelligence["dominant_region"] = hc_full.groupby("HC_Cluster_Adi")["Region"].apply(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown"
)
hc_intelligence["dominant_device"] = hc_full.groupby("HC_Cluster_Adi")["PreferredDevice"].apply(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown"
)

hc_intelligence["discount_affinity"] = (
    hc_full.groupby("HC_Cluster_Adi")["DiscountsAvailed"].apply(lambda x: (x > 0).sum()) / 
    hc_full.groupby("HC_Cluster_Adi").size() * 100
).round(1)

hc_full["age_group"] = hc_full["Age"].apply(get_age_group)
hc_intelligence["dominant_age_group"] = hc_full.groupby("HC_Cluster_Adi")["age_group"].apply(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown"
)

hc_intelligence["risk_score"] = (
    (hc_intelligence["recency_mean"] / hc_intelligence["recency_mean"].max()) * 0.5 +
    (1 - hc_intelligence["frequency_mean"] / hc_intelligence["frequency_mean"].max()) * 0.3 +
    (1 - hc_intelligence["monetary_mean"] / hc_intelligence["monetary_mean"].max()) * 0.2
).round(3)

hc_intelligence["risk_level"] = hc_intelligence["risk_score"].apply(assign_risk_level)
hc_intelligence["purchase_rate_%"] = (hc_intelligence["PurchaseStatus_mean"] * 100).round(1)

hc_intelligence = hc_intelligence.reset_index()
hc_intelligence.rename(columns={"HC_Cluster_Adi": "cluster_name"}, inplace=True)

cols_hc = ["cluster_name", "customer_count", "recency_mean", "frequency_mean", "monetary_mean",
           "TotalSpent_mean", "purchase_rate_%", "risk_score", "risk_level",
           "dominant_product_category", "dominant_region", "dominant_age_group",
           "dominant_device", "discount_affinity"]

hc_intelligence = hc_intelligence[[c for c in cols_hc if c in hc_intelligence.columns]]
hc_intelligence.to_csv(f"{INTELLIGENCE_DIR}/hc_segment_intelligence.csv", index=False)

print(f"✅ Saved: hc_segment_intelligence.csv ({len(hc_intelligence)} clusters)")
print(f"\n{hc_intelligence[['cluster_name', 'customer_count', 'purchase_rate_%', 'risk_level']].to_string()}")

print("\n" + "="*70)
print("✅ SEGMENT INTELLIGENCE ENGINE COMPLETED")
print("="*70)
print(f"\n📊 RFM: {len(rfm_intelligence)} segments × {rfm_intelligence['customer_count'].sum():,} customers")
print(f"📊 KMeans: {len(kmeans_intelligence)} clusters × {kmeans_intelligence['customer_count'].sum():,} customers")
print(f"📊 HC: {len(hc_intelligence)} clusters × {hc_intelligence['customer_count'].sum():,} customers (validation sample)")
print("="*70)