#############################################
# RFM ANALYSIS - CUSTOMER SEGMENTATION
#############################################

import os
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

#############################################
# 1. PATHS
#############################################

DATA_PATH = "data/clean_customer_data.csv"
OUTPUT_DIR = "data/RFM_Output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

#############################################
# 2. LOAD CLEAN DATA
#############################################

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("RFM ANALİZİ BAŞLADI")
print("=" * 60)

print(f"Veri boyutu: {df.shape}")

#############################################
# 3. AGE GROUP
#############################################

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 25, 35, 50, 100],
    labels=["18-25", "26-35", "36-50", "50+"]
)

#############################################
# 4. RFM METRICS
#############################################

rfm = pd.DataFrame()

rfm["CustomerID"] = df["CustomerID"]
rfm["recency"] = df["LastPurchaseDaysAgo"]
rfm["frequency"] = df["NumberOfPurchases"]
rfm["monetary"] = df["TotalSpent"]

#############################################
# 5. RFM SCORES
#############################################

rfm["recency_score"] = pd.qcut(
    rfm["recency"],
    5,
    labels=[5, 4, 3, 2, 1],
    duplicates="drop"
)

rfm["frequency_score"] = pd.qcut(
    rfm["frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5],
    duplicates="drop"
)

rfm["monetary_score"] = pd.qcut(
    rfm["monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5],
    duplicates="drop"
)

rfm["RF_SCORE"] = (
    rfm["recency_score"].astype(str) +
    rfm["frequency_score"].astype(str)
)

rfm["RFM_SCORE"] = (
    rfm["recency_score"].astype(str) +
    rfm["frequency_score"].astype(str) +
    rfm["monetary_score"].astype(str)
)

#############################################
# 6. RFM SEGMENT MAPPING
#############################################

seg_map = {
    r"[1-2][1-2]": "hibernating",
    r"[1-2][3-4]": "at_risk",
    r"[1-2]5": "cant_loose",
    r"3[1-2]": "about_to_sleep",
    r"33": "need_attention",
    r"[3-4][4-5]": "loyal_customers",
    r"41": "promising",
    r"51": "new_customers",
    r"[4-5][2-3]": "potential_loyalists",
    r"5[4-5]": "champions"
}

rfm["RFM_Segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

#############################################
# 7. MERGE RFM WITH CUSTOMER DATA
#############################################

df_rfm = df.merge(
    rfm[
        [
            "CustomerID",
            "recency",
            "frequency",
            "monetary",
            "recency_score",
            "frequency_score",
            "monetary_score",
            "RF_SCORE",
            "RFM_SCORE",
            "RFM_Segment"
        ]
    ],
    on="CustomerID",
    how="left"
)

#############################################
# 8. RFM SEGMENT SUMMARY
#############################################

rfm_segment_summary = df_rfm.groupby("RFM_Segment").agg(
    Musteri_Sayisi=("CustomerID", "count"),
    Ortalama_Recency=("recency", "mean"),
    Ortalama_Frequency=("frequency", "mean"),
    Ortalama_Monetary=("monetary", "mean"),
    Satin_Alma_Orani=("PurchaseStatus", "mean"),
    Ortalama_Discount=("DiscountsAvailed", "mean"),
    Ortalama_Website_Suresi=("TimeSpentOnWebsite", "mean"),
    Ortalama_Memnuniyet=("CustomerSatisfaction", "mean"),
    Ortalama_Min_Favori_Satis=("min_favorite_purchases", "mean")
).sort_values("Musteri_Sayisi", ascending=False)

rfm_segment_summary.to_csv(
    f"{OUTPUT_DIR}/rfm_segment_summary.csv"
)

print("\n##################### RFM SEGMENT SUMMARY #####################")
print(rfm_segment_summary)

#############################################
# 9. SEGMENT DISTRIBUTION
#############################################

rfm_segment_distribution = df_rfm["RFM_Segment"].value_counts().reset_index()
rfm_segment_distribution.columns = ["RFM_Segment", "Customer_Count"]
rfm_segment_distribution["Ratio"] = (
    rfm_segment_distribution["Customer_Count"] /
    len(df_rfm) * 100
).round(2)

rfm_segment_distribution.to_csv(
    f"{OUTPUT_DIR}/rfm_segment_distribution.csv",
    index=False
)

#############################################
# 10. RFM × PRODUCT CATEGORY
#############################################

rfm_category_summary = df_rfm.groupby(
    ["RFM_Segment", "ProductCategory"]
).agg(
    Musteri_Sayisi=("CustomerID", "count"),
    Ortalama_Harcama=("TotalSpent", "mean"),
    Satin_Alma_Orani=("PurchaseStatus", "mean"),
    Ortalama_Min_Favori_Satis=("min_favorite_purchases", "mean")
).reset_index()

rfm_category_summary.to_csv(
    f"{OUTPUT_DIR}/rfm_category_summary.csv",
    index=False
)

#############################################
# 11. RFM × GENDER
#############################################

rfm_gender_summary = df_rfm.groupby(
    ["RFM_Segment", "Gender"]
).agg(
    Musteri_Sayisi=("CustomerID", "count"),
    Ortalama_Harcama=("TotalSpent", "mean"),
    Satin_Alma_Orani=("PurchaseStatus", "mean")
).reset_index()

rfm_gender_summary.to_csv(
    f"{OUTPUT_DIR}/rfm_gender_summary.csv",
    index=False
)

#############################################
# 12. RFM × REGION
#############################################

rfm_region_summary = df_rfm.groupby(
    ["RFM_Segment", "Region"]
).agg(
    Musteri_Sayisi=("CustomerID", "count"),
    Ortalama_Harcama=("TotalSpent", "mean"),
    Satin_Alma_Orani=("PurchaseStatus", "mean")
).reset_index()

rfm_region_summary.to_csv(
    f"{OUTPUT_DIR}/rfm_region_summary.csv",
    index=False
)

#############################################
# 13. RFM × REFERRAL SOURCE
#############################################

rfm_source_summary = df_rfm.groupby(
    ["RFM_Segment", "ReferralSource"]
).agg(
    Musteri_Sayisi=("CustomerID", "count"),
    Ortalama_Harcama=("TotalSpent", "mean"),
    Satin_Alma_Orani=("PurchaseStatus", "mean")
).reset_index()

rfm_source_summary.to_csv(
    f"{OUTPUT_DIR}/rfm_source_summary.csv",
    index=False
)

#############################################
# 14. RFM × AGE GROUP
#############################################

rfm_age_summary = df_rfm.groupby(
    ["RFM_Segment", "AgeGroup"],
    observed=True
).agg(
    Musteri_Sayisi=("CustomerID", "count"),
    Ortalama_Harcama=("TotalSpent", "mean"),
    Satin_Alma_Orani=("PurchaseStatus", "mean")
).reset_index()

rfm_age_summary.to_csv(
    f"{OUTPUT_DIR}/rfm_age_summary.csv",
    index=False
)

#############################################
# 15. RFM × DISCOUNT ANALYSIS
#############################################

rfm_discount_summary = df_rfm.groupby("RFM_Segment").agg(
    Musteri_Sayisi=("CustomerID", "count"),
    Ortalama_Discount=("DiscountsAvailed", "mean"),
    Median_Discount=("DiscountsAvailed", "median"),
    Max_Discount=("DiscountsAvailed", "max"),
    Satin_Alma_Orani=("PurchaseStatus", "mean"),
    Ortalama_Harcama=("TotalSpent", "mean")
).sort_values("Ortalama_Discount", ascending=False)

rfm_discount_summary.to_csv(
    f"{OUTPUT_DIR}/rfm_discount_summary.csv"
)

#############################################
# 16. DISCOUNT AFFINITY FLAG
#############################################

discount_threshold = df_rfm["DiscountsAvailed"].median()

df_rfm["DiscountAffinity"] = np.where(
    df_rfm["DiscountsAvailed"] >= discount_threshold,
    "Discount_Oriented",
    "Non_Discount_Oriented"
)

discount_affinity_summary = df_rfm.groupby(
    ["RFM_Segment", "DiscountAffinity"]
).agg(
    Musteri_Sayisi=("CustomerID", "count"),
    Ortalama_Harcama=("TotalSpent", "mean"),
    Satin_Alma_Orani=("PurchaseStatus", "mean")
).reset_index()

discount_affinity_summary.to_csv(
    f"{OUTPUT_DIR}/discount_affinity_summary.csv",
    index=False
)

#############################################
# 17. CAMPAIGN STRATEGY RULES
#############################################

def campaign_strategy(row):
    segment = row["RFM_Segment"]
    discount = row["DiscountAffinity"]
    category = row["ProductCategory"]

    if segment == "champions":
        return f"{category} kategorisinde VIP özel erişim / yeni ürün önceliği"

    elif segment == "loyal_customers":
        return f"{category} kategorisinde sadakat ödülü ve çapraz satış kampanyası"

    elif segment == "potential_loyalists":
        return f"{category} kategorisinde loyalty üyeliğine teşvik kampanyası"

    elif segment == "new_customers":
        return f"{category} kategorisinde ikinci alışverişe özel teklif"

    elif segment == "promising":
        return f"{category} kategorisinde düşük indirimli deneme kampanyası"

    elif segment == "need_attention":
        return f"{category} kategorisinde kişiselleştirilmiş hatırlatma kampanyası"

    elif segment == "about_to_sleep":
        return f"{category} kategorisinde yeniden aktivasyon kampanyası"

    elif segment == "at_risk":
        if discount == "Discount_Oriented":
            return f"{category} kategorisinde geri kazanım indirimi"
        else:
            return f"{category} kategorisinde özel avantaj / ücretsiz kargo kampanyası"

    elif segment == "cant_loose":
        return f"{category} kategorisinde yüksek değerli müşteri geri kazanım kampanyası"

    elif segment == "hibernating":
        if discount == "Discount_Oriented":
            return f"{category} kategorisinde agresif reaktivasyon indirimi"
        else:
            return f"{category} kategorisinde düşük maliyetli bilgilendirme kampanyası"

    else:
        return f"{category} kategorisinde genel kampanya"


df_rfm["RecommendedCampaign"] = df_rfm.apply(campaign_strategy, axis=1)

#############################################
# 18. CAMPAIGN PRIORITY
#############################################

priority_map = {
    "champions": "High_Value_Retention",
    "loyal_customers": "Retention",
    "potential_loyalists": "Growth",
    "new_customers": "Activation",
    "promising": "Activation",
    "need_attention": "Reactivation",
    "about_to_sleep": "Reactivation",
    "at_risk": "Win_Back",
    "cant_loose": "Critical_Win_Back",
    "hibernating": "Low_Cost_Reactivation"
}

df_rfm["CampaignPriority"] = df_rfm["RFM_Segment"].map(priority_map)

#############################################
# 19. CAMPAIGN CUSTOMER LIST FOR N8N
#############################################

campaign_customer_list = df_rfm[
    [
        "CustomerID",
        "RFM_Segment",
        "CampaignPriority",
        "RecommendedCampaign",
        "Age",
        "AgeGroup",
        "Gender",
        "Region",
        "ReferralSource",
        "ProductCategory",
        "DiscountsAvailed",
        "DiscountAffinity",
        "min_favorite_purchases",
        "PurchaseStatus",
        "TotalSpent",
        "LastPurchaseDaysAgo"
    ]
].copy()

campaign_customer_list.to_csv(
    f"{OUTPUT_DIR}/campaign_customer_list.csv",
    index=False
)

#############################################
# 20. HIGH PRIORITY CAMPAIGN LIST
#############################################

high_priority_segments = [
    "champions",
    "loyal_customers",
    "potential_loyalists",
    "at_risk",
    "cant_loose"
]

high_priority_campaign_list = campaign_customer_list[
    campaign_customer_list["RFM_Segment"].isin(high_priority_segments)
]

high_priority_campaign_list.to_csv(
    f"{OUTPUT_DIR}/high_priority_campaign_customer_list.csv",
    index=False
)

#############################################
# 21. DISCOUNT CAMPAIGN TARGET LIST
#############################################

discount_campaign_list = campaign_customer_list[
    (
        campaign_customer_list["RFM_Segment"].isin(
            ["at_risk", "cant_loose", "hibernating", "about_to_sleep"]
        )
    )
    &
    (
        campaign_customer_list["DiscountAffinity"] == "Discount_Oriented"
    )
]

discount_campaign_list.to_csv(
    f"{OUTPUT_DIR}/discount_campaign_customer_list.csv",
    index=False
)

#############################################
# 22. FINAL RFM DATASET
#############################################

df_rfm.to_csv(
    f"{OUTPUT_DIR}/customer_rfm_final.csv",
    index=False
)

#############################################
# 23. RFM SUMMARY NOTES
#############################################

rfm_notes = pd.DataFrame({
    "Topic": [
        "RFM Yaklaşımı",
        "Recency",
        "Frequency",
        "Monetary",
        "Segmentasyon",
        "ProductCategory",
        "DiscountAffinity",
        "N8N Export"
    ],
    "Explanation": [
        "Klasik transactional RFM yerine behavioral RFM uygulanmıştır.",
        "LastPurchaseDaysAgo kullanılmıştır. Düşük değer daha iyi kabul edilmiştir.",
        "NumberOfPurchases kullanılmıştır. Yüksek değer daha iyi kabul edilmiştir.",
        "TotalSpent kullanılmıştır. Yüksek değer daha iyi kabul edilmiştir.",
        "Derslerde kullanılan RF_SCORE regex segment mantığı uygulanmıştır.",
        "ProductCategory favori kategori olarak yorumlanmıştır.",
        "DiscountsAvailed medyan değere göre indirim odaklılık etiketi oluşturulmuştur.",
        "Kampanya otomasyonu için customer list CSV dosyaları oluşturulmuştur."
    ]
})

rfm_notes.to_csv(
    f"{OUTPUT_DIR}/rfm_notes.csv",
    index=False
)

#############################################
# FINAL LOG
#############################################

print("\n" + "=" * 60)
print("RFM ANALİZİ TAMAMLANDI")
print("=" * 60)

print(f"Tüm RFM çıktıları '{OUTPUT_DIR}' klasörüne kaydedildi.")
print(f"Final RFM dataset: {OUTPUT_DIR}/customer_rfm_final.csv")
print(f"N8N kampanya listesi: {OUTPUT_DIR}/campaign_customer_list.csv")

print("\nSegment dağılımı:")
print(df_rfm["RFM_Segment"].value_counts())

print("=" * 60)