import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

#############################################
# 1. PATHS
#############################################

DATA_PATH = "data/RFM_Output/customer_rfm_final.csv"
OUTPUT_DIR = "data/KMeans_Output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

#############################################
# 2. LOAD DATA
#############################################

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("K-MEANS CLUSTERING BAŞLADI (RFM Only)")
print("=" * 60)
print(f"Veri boyutu: {df.shape}")

#############################################
# 3. FEATURE SELECTION - SADECE RFM
#############################################

features = ["recency", "frequency", "monetary"]

X = df[features].copy()

print(f"\nKullanılan özellikler: {features}")

#############################################
# 4. SCALING
#############################################

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Veriler ölçeklendi (StandardScaler)")

#############################################
# 5. ELBOW + SİLHOUETTE
#############################################

print("\n--- Elbow + Silhouette ---")

inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    sil = silhouette_score(X_scaled, kmeans.labels_, sample_size=10000, random_state=42)
    silhouette_scores.append(sil)
    print(f"K={k} | Inertia: {kmeans.inertia_:.0f} | Silhouette: {sil:.4f}")

# Grafik kaydet
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('K (Cluster Sayısı)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method - RFM')
axes[0].grid(True)

axes[1].plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('K (Cluster Sayısı)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Scores - RFM')
axes[1].grid(True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/elbow_silhouette_rfm.png", dpi=150, bbox_inches='tight')
plt.close()

# CSV kaydet
elbow_df = pd.DataFrame({
    'K': list(K_range),
    'Inertia': inertias,
    'Silhouette_Score': silhouette_scores
})
elbow_df.to_csv(f"{OUTPUT_DIR}/elbow_results_rfm.csv", index=False)

#############################################
# 6. OPTİMAL K
#############################################

optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"\nOptimal K: {optimal_k}")
print(f"Max Silhouette Score: {max(silhouette_scores):.4f}")

#############################################
# 7. FİNAL K-MEANS
#############################################

print(f"\n--- Final K-Means (K={optimal_k}) ---")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_final.fit(X_scaled)

df["KMeans_Cluster"] = kmeans_final.labels_

print(f"Cluster dağılımı:\n{df['KMeans_Cluster'].value_counts().sort_index()}")

#############################################
# 8. CLUSTER PROFİLİ
#############################################

cluster_profile = df.groupby("KMeans_Cluster").agg(
    Musteri_Sayisi=("CustomerID", "count"),
    Ort_Recency=("recency", "mean"),
    Ort_Frequency=("frequency", "mean"),
    Ort_Monetary=("monetary", "mean"),
    Satin_Alma_Orani=("PurchaseStatus", "mean"),
    Ort_Memnuniyet=("CustomerSatisfaction", "mean"),
).round(3)

cluster_profile["Musteri_Orani"] = (
    cluster_profile["Musteri_Sayisi"] / len(df) * 100
).round(2)

print("\n--- Cluster Profili ---")
print(cluster_profile)

#############################################
# 9. CLUSTER İSİMLENDİRME
#############################################

def name_cluster(row):
    recency = row["Ort_Recency"]
    frequency = row["Ort_Frequency"]
    monetary = row["Ort_Monetary"]
    purchase_rate = row["Satin_Alma_Orani"]

    if recency < 30 and frequency > 14 and monetary > 1500:
        return "Yüksek Değerli Aktif Müşteriler"
    elif recency < 30 and purchase_rate > 0.5:
        return "Aktif Alıcılar"
    elif recency > 90 and monetary > 1500:
        return "Kaybolmakta Olan Değerliler"
    elif recency > 90 and purchase_rate < 0.1:
        return "Pasif Müşteriler"
    elif frequency > 14:
        return "Sık Alışveriş Yapanlar"
    elif monetary > 1500:
        return "Yüksek Harcamalılar"
    else:
        return "Orta Segment Müşteriler"

cluster_profile["Cluster_Adi"] = cluster_profile.apply(name_cluster, axis=1)

print("\n--- Cluster İsimleri ---")
print(cluster_profile[["Musteri_Sayisi", "Musteri_Orani", "Cluster_Adi"]])

cluster_profile.to_csv(f"{OUTPUT_DIR}/kmeans_cluster_profile_rfm.csv")

#############################################
# 10. RFM SEGMENT × KMEANS KARŞILAŞTIRMASI
#############################################

print("\n--- RFM Segment × KMeans Cluster Karşılaştırması ---")

cluster_name_map = cluster_profile["Cluster_Adi"].to_dict()
df["KMeans_Cluster_Adi"] = df["KMeans_Cluster"].map(cluster_name_map)

rfm_kmeans_cross = df.groupby(
    ["RFM_Segment", "KMeans_Cluster_Adi"]
).agg(
    Musteri_Sayisi=("CustomerID", "count")
).reset_index()

rfm_kmeans_cross.to_csv(f"{OUTPUT_DIR}/rfm_kmeans_cross.csv", index=False)

print(rfm_kmeans_cross.sort_values("Musteri_Sayisi", ascending=False).head(20))

#############################################
# 11. KATEGORİK ANALİZLER
#############################################

cluster_category = df.groupby(
    ["KMeans_Cluster_Adi", "ProductCategory"]
).agg(
    Musteri_Sayisi=("CustomerID", "count"),
    Ort_Harcama=("TotalSpent", "mean"),
    Satin_Alma_Orani=("PurchaseStatus", "mean")
).reset_index()
cluster_category.to_csv(f"{OUTPUT_DIR}/cluster_category_rfm.csv", index=False)

cluster_region = df.groupby(
    ["KMeans_Cluster_Adi", "Region"]
).agg(
    Musteri_Sayisi=("CustomerID", "count")
).reset_index()
cluster_region.to_csv(f"{OUTPUT_DIR}/cluster_region_rfm.csv", index=False)

#############################################
# 12. FİNAL DATASET
#############################################

df.to_csv(f"{OUTPUT_DIR}/customer_kmeans_rfm_final.csv", index=False)

#############################################
# 13. ÖZET
#############################################

print("\n" + "=" * 60)
print("K-MEANS (RFM Only) TAMAMLANDI")
print("=" * 60)
print(f"Optimal K: {optimal_k}")
print(f"Max Silhouette: {max(silhouette_scores):.4f}")
print(f"Toplam müşteri: {len(df)}")
print(f"\nKaydedilen dosyalar:")
for f_name in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  ✅ {OUTPUT_DIR}/{f_name}")