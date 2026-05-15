import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

#############################################
# 1. PATHS
#############################################

DATA_PATH = "data/RFM_Output/customer_rfm_final.csv"
KMEANS_PATH = "data/KMeans_Output/customer_kmeans_rfm_final.csv"
OUTPUT_DIR = "data/HC_Output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

#############################################
# 2. LOAD DATA
#############################################

df = pd.read_csv(DATA_PATH)
df_kmeans = pd.read_csv(KMEANS_PATH)

print("=" * 60)
print("HİERARCHİCAL CLUSTERING BAŞLADI")
print("=" * 60)
print(f"Toplam veri: {df.shape}")

#############################################
# 3. FEATURE SELECTION
#############################################

features = ["recency", "frequency", "monetary"]
X = df[features].copy()

#############################################
# 4. SCALING
#############################################

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#############################################
# 5. TEMSİLİ ÖRNEKLEM - 50K
#############################################

print("\n50K temsili örneklem alınıyor...")

np.random.seed(42)
sample_idx = np.random.choice(len(X_scaled), size=50000, replace=False)
X_sample = X_scaled[sample_idx]
df_sample = df.iloc[sample_idx].copy().reset_index(drop=True)

print(f"Örneklem boyutu: {X_sample.shape}")

#############################################
# 6. DENDROGRAM
#############################################

print("\nDendrogram oluşturuluyor...")

linked = linkage(X_sample[:5000], method="ward")

plt.figure(figsize=(14, 6))
dendrogram(
    linked,
    truncate_mode="lastp",
    p=20,
    show_leaf_counts=True,
    leaf_rotation=90,
    leaf_font_size=10
)
plt.title("Hierarchical Clustering Dendrogram\n(Ward Linkage, 5K Örneklem)")
plt.xlabel("Cluster")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/dendrogram.png", dpi=150, bbox_inches="tight")
plt.close()
print("Dendrogram kaydedildi.")

#############################################
# 7. SİLHOUETTE ANALİZİ - ÖRNEKLEM
#############################################

print("\n--- Silhouette Analizi (50K örneklem) ---")

silhouette_scores = []
K_range = range(2, 9)

for k in K_range:
    hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = hc.fit_predict(X_sample)
    sil = silhouette_score(X_sample, labels, sample_size=10000, random_state=42)
    silhouette_scores.append(sil)
    print(f"K={k} | Silhouette: {sil:.4f}")

optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"\nOptimal K: {optimal_k}")
print(f"Max Silhouette: {max(silhouette_scores):.4f}")

# Silhouette grafiği
plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
plt.xlabel('K (Cluster Sayısı)')
plt.ylabel('Silhouette Score')
plt.title('HC Silhouette Scores (50K Örneklem)')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/hc_silhouette.png", dpi=150, bbox_inches="tight")
plt.close()

# CSV kaydet
sil_df = pd.DataFrame({
    'K': list(K_range),
    'Silhouette_Score': silhouette_scores
})
sil_df.to_csv(f"{OUTPUT_DIR}/hc_silhouette_results.csv", index=False)

#############################################
# 8. FİNAL HC - ÖRNEKLEM ÜZERİNDE
#############################################

print(f"\n--- Final HC (K={optimal_k}) - 50K Örneklem ---")

hc_final = AgglomerativeClustering(n_clusters=optimal_k, linkage="ward")
df_sample["HC_Cluster"] = hc_final.fit_predict(X_sample)

print(f"Cluster dağılımı:\n{df_sample['HC_Cluster'].value_counts().sort_index()}")

#############################################
# 9. HC CLUSTER PROFİLİ
#############################################

hc_profile = df_sample.groupby("HC_Cluster").agg(
    Musteri_Sayisi=("CustomerID", "count"),
    Ort_Recency=("recency", "mean"),
    Ort_Frequency=("frequency", "mean"),
    Ort_Monetary=("monetary", "mean"),
    Satin_Alma_Orani=("PurchaseStatus", "mean"),
    Ort_Memnuniyet=("CustomerSatisfaction", "mean"),
).round(3)

hc_profile["Musteri_Orani"] = (
    hc_profile["Musteri_Sayisi"] / len(df_sample) * 100
).round(2)

print("\n--- HC Cluster Profili ---")
print(hc_profile)

#############################################
# 10. HC CLUSTER İSİMLENDİRME
#############################################

def name_hc_cluster(row):
    recency = row["Ort_Recency"]
    frequency = row["Ort_Frequency"]
    monetary = row["Ort_Monetary"]

    if recency < 30 and frequency > 14 and monetary > 1500:
        return "Yüksek Değerli Aktif"
    elif recency < 30 and monetary > 1500:
        return "Aktif Yüksek Harcamalı"
    elif recency < 30 and frequency < 8:
        return "Aktif Düşük Frekanslı"
    elif recency < 30:
        return "Aktif Orta Segment"
    elif recency > 90 and monetary > 1500:
        return "Kaybolmakta Olan Değerliler"
    elif recency > 90 and frequency > 14:
        return "Kaybolmakta Olan Sık Alıcılar"
    elif recency > 90:
        return "Pasif Müşteriler"
    else:
        return "Orta Aktiflik"

hc_profile["HC_Cluster_Adi"] = hc_profile.apply(name_hc_cluster, axis=1)

print("\n--- HC Cluster İsimleri ---")
print(hc_profile[["Musteri_Sayisi", "Musteri_Orani", "HC_Cluster_Adi"]])

hc_profile.to_csv(f"{OUTPUT_DIR}/hc_cluster_profile.csv")

#############################################
# 11. KMEANS × HC KARŞILAŞTIRMASI
#############################################

print("\n--- KMeans × HC Karşılaştırması (Örneklem) ---")

hc_name_map = hc_profile["HC_Cluster_Adi"].to_dict()
df_sample["HC_Cluster_Adi"] = df_sample["HC_Cluster"].map(hc_name_map)
df_sample["KMeans_Cluster_Adi"] = df_kmeans.iloc[sample_idx]["KMeans_Cluster_Adi"].values

kmeans_hc_cross = df_sample.groupby(
    ["KMeans_Cluster_Adi", "HC_Cluster_Adi"]
).agg(
    Musteri_Sayisi=("CustomerID", "count")
).reset_index()

kmeans_hc_cross.to_csv(f"{OUTPUT_DIR}/kmeans_hc_cross.csv", index=False)

print(kmeans_hc_cross.sort_values("Musteri_Sayisi", ascending=False))

#############################################
# 12. RFM × HC KARŞILAŞTIRMASI
#############################################

rfm_hc_cross = df_sample.groupby(
    ["RFM_Segment", "HC_Cluster_Adi"]
).agg(
    Musteri_Sayisi=("CustomerID", "count")
).reset_index()

rfm_hc_cross.to_csv(f"{OUTPUT_DIR}/rfm_hc_cross.csv", index=False)

print("\n--- RFM × HC Karşılaştırması ---")
print(rfm_hc_cross.sort_values("Musteri_Sayisi", ascending=False).head(15))

#############################################
# 13. ÜÇ YÖNTEM KARŞILAŞTIRMASI
#############################################

three_method = df_sample[[
    "CustomerID",
    "recency",
    "frequency",
    "monetary",
    "RFM_Segment",
    "KMeans_Cluster_Adi",
    "HC_Cluster_Adi",
    "PurchaseStatus",
    "TotalSpent"
]].copy()

three_method.to_csv(f"{OUTPUT_DIR}/three_method_comparison.csv", index=False)

#############################################
# 14. ÖZET
#############################################

print("\n" + "=" * 60)
print("HİERARCHİCAL CLUSTERING TAMAMLANDI")
print("=" * 60)
print(f"Örneklem boyutu: 50,000")
print(f"Optimal K: {optimal_k}")
print(f"Max Silhouette: {max(silhouette_scores):.4f}")
print(f"\nNot: HC 50K temsili örneklem üzerinde çalıştırıldı.")
print(f"K-Means ile tutarlılık kmeans_hc_cross.csv dosyasında.")
print(f"\nKaydedilen dosyalar:")
for f_name in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  ✅ {OUTPUT_DIR}/{f_name}")