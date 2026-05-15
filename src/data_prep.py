#############################################
# DATA CLEANING BEFORE RFM
#############################################

import os
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

#############################################
# PATHS
#############################################

DATA_PATH = "data/Raw_data/customerData_500k_v2.csv"
OUTPUT_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

#############################################
# LOAD DATA
#############################################

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATA CLEANING BAŞLADI")
print("=" * 60)

print(f"İlk veri boyutu: {df.shape}")

#############################################
# 1. DUPLICATE CONTROL
#############################################

print("\n##################### DUPLICATE CONTROL #####################")

duplicate_count = df.duplicated().sum()

print(f"Duplicate satır sayısı: {duplicate_count}")

if duplicate_count > 0:
    df.drop_duplicates(inplace=True)
    print(f"Duplicate silindi. Yeni boyut: {df.shape}")
else:
    print("Duplicate satır bulunmadı.")

#############################################
# 2. NEGATIVE VALUE ANALYSIS
#############################################

print("\n##################### NEGATIVE VALUE ANALYSIS #####################")

numeric_cols = df.select_dtypes(include=np.number).columns

negative_summary_before = pd.DataFrame({
    "Column": numeric_cols,
    "Negative_Count": [(df[col] < 0).sum() for col in numeric_cols]
})

negative_summary_before["Negative_Ratio"] = (
    negative_summary_before["Negative_Count"] / len(df) * 100
)

print(negative_summary_before)

negative_summary_before.to_csv(
    f"{OUTPUT_DIR}/negative_summary_before_cleaning.csv",
    index=False
)

#############################################
# 3. REMOVE IMPOSSIBLE NEGATIVE VALUES
#############################################

print("\n##################### REMOVE IMPOSSIBLE NEGATIVES #####################")

initial_rows = len(df)

# NumberOfPurchases negatif olamaz
df = df[df["NumberOfPurchases"] >= 0]

# TotalSpent negatif olamaz
df = df[df["TotalSpent"] >= 0]

# Recency negatif olamaz
df = df[df["LastPurchaseDaysAgo"] >= 0]

removed_rows = initial_rows - len(df)

print(f"Silinen satır sayısı: {removed_rows}")
print(f"Kalan veri boyutu: {df.shape}")

#############################################
# 4. CLIP NOISY NEGATIVE VALUES
#############################################

print("\n##################### CLIP NOISY NEGATIVES #####################")

# Küçük negatif tenure değerlerini 0 yap
df["CustomerTenureYears"] = df["CustomerTenureYears"].clip(lower=0)

# Küçük negatif website sürelerini 0 yap
df["TimeSpentOnWebsite"] = df["TimeSpentOnWebsite"].clip(lower=0)

print("CustomerTenureYears negatif değerleri clip edildi.")
print("TimeSpentOnWebsite negatif değerleri clip edildi.")

#############################################
# 5. FINAL NEGATIVE CONTROL
#############################################

print("\n##################### FINAL NEGATIVE CONTROL #####################")

numeric_cols = df.select_dtypes(include=np.number).columns

negative_summary_after = pd.DataFrame({
    "Column": numeric_cols,
    "Negative_Count": [(df[col] < 0).sum() for col in numeric_cols]
})

negative_summary_after["Negative_Ratio"] = (
    negative_summary_after["Negative_Count"] / len(df) * 100
)

print(negative_summary_after)

negative_summary_after.to_csv(
    f"{OUTPUT_DIR}/negative_summary_after_cleaning.csv",
    index=False
)

#############################################
# 6. NULL CONTROL
#############################################

print("\n##################### NULL CONTROL #####################")

null_summary = pd.DataFrame({
    "Column": df.columns,
    "Null_Count": df.isnull().sum().values,
    "Null_Ratio": (df.isnull().sum().values / len(df)) * 100
})

print(null_summary)

null_summary.to_csv(
    f"{OUTPUT_DIR}/null_summary_after_cleaning.csv",
    index=False
)

#############################################
# 7. DATA TYPE CONTROL
#############################################

print("\n##################### DATA TYPES #####################")

dtype_summary = pd.DataFrame({
    "Column": df.columns,
    "Dtype": df.dtypes.astype(str).values
})

print(dtype_summary)

dtype_summary.to_csv(
    f"{OUTPUT_DIR}/dtype_summary.csv",
    index=False
)
#############################################
# 8. FEATURE ENGINEERING STEP 1
#############################################

print("\n##################### FEATURE ENGINEERING STEP 1 #####################")

# Favori kategori minimum satın alma tahmini
PURCHASE_DIVIDER = 5

df["min_favorite_purchases"] = (
    df["NumberOfPurchases"] // PURCHASE_DIVIDER
) + 1

print("Yeni kolon oluşturuldu: min_favorite_purchases")

print(df["min_favorite_purchases"].describe())

# Feature summary
favorite_purchase_summary = pd.DataFrame({
    "Metric": [
        "Min_Value",
        "Max_Value",
        "Mean_Value",
        "Median_Value"
    ],
    "Value": [
        df["min_favorite_purchases"].min(),
        df["min_favorite_purchases"].max(),
        round(df["min_favorite_purchases"].mean(), 2),
        round(df["min_favorite_purchases"].median(), 2)
    ]
})

print(favorite_purchase_summary)

favorite_purchase_summary.to_csv(
    f"{OUTPUT_DIR}/favorite_purchase_summary.csv",
    index=False
)
#############################################
# 9. FINAL DESCRIPTIVE STATS
#############################################

print("\n##################### FINAL DESCRIPTIVE STATS #####################")

final_stats = df.describe(
    percentiles=[0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.99]
).T

print(final_stats)

final_stats.to_csv(
    f"{OUTPUT_DIR}/final_descriptive_stats.csv"
)

#############################################
# 10. RFM READINESS CHECK
#############################################

print("\n##################### RFM READINESS CHECK #####################")

rfm_check = pd.DataFrame({
    "Metric": [
        "Recency_Negative_Count",
        "Frequency_Negative_Count",
        "Monetary_Negative_Count",
        "Recency_Null_Count",
        "Frequency_Null_Count",
        "Monetary_Null_Count"
    ],
    "Value": [
        (df["LastPurchaseDaysAgo"] < 0).sum(),
        (df["NumberOfPurchases"] < 0).sum(),
        (df["TotalSpent"] < 0).sum(),
        df["LastPurchaseDaysAgo"].isnull().sum(),
        df["NumberOfPurchases"].isnull().sum(),
        df["TotalSpent"].isnull().sum()
    ]
})

print(rfm_check)

rfm_check.to_csv(
    f"{OUTPUT_DIR}/rfm_readiness_check.csv",
    index=False
)

#############################################
# 11. CLEANING SUMMARY
#############################################

print("\n##################### CLEANING SUMMARY #####################")

cleaning_summary = pd.DataFrame({
    "Metric": [
        "Initial_Row_Count",
        "Final_Row_Count",
        "Removed_Row_Count",
        "Removed_Row_Ratio",
        "Duplicate_Count",
        "Negative_NumberOfPurchases_Removed",
        "Negative_TotalSpent_Removed",
        "Negative_LastPurchaseDaysAgo_Removed"
    ],
    "Value": [
        initial_rows,
        len(df),
        removed_rows,
        round((removed_rows / initial_rows) * 100, 4),
        duplicate_count,
        negative_summary_before.loc[
            negative_summary_before["Column"] == "NumberOfPurchases",
            "Negative_Count"
        ].values[0],
        negative_summary_before.loc[
            negative_summary_before["Column"] == "TotalSpent",
            "Negative_Count"
        ].values[0],
        negative_summary_before.loc[
            negative_summary_before["Column"] == "LastPurchaseDaysAgo",
            "Negative_Count"
        ].values[0]
    ]
})

print(cleaning_summary)

cleaning_summary.to_csv(
    f"{OUTPUT_DIR}/cleaning_summary.csv",
    index=False
)

#############################################
# 12. CUSTOMER ID GENERATION
#############################################

print("\n##################### CUSTOMER ID GENERATION #####################")

# Unique Customer ID oluştur
df = df.reset_index(drop=True)

df.insert(
    0,                     # ilk kolon olarak ekle
    "CustomerID",
    ["CUST_" + str(i).zfill(6) for i in range(1, len(df) + 1)]
)

print("CustomerID kolonu oluşturuldu.")

print(df[["CustomerID"]].head())

print(f"Toplam unique CustomerID: {df['CustomerID'].nunique()}")

customer_id_summary = pd.DataFrame({
    "Metric": [
        "Unique_Customer_Count",
        "Duplicate_CustomerID_Count"
    ],
    "Value": [
        df["CustomerID"].nunique(),
        df["CustomerID"].duplicated().sum()
    ]
})

customer_id_summary.to_csv(
    f"{OUTPUT_DIR}/customer_id_summary.csv",
    index=False
)

#############################################
# 13. SAVE CLEAN DATASET
#############################################

clean_data_path = f"{OUTPUT_DIR}/clean_customer_data.csv"

df.to_csv(clean_data_path, index=False)

#############################################
# FINAL LOG
#############################################

print("\n" + "=" * 60)
print("DATA CLEANING TAMAMLANDI")
print("=" * 60)

print(f"Temiz veri kaydedildi:\n{clean_data_path}")

print("\nRFM için hazır kolonlar:")
print("- Recency  : LastPurchaseDaysAgo")
print("- Frequency: NumberOfPurchases")
print("- Monetary : TotalSpent")

print("\nFinal veri boyutu:", df.shape)

print("=" * 60)