#############################################
# ADVANCED FUNCTIONAL EDA - CUSTOMER DATA
#############################################

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

#############################################
# 0. Veri Yükleme
#############################################

DATA_PATH = "data/customerData_500k_v2.csv"
OUTPUT_DIR = "data/Eda_Output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("EDA BAŞLADI")
print(f"Veri Boyutu: {df.shape}")
print("=" * 60)


#############################################
# 1. Genel Resim
#############################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


#############################################
# 2. Değişkenlerin Yakalanması
#############################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [
        col for col in dataframe.columns
        if str(dataframe[col].dtypes) in ["category", "object", "bool"]
    ]

    num_but_cat = [
        col for col in dataframe.columns
        if dataframe[col].nunique() < cat_th
        and dataframe[col].dtypes in ["int64", "float64", "int32", "float32"]
    ]

    cat_but_car = [
        col for col in dataframe.columns
        if dataframe[col].nunique() > car_th
        and str(dataframe[col].dtypes) in ["category", "object"]
    ]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [
        col for col in dataframe.columns
        if dataframe[col].dtypes in ["int64", "float64", "int32", "float32"]
    ]

    num_cols = [col for col in num_cols if col not in cat_cols]

    print("##################### Variable Summary #####################")
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


#############################################
# 3. Kategorik Değişken Analizi
#############################################

def cat_summary(dataframe, col_name):
    summary = pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        "Ratio": 100 * dataframe[col_name].value_counts(normalize=True)
    })

    return summary


cat_summary_dict = {}

for col in cat_cols:
    cat_summary_dict[col] = cat_summary(df, col)
    cat_summary_dict[col].to_csv(f"{OUTPUT_DIR}/cat_summary_{col}.csv")

    print(f"##################### {col} #####################")
    print(cat_summary_dict[col])
    print()


#############################################
# 4. Sayısal Değişken Analizi
#############################################

def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.99]
    return dataframe[numerical_col].describe(quantiles).to_frame().T


num_summary_list = []

for col in num_cols:
    summary = num_summary(df, col)
    summary.index = [col]
    num_summary_list.append(summary)

    print(f"##################### {col} #####################")
    print(summary)
    print()

num_summary_df = pd.concat(num_summary_list)
num_summary_df.to_csv(f"{OUTPUT_DIR}/num_summary.csv")


#############################################
# 5. Hedef Değişken Analizi
#############################################

TARGET = "PurchaseStatus"

target_summary = pd.DataFrame({
    TARGET: df[TARGET].value_counts(),
    "Ratio": 100 * df[TARGET].value_counts(normalize=True)
})

target_summary.to_csv(f"{OUTPUT_DIR}/target_summary.csv")

print("##################### Target Summary #####################")
print(target_summary)
print(f"Genel satın alma oranı: %{df[TARGET].mean() * 100:.2f}")


def target_summary_with_cat(dataframe, target, categorical_col):
    summary = dataframe.groupby(categorical_col, observed=True).agg(
        TARGET_MEAN=(target, "mean"),
        COUNT=(target, "count")
    ).sort_values("TARGET_MEAN", ascending=False)

    return summary


target_cat_summary_dict = {}

for col in cat_cols:
    if col != TARGET:
        summary = target_summary_with_cat(df, TARGET, col)
        target_cat_summary_dict[col] = summary
        summary.to_csv(f"{OUTPUT_DIR}/target_cat_summary_{col}.csv")

        print(f"##################### {col} x {TARGET} #####################")
        print(summary)
        print()


def target_summary_with_num(dataframe, target, numerical_col):
    summary = dataframe.groupby(target).agg({
        numerical_col: ["mean", "median", "std", "count"]
    })

    return summary


target_num_summary_list = []

for col in num_cols:
    if col != TARGET:
        summary = target_summary_with_num(df, TARGET, col)
        summary.columns = [
            f"{col}_mean",
            f"{col}_median",
            f"{col}_std",
            f"{col}_count"
        ]
        target_num_summary_list.append(summary)

        print(f"##################### {col} x {TARGET} #####################")
        print(summary)
        print()

target_num_summary_df = pd.concat(target_num_summary_list, axis=1)
target_num_summary_df.to_csv(f"{OUTPUT_DIR}/target_num_summary.csv")


#############################################
# 6. Korelasyon Analizi
#############################################

def high_correlated_cols(dataframe, corr_th=0.90):
    corr = dataframe.corr(numeric_only=True)
    cor_matrix = corr.abs()

    upper_triangle_matrix = cor_matrix.where(
        np.triu(np.ones(cor_matrix.shape), k=1).astype(bool)
    )

    drop_list = [
        col for col in upper_triangle_matrix.columns
        if any(upper_triangle_matrix[col] > corr_th)
    ]

    return drop_list, corr


numeric_df = df.select_dtypes(include=["int64", "float64", "int32", "float32"])
numeric_features = numeric_df.drop(columns=[TARGET], errors="ignore")

high_corr_cols, corr_matrix = high_correlated_cols(numeric_features, corr_th=0.90)

corr_matrix.to_csv(f"{OUTPUT_DIR}/correlation_matrix.csv")

high_corr_df = pd.DataFrame({
    "High_Correlated_Columns": high_corr_cols
})

high_corr_df.to_csv(f"{OUTPUT_DIR}/high_correlated_columns.csv", index=False)

print("##################### High Correlated Columns #####################")
print(high_corr_cols)


#############################################
# 7. Negatif Değer Analizi
#############################################

negative_summary_list = []

for col in numeric_df.columns:
    neg_count = df[df[col] < 0].shape[0]

    if neg_count > 0:
        negative_summary_list.append({
            "Column": col,
            "Negative_Count": neg_count,
            "Negative_Ratio": neg_count / len(df) * 100,
            "Min_Value": df[df[col] < 0][col].min(),
            "Mean_Negative_Value": df[df[col] < 0][col].mean()
        })

negative_summary = pd.DataFrame(negative_summary_list)
negative_summary.to_csv(f"{OUTPUT_DIR}/negative_value_summary.csv", index=False)

print("##################### Negative Value Summary #####################")
print(negative_summary)

negative_cols = negative_summary["Column"].tolist() if not negative_summary.empty else []

if len(negative_cols) > 0:
    negative_index = pd.Index([])

    for col in negative_cols:
        negative_index = negative_index.union(df[df[col] < 0].index)

    negative_rows = df.loc[negative_index]
    negative_rows.to_csv(f"{OUTPUT_DIR}/negative_rows.csv", index=False)

    print(f"Toplam unique negatif satır: {len(negative_rows)}")
else:
    print("Negatif değer bulunmadı.")


#############################################
# 8. Yeni Gruplama Değişkenleri
#############################################

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 25, 35, 50, 100],
    labels=["18-25", "26-35", "36-50", "50+"]
)

df["IncomeGroup"] = pd.cut(
    df["AnnualIncome"],
    bins=[0, 50000, 100000, 150000, 300000],
    labels=["Dusuk", "Orta", "Yuksek", "Cok_Yuksek"]
)

df["TenureGroup"] = pd.cut(
    df["CustomerTenureYears"],
    bins=[-1, 0, 1, 3, 6, 20],
    labels=["0_veya_Negatif", "0_1_Yil", "1_3_Yil", "3_6_Yil", "6_Plus_Yil"]
)

df["RecencyGroup"] = pd.cut(
    df["LastPurchaseDaysAgo"],
    bins=[-20, 0, 30, 60, 90, 200],
    labels=["Negatif", "0_30_Gun", "30_60_Gun", "60_90_Gun", "90_Plus_Gun"]
)

df["TimeGroup"] = pd.cut(
    df["TimeSpentOnWebsite"],
    bins=[-5, 0, 20, 40, 60, 100],
    labels=["Negatif", "0_20", "20_40", "40_60", "60_Plus"]
)

df["min_favorite_purchases"] = (df["NumberOfPurchases"] // 5) + 1


#############################################
# 9. İş Problemine Özel Grup Analizleri
#############################################

def group_target_summary(dataframe, group_col, target=TARGET):
    summary = dataframe.groupby(group_col, observed=True).agg(
        Satin_Alma_Orani=(target, "mean"),
        Musteri_Sayisi=(target, "count")
    ).sort_values("Satin_Alma_Orani", ascending=False)

    return summary


recency_summary = group_target_summary(df, "RecencyGroup")
satisfaction_summary = group_target_summary(df, "CustomerSatisfaction")
loyalty_summary = group_target_summary(df, "LoyaltyProgram")
time_summary = group_target_summary(df, "TimeGroup")
age_summary = group_target_summary(df, "AgeGroup")
tenure_summary = group_target_summary(df, "TenureGroup")
income_summary = group_target_summary(df, "IncomeGroup")

recency_summary.to_csv(f"{OUTPUT_DIR}/recency_summary.csv")
satisfaction_summary.to_csv(f"{OUTPUT_DIR}/satisfaction_summary.csv")
loyalty_summary.to_csv(f"{OUTPUT_DIR}/loyalty_summary.csv")
time_summary.to_csv(f"{OUTPUT_DIR}/time_summary.csv")
age_summary.to_csv(f"{OUTPUT_DIR}/age_summary.csv")
tenure_summary.to_csv(f"{OUTPUT_DIR}/tenure_summary.csv")
income_summary.to_csv(f"{OUTPUT_DIR}/income_summary.csv")


#############################################
# 10. Segment Analizi
#############################################

segment_summary = df.groupby("CustomerSegment", observed=True).agg(
    Musteri_Sayisi=(TARGET, "count"),
    Satin_Alma_Orani=(TARGET, "mean"),
    Ortalama_Recency=("LastPurchaseDaysAgo", "mean"),
    Ortalama_Memnuniyet=("CustomerSatisfaction", "mean"),
    Ortalama_Harcama=("TotalSpent", "mean")
).sort_values("Satin_Alma_Orani", ascending=False)

segment_summary.to_csv(f"{OUTPUT_DIR}/segment_summary.csv")

lowest_segment = segment_summary["Satin_Alma_Orani"].idxmin()
lowest_rate = segment_summary.loc[lowest_segment, "Satin_Alma_Orani"]

print("##################### Segment Summary #####################")
print(segment_summary)
print(f"En düşük satın alma oranına sahip segment: {lowest_segment}")
print(f"Satın alma oranı: %{lowest_rate * 100:.2f}")


#############################################
# 11. VIP Analizi
#############################################

if "VIP" in df["CustomerSegment"].unique():
    vip_df = df[df["CustomerSegment"] == "VIP"]

    vip_recency_summary = vip_df.groupby("RecencyGroup", observed=True).agg(
        Satin_Alma_Orani=(TARGET, "mean"),
        Musteri_Sayisi=(TARGET, "count")
    )

    vip_recency_summary.to_csv(f"{OUTPUT_DIR}/vip_recency_summary.csv")

    vip_90_plus = vip_df[vip_df["LastPurchaseDaysAgo"] > 90].shape[0]
    vip_90_ratio = vip_90_plus / len(vip_df) * 100

    vip_summary = pd.DataFrame({
        "Metric": ["VIP_Count", "VIP_90_Plus_Count", "VIP_90_Plus_Ratio"],
        "Value": [len(vip_df), vip_90_plus, vip_90_ratio]
    })

    vip_summary.to_csv(f"{OUTPUT_DIR}/vip_summary.csv", index=False)


#############################################
# 12. Davranışsal Analizler
#############################################

session_summary = df.groupby("SessionCount").agg(
    Satin_Alma_Orani=(TARGET, "mean"),
    Musteri_Sayisi=(TARGET, "count")
).sort_index()

session_summary.to_csv(f"{OUTPUT_DIR}/session_summary.csv")

loyalty_spent_summary = df.groupby(["LoyaltyProgram", TARGET]).agg(
    Ortalama_Harcama=("TotalSpent", "mean"),
    Musteri_Sayisi=("TotalSpent", "count")
)

loyalty_spent_summary.to_csv(f"{OUTPUT_DIR}/loyalty_spent_summary.csv")


#############################################
# 13. Demografik Analizler
#############################################

age_category_summary = df.groupby(
    ["AgeGroup", "ProductCategory"],
    observed=True
)[TARGET].mean().unstack()

age_category_summary.to_csv(f"{OUTPUT_DIR}/age_category_summary.csv")

age_device_summary = df.groupby(
    ["AgeGroup", "PreferredDevice"],
    observed=True
).size().unstack().fillna(0).astype(int)

age_device_summary.to_csv(f"{OUTPUT_DIR}/age_device_summary.csv")


#############################################
# 14. Kanal ve Bölge Analizi
#############################################

referral_summary = df.groupby("ReferralSource").agg(
    Musteri_Sayisi=(TARGET, "count"),
    Satin_Alma_Orani=(TARGET, "mean"),
    Ortalama_Harcama=("TotalSpent", "mean")
).sort_values("Satin_Alma_Orani", ascending=False)

referral_summary.to_csv(f"{OUTPUT_DIR}/referral_summary.csv")

region_summary = df.groupby("Region").agg(
    Musteri_Sayisi=(TARGET, "count"),
    Satin_Alma_Orani=(TARGET, "mean"),
    Ortalama_Harcama=("TotalSpent", "mean")
).sort_values("Satin_Alma_Orani", ascending=False)

region_summary.to_csv(f"{OUTPUT_DIR}/region_summary.csv")


#############################################
# 15. Kategori Analizi
#############################################

category_summary = df.groupby("ProductCategory").agg(
    Musteri_Sayisi=(TARGET, "count"),
    Satin_Alma_Orani=(TARGET, "mean"),
    Ortalama_Harcama=("TotalSpent", "mean"),
    Toplam_Min_Satis=("min_favorite_purchases", "sum")
).sort_values("Satin_Alma_Orani", ascending=False)

category_summary["Min_Satis_Orani"] = (
    category_summary["Toplam_Min_Satis"] /
    category_summary["Toplam_Min_Satis"].sum() * 100
)

category_summary.to_csv(f"{OUTPUT_DIR}/category_summary.csv")


#############################################
# 16. Outlier Analizi
#############################################

q1 = df["TotalSpent"].quantile(0.25)
q3 = df["TotalSpent"].quantile(0.75)
iqr = q3 - q1

lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr

outlier_df = df[
    (df["TotalSpent"] < lower_limit) |
    (df["TotalSpent"] > upper_limit)
]

outlier_df.to_csv(f"{OUTPUT_DIR}/totalspent_outliers.csv", index=False)

outlier_summary = pd.DataFrame({
    "Metric": [
        "Q1",
        "Q3",
        "IQR",
        "Lower_Limit",
        "Upper_Limit",
        "Outlier_Count",
        "Outlier_Ratio"
    ],
    "Value": [
        q1,
        q3,
        iqr,
        lower_limit,
        upper_limit,
        len(outlier_df),
        len(outlier_df) / len(df) * 100
    ]
})

outlier_summary.to_csv(f"{OUTPUT_DIR}/outlier_summary.csv", index=False)


#############################################
# 17. Kritik Müşteri Grupları
#############################################

satisfied_but_inactive = df[
    (df["CustomerSatisfaction"] >= 4) &
    (df["LastPurchaseDaysAgo"] > 90)
]

unsatisfied_but_recent = df[
    (df["CustomerSatisfaction"] <= 2) &
    (df["LastPurchaseDaysAgo"] <= 30)
]

loyalty_but_not_purchase = df[
    (df["LoyaltyProgram"] == 1) &
    (df[TARGET] == 0)
]

satisfied_but_inactive.to_csv(f"{OUTPUT_DIR}/satisfied_but_inactive.csv", index=False)
unsatisfied_but_recent.to_csv(f"{OUTPUT_DIR}/unsatisfied_but_recent.csv", index=False)
loyalty_but_not_purchase.to_csv(f"{OUTPUT_DIR}/loyalty_but_not_purchase.csv", index=False)

critical_summary = pd.DataFrame({
    "Group": [
        "Satisfied_But_Inactive",
        "Unsatisfied_But_Recent",
        "Loyalty_But_Not_Purchase"
    ],
    "Count": [
        len(satisfied_but_inactive),
        len(unsatisfied_but_recent),
        len(loyalty_but_not_purchase)
    ],
    "Ratio": [
        len(satisfied_but_inactive) / len(df) * 100,
        len(unsatisfied_but_recent) / len(df) * 100,
        len(loyalty_but_not_purchase) / len(df) * 100
    ],
    "Purchase_Rate": [
        satisfied_but_inactive[TARGET].mean(),
        unsatisfied_but_recent[TARGET].mean(),
        loyalty_but_not_purchase[TARGET].mean()
    ],
    "Avg_Recency": [
        satisfied_but_inactive["LastPurchaseDaysAgo"].mean(),
        unsatisfied_but_recent["LastPurchaseDaysAgo"].mean(),
        loyalty_but_not_purchase["LastPurchaseDaysAgo"].mean()
    ]
})

if "VIP" in df["CustomerSegment"].unique():
    vip_not_purchase = df[
        (df["CustomerSegment"] == "VIP") &
        (df[TARGET] == 0)
    ]

    vip_not_purchase.to_csv(f"{OUTPUT_DIR}/vip_not_purchase.csv", index=False)

    extra_row = pd.DataFrame({
        "Group": ["VIP_But_Not_Purchase"],
        "Count": [len(vip_not_purchase)],
        "Ratio": [len(vip_not_purchase) / len(df) * 100],
        "Purchase_Rate": [vip_not_purchase[TARGET].mean()],
        "Avg_Recency": [vip_not_purchase["LastPurchaseDaysAgo"].mean()]
    })

    critical_summary = pd.concat([critical_summary, extra_row], ignore_index=True)

critical_summary.to_csv(f"{OUTPUT_DIR}/critical_customer_groups_summary.csv", index=False)


#############################################
# 18. Feature Importance
#############################################

df_model = df.copy()

drop_cols = [
    TARGET,
    "AgeGroup",
    "IncomeGroup",
    "TenureGroup",
    "RecencyGroup",
    "TimeGroup",
    "min_favorite_purchases"
]

X = df_model.drop(columns=drop_cols, errors="ignore")
y = df_model[TARGET]

X = pd.get_dummies(X, drop_first=True)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf.fit(X, y)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

feature_importance.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)


#############################################
# 19. Streamlit Ana Sayfa İçin Master Summary
#############################################

eda_summary = pd.DataFrame({
    "Metric": [
        "Total_Rows",
        "Total_Columns",
        "Categorical_Columns",
        "Numerical_Columns",
        "Cardinal_Columns",
        "Purchase_Rate",
        "Missing_Value_Count",
        "Negative_Value_Column_Count",
        "High_Correlated_Column_Count",
        "TotalSpent_Outlier_Count",
        "TotalSpent_Outlier_Ratio"
    ],
    "Value": [
        df.shape[0],
        df.shape[1],
        len(cat_cols),
        len(num_cols),
        len(cat_but_car),
        round(df[TARGET].mean(), 4),
        df.isnull().sum().sum(),
        len(negative_cols),
        len(high_corr_cols),
        len(outlier_df),
        round(len(outlier_df) / len(df) * 100, 4)
    ]
})

eda_summary.to_csv(f"{OUTPUT_DIR}/eda_summary.csv", index=False)


#############################################
# 20. Final Temiz Veri Kaydı
#############################################

df.to_csv(f"{OUTPUT_DIR}/customerData_eda_enriched.csv", index=False)


#############################################
# 21. EDA Sonuç Yorumu
#############################################

eda_notes = pd.DataFrame({
    "Topic": [
        "EDA Yaklaşımı",
        "Target",
        "Correlation",
        "Feature Importance",
        "Leakage Uyarısı",
        "Business Insight"
    ],
    "Note": [
        "Analiz fonksiyonel EDA yapısı ile gerçekleştirilmiştir.",
        "PurchaseStatus hedef değişken olarak kullanılmıştır.",
        "Korelasyon analizinde hedef değişken multicollinearity kontrolünden çıkarılmıştır.",
        "RandomForest feature importance sadece keşif amaçlı kullanılmıştır.",
        "LastPurchaseDaysAgo gibi değişkenler hedefi çok güçlü açıklayabilir. Modelleme öncesi leakage kontrolü yapılmalıdır.",
        "Segment, recency, loyalty, memnuniyet ve VIP davranışları iş problemi açısından ayrıca analiz edilmiştir."
    ]
})

eda_notes.to_csv(f"{OUTPUT_DIR}/eda_notes.csv", index=False)


print("=" * 60)
print("EDA TAMAMLANDI")
print(f"Tüm çıktılar '{OUTPUT_DIR}/' klasörüne kaydedildi.")
print("=" * 60)

#############################################
# 22 Favori Kategori Minimum Satış Tahmini
#############################################

# Her müşteri için favori kategoriden minimum satın alma tahmini
df["min_favorite_purchases"] = (df["NumberOfPurchases"] // 5) + 1

print("##################### Favorite Category Minimum Sales Analysis #####################")

category_min_sales = df.groupby("ProductCategory")["min_favorite_purchases"].sum()
total_min_sales = category_min_sales.sum()

category_min_sales_summary = pd.DataFrame({
    "Toplam_Min_Satis": category_min_sales,
    "Oran": (category_min_sales / total_min_sales * 100).round(2)
}).sort_values("Toplam_Min_Satis", ascending=False)

print(category_min_sales_summary)

category_min_sales_summary.to_csv(
    f"{OUTPUT_DIR}/category_min_sales_summary.csv"
)


cat_gender_min_sales = df.groupby(
    ["ProductCategory", "Gender"]
)["min_favorite_purchases"].sum().unstack()

cat_gender_min_sales["Toplam"] = cat_gender_min_sales.sum(axis=1)

if "Female" in cat_gender_min_sales.columns:
    cat_gender_min_sales["Female_Ratio"] = (
        cat_gender_min_sales["Female"] /
        cat_gender_min_sales["Toplam"] * 100
    ).round(2)

if "Male" in cat_gender_min_sales.columns:
    cat_gender_min_sales["Male_Ratio"] = (
        cat_gender_min_sales["Male"] /
        cat_gender_min_sales["Toplam"] * 100
    ).round(2)

print("##################### Category x Gender Minimum Sales #####################")
print(cat_gender_min_sales)

cat_gender_min_sales.to_csv(
    f"{OUTPUT_DIR}/category_gender_min_sales.csv"
)


cat_segment_min_sales = df.groupby(
    ["ProductCategory", "CustomerSegment"]
)["min_favorite_purchases"].sum().unstack()

print("##################### Category x Segment Minimum Sales #####################")
print(cat_segment_min_sales)

cat_segment_min_sales.to_csv(
    f"{OUTPUT_DIR}/category_segment_min_sales.csv"
)


cat_region_min_sales = df.groupby(
    ["ProductCategory", "Region"]
)["min_favorite_purchases"].sum().unstack()

print("##################### Category x Region Minimum Sales #####################")
print(cat_region_min_sales)

cat_region_min_sales.to_csv(
    f"{OUTPUT_DIR}/category_region_min_sales.csv"
)


cat_referral_min_sales = df.groupby(
    ["ProductCategory", "ReferralSource"]
)["min_favorite_purchases"].sum().unstack()

print("##################### Category x ReferralSource Minimum Sales #####################")
print(cat_referral_min_sales)

cat_referral_min_sales.to_csv(
    f"{OUTPUT_DIR}/category_referral_min_sales.csv"
)


cat_age_min_sales = df.groupby(
    ["AgeGroup", "ProductCategory"],
    observed=True
)["min_favorite_purchases"].sum().unstack()

print("##################### AgeGroup x Category Minimum Sales #####################")
print(cat_age_min_sales)

cat_age_min_sales.to_csv(
    f"{OUTPUT_DIR}/age_category_min_sales.csv"
)