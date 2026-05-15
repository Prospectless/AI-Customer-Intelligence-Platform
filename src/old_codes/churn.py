import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score
)
from xgboost import XGBClassifier

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

#############################################
# 1. PATHS
#############################################

DATA_PATH = "data/RFM_Output/customer_rfm_final.csv"
OUTPUT_DIR = "data/Churn_Output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

#############################################
# 2. LOAD DATA
#############################################

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("CHURN PREDICTION BAŞLADI")
print("=" * 60)
print(f"Veri boyutu: {df.shape}")
print(f"\nHedef değişken dağılımı:")
print(df["PurchaseStatus"].value_counts())
print(f"Satın alma oranı: {df['PurchaseStatus'].mean()*100:.1f}%")
print(f"Churn oranı (1-Purchase): {(1 - df['PurchaseStatus'].mean())*100:.1f}%")

#############################################
# 3. FEATURE SELECTION
#############################################

# Kategorik kolonları encode et
le = LabelEncoder()

cat_cols = [
    "Gender",
    "ProductCategory",
    "PreferredDevice",
    "Region",
    "ReferralSource",
    "CustomerSegment",
    "RFM_Segment",
    "DiscountAffinity"
]

df_model = df.copy()

for col in cat_cols:
    if col in df_model.columns:
        df_model[col] = le.fit_transform(df_model[col].astype(str))

# Kullanılacak özellikler
feature_cols = [
    "Age",
    "AnnualIncome",
    "NumberOfPurchases",
    "TimeSpentOnWebsite",
    "CustomerTenureYears",
    "LastPurchaseDaysAgo",
    "LoyaltyProgram",
    "DiscountsAvailed",
    "SessionCount",
    "CustomerSatisfaction",
    "TotalSpent",
    "min_favorite_purchases",
    "recency_score",
    "frequency_score",
    "monetary_score",
    "Gender",
    "ProductCategory",
    "PreferredDevice",
    "Region",
    "ReferralSource",
    "CustomerSegment",
    "RFM_Segment",
    "DiscountAffinity"
]

# Sadece mevcut kolonları al
feature_cols = [col for col in feature_cols if col in df_model.columns]

X = df_model[feature_cols]
y = df_model["PurchaseStatus"]

print(f"\nKullanılan feature sayısı: {len(feature_cols)}")
print(f"Feature listesi: {feature_cols}")

#############################################
# 4. TRAIN-TEST SPLIT
#############################################

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain boyutu: {X_train.shape}")
print(f"Test boyutu: {X_test.shape}")

#############################################
# 5. SCALING
#############################################

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#############################################
# 6. MODEL 1 - LOGISTIC REGRESSION
#############################################

print("\n" + "=" * 60)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 60)

lr = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

lr_auc = roc_auc_score(y_test, y_prob_lr)
lr_f1 = f1_score(y_test, y_pred_lr)
lr_acc = accuracy_score(y_test, y_pred_lr)

print(f"AUC: {lr_auc:.4f}")
print(f"F1 Score: {lr_f1:.4f}")
print(f"Accuracy: {lr_acc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

#############################################
# 7. MODEL 2 - RANDOM FOREST
#############################################

print("\n" + "=" * 60)
print("MODEL 2: RANDOM FOREST")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

rf_auc = roc_auc_score(y_test, y_prob_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_acc = accuracy_score(y_test, y_pred_rf)

print(f"AUC: {rf_auc:.4f}")
print(f"F1 Score: {rf_f1:.4f}")
print(f"Accuracy: {rf_acc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

#############################################
# 8. MODEL 3 - XGBOOST
#############################################

print("\n" + "=" * 60)
print("MODEL 3: XGBOOST")
print("=" * 60)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb = XGBClassifier(
    n_estimators=100,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,
    eval_metric="logloss",
    verbosity=0
)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

xgb_auc = roc_auc_score(y_test, y_prob_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb)
xgb_acc = accuracy_score(y_test, y_pred_xgb)

print(f"AUC: {xgb_auc:.4f}")
print(f"F1 Score: {xgb_f1:.4f}")
print(f"Accuracy: {xgb_acc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))

#############################################
# 9. MODEL KARŞILAŞTIRMASI
#############################################

print("\n" + "=" * 60)
print("MODEL KARŞILAŞTIRMASI")
print("=" * 60)

model_comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "AUC": [lr_auc, rf_auc, xgb_auc],
    "F1_Score": [lr_f1, rf_f1, xgb_f1],
    "Accuracy": [lr_acc, rf_acc, xgb_acc]
}).round(4)

print(model_comparison)
model_comparison.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)

best_model_name = model_comparison.loc[model_comparison["AUC"].idxmax(), "Model"]
print(f"\nEn iyi model (AUC): {best_model_name}")

#############################################
# 10. FEATURE IMPORTANCE
#############################################

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (XGBoost)")
print("=" * 60)

feature_importance = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": xgb.feature_importances_
}).sort_values("Importance", ascending=False).round(4)

print(feature_importance.head(15))
feature_importance.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)

#############################################
# 11. ROC CURVE GRAFİĞİ
#############################################

plt.figure(figsize=(10, 7))

for name, y_prob, auc in [
    ("Logistic Regression", y_prob_lr, lr_auc),
    ("Random Forest", y_prob_rf, rf_auc),
    ("XGBoost", y_prob_xgb, xgb_auc)
]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.4f})")

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Churn Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()

#############################################
# 12. CONFUSION MATRIX
#############################################

best_y_pred = y_pred_xgb
cm = confusion_matrix(y_test, best_y_pred)

plt.figure(figsize=(7, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix - XGBoost")
plt.colorbar()
plt.xticks([0, 1], ["Satın Almaz (Churn)", "Satın Alır"])
plt.yticks([0, 1], ["Satın Almaz (Churn)", "Satın Alır"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black",
                fontsize=14)

plt.ylabel("Gerçek")
plt.xlabel("Tahmin")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

#############################################
# 13. FEATURE IMPORTANCE GRAFİĞİ
#############################################

plt.figure(figsize=(10, 8))
top15 = feature_importance.head(15)
plt.barh(top15["Feature"][::-1], top15["Importance"][::-1])
plt.xlabel("Importance")
plt.title("Top 15 Feature Importance - XGBoost")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

#############################################
# 14. TAHMİNLERİ KAYDET
#############################################

# Test setindeki index'leri al
test_index = X_test.index

predictions = pd.DataFrame({
    "CustomerID": df.loc[test_index, "CustomerID"],
    "RFM_Segment": df.loc[test_index, "RFM_Segment"],
    "Gercek_PurchaseStatus": y_test.values,
    "LR_Tahmin": y_pred_lr,
    "LR_Olasilik": y_prob_lr.round(4),
    "RF_Tahmin": y_pred_rf,
    "RF_Olasilik": y_prob_rf.round(4),
    "XGB_Tahmin": y_pred_xgb,
    "XGB_Olasilik": y_prob_xgb.round(4),
})

predictions.to_csv(f"{OUTPUT_DIR}/churn_predictions.csv", index=False)

#############################################
# 15. SEGMENT BAZLI CHURN ANALİZİ - FIXED
#############################################

print("\n" + "=" * 60)
print("SEGMENT BAZLI CHURN ANALİZİ (FIXED)")
print("=" * 60)

# FIX: Churn = NOT purchasing (PurchaseStatus = 0)
predictions_for_segment = predictions.copy()
predictions_for_segment["Gercek_Churn"] = 1 - predictions_for_segment["Gercek_PurchaseStatus"]
predictions_for_segment["XGB_Churn_Tahmin"] = 1 - predictions_for_segment["XGB_Tahmin"]

segment_churn = predictions_for_segment.groupby("RFM_Segment").agg(
    Musteri_Sayisi=("CustomerID", "count"),
    Gercek_Churn_Orani=("Gercek_Churn", "mean"),  # FIXED: Now calculates churn correctly
    XGB_Tahmin_Orani=("XGB_Churn_Tahmin", "mean"),  # FIXED: Now predicts churn correctly
    Ort_Churn_Olasiligi=("XGB_Olasilik", "mean")  # Churn probability
).round(3).sort_values("Gercek_Churn_Orani", ascending=False)

print(segment_churn)
segment_churn.to_csv(f"{OUTPUT_DIR}/segment_churn_analysis.csv")

#############################################
# 16. ÖZET
#############################################

print("\n" + "=" * 60)
print("CHURN PREDICTION TAMAMLANDI")
print("=" * 60)
print(f"\nModel Sonuçları:")
print(model_comparison.to_string())
print(f"\nEn iyi model: {best_model_name}")
print(f"\nSegment Churn Analizi:")
print(segment_churn.to_string())
print(f"\nKaydedilen dosyalar:")
for f_name in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  ✅ {OUTPUT_DIR}/{f_name}")