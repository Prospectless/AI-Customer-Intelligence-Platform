import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/customerData_500k_v2.csv')

print("=" * 60)
print("1. CİNSİYET KIRILIMI")
print("=" * 60)

print("\n--- Cinsiyet Bazlı Temel Metrikler ---")
print(df.groupby('Gender')[['NumberOfPurchases', 'TotalSpent', 'CustomerSatisfaction', 'TimeSpentOnWebsite', 'SessionCount']].mean().round(2))

print("\n--- Cinsiyet × PurchaseStatus ---")
print(df.groupby('Gender')['PurchaseStatus'].value_counts(normalize=True).round(3))

print("\n--- Cinsiyet × ProductCategory ---")
print(df.groupby(['Gender', 'ProductCategory']).size().unstack().fillna(0).astype(int))

print("\n--- Cinsiyet × CustomerSegment ---")
print(df.groupby(['Gender', 'CustomerSegment']).size().unstack().fillna(0).astype(int))

print("\n--- Cinsiyet × PreferredDevice ---")
print(df.groupby(['Gender', 'PreferredDevice']).size().unstack().fillna(0).astype(int))

print("\n--- Cinsiyet × LoyaltyProgram ---")
print(df.groupby('Gender')['LoyaltyProgram'].mean().round(3))

print("\n--- Cinsiyet × Region ---")
print(df.groupby(['Gender', 'Region']).size().unstack().fillna(0).astype(int))

print("\n" + "=" * 60)
print("2. KANAL (ReferralSource) KIRILIMI")
print("=" * 60)

print("\n--- Kanal Bazlı Müşteri Sayısı ve Oranı ---")
kanal = df['ReferralSource'].value_counts()
print(pd.DataFrame({'Müşteri': kanal, 'Oran (%)': (kanal/len(df)*100).round(2)}))

print("\n--- Kanal × PurchaseStatus ---")
print(df.groupby('ReferralSource')['PurchaseStatus'].agg(['sum', 'mean']).rename(columns={'sum': 'Toplam Satın Alma', 'mean': 'Satın Alma Oranı'}).round(3))

print("\n--- Kanal × TotalSpent ---")
print(df.groupby('ReferralSource')['TotalSpent'].describe().round(2))

print("\n--- Kanal × CustomerSegment ---")
print(df.groupby(['ReferralSource', 'CustomerSegment']).size().unstack().fillna(0).astype(int))

print("\n--- Kanal × ProductCategory ---")
print(df.groupby(['ReferralSource', 'ProductCategory']).size().unstack().fillna(0).astype(int))

print("\n--- Kanal × Cinsiyet ---")
print(df.groupby(['ReferralSource', 'Gender']).size().unstack().fillna(0).astype(int))

print("\n" + "=" * 60)
print("3. KATEGORİ KIRILIMI")
print("=" * 60)

print("\n--- Kategori Bazlı Temel Metrikler ---")
print(df.groupby('ProductCategory')[['TotalSpent', 'NumberOfPurchases', 'CustomerSatisfaction']].mean().round(2))

print("\n--- Kategori × Cinsiyet ---")
print(df.groupby(['ProductCategory', 'Gender']).size().unstack().fillna(0).astype(int))

print("\n--- Kategori × Region ---")
print(df.groupby(['ProductCategory', 'Region']).size().unstack().fillna(0).astype(int))

print("\n--- Kategori × PurchaseStatus ---")
print(df.groupby('ProductCategory')['PurchaseStatus'].mean().round(3))

print("\n--- Kategori × PreferredDevice ---")
print(df.groupby(['ProductCategory', 'PreferredDevice']).size().unstack().fillna(0).astype(int))

print("\n" + "=" * 60)
print("4. DAVRANIŞSAL ANALİZ")
print("=" * 60)

print("\n--- LoyaltyProgram × PurchaseStatus ---")
print(df.groupby('LoyaltyProgram')['PurchaseStatus'].agg(['sum', 'mean']).rename(columns={'sum': 'Toplam', 'mean': 'Oran'}).round(3))

print("\n--- DiscountsAvailed × CustomerSatisfaction ---")
print(df.groupby('DiscountsAvailed')['CustomerSatisfaction'].mean().round(3))

print("\n--- SessionCount × PurchaseStatus ---")
print(df.groupby('SessionCount')['PurchaseStatus'].mean().round(3))

print("\n--- TimeSpentOnWebsite Grupları × PurchaseStatus ---")
df['TimeGroup'] = pd.cut(df['TimeSpentOnWebsite'], bins=[-5, 0, 20, 40, 60, 80], labels=['Negatif', '0-20', '20-40', '40-60', '60+'])
print(df.groupby('TimeGroup', observed=True)['PurchaseStatus'].mean().round(3))

print("\n" + "=" * 60)
print("5. DEMOGRAFİK ANALİZ")
print("=" * 60)

df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 25, 35, 50, 100], labels=['18-25', '26-35', '36-50', '50+'])
df['IncomeGroup'] = pd.cut(df['AnnualIncome'], bins=[0, 50000, 100000, 150000, 300000], labels=['Düşük', 'Orta', 'Yüksek', 'Çok Yüksek'])

print("\n--- Yaş Grubu × ProductCategory ---")
print(df.groupby(['AgeGroup', 'ProductCategory'], observed=True).size().unstack().fillna(0).astype(int))

print("\n--- Yaş Grubu × PurchaseStatus ---")
print(df.groupby('AgeGroup', observed=True)['PurchaseStatus'].mean().round(3))

print("\n--- Yaş Grubu × PreferredDevice ---")
print(df.groupby(['AgeGroup', 'PreferredDevice'], observed=True).size().unstack().fillna(0).astype(int))

print("\n--- Gelir Grubu × TotalSpent ---")
print(df.groupby('IncomeGroup', observed=True)['TotalSpent'].mean().round(2))

print("\n--- Gelir Grubu × CustomerSegment ---")
print(df.groupby(['IncomeGroup', 'CustomerSegment'], observed=True).size().unstack().fillna(0).astype(int))

print("\n" + "=" * 60)
print("6. TENURE ANALİZİ")
print("=" * 60)

df['TenureGroup'] = pd.cut(df['CustomerTenureYears'], bins=[-1, 0, 1, 3, 6, 20], labels=['Negatif', '0-1 Yıl', '1-3 Yıl', '3-6 Yıl', '6+ Yıl'])

print("\n--- Tenure Grubu × CustomerSatisfaction ---")
print(df.groupby('TenureGroup', observed=True)['CustomerSatisfaction'].mean().round(3))

print("\n--- Tenure Grubu × LoyaltyProgram ---")
print(df.groupby('TenureGroup', observed=True)['LoyaltyProgram'].mean().round(3))

print("\n--- Tenure Grubu × TotalSpent ---")
print(df.groupby('TenureGroup', observed=True)['TotalSpent'].mean().round(2))

print("\n--- Tenure Grubu × PurchaseStatus ---")
print(df.groupby('TenureGroup', observed=True)['PurchaseStatus'].mean().round(3))

print("\n" + "=" * 60)
print("7. CHURN ODAKLI ANALİZ")
print("=" * 60)

df['RecencyGroup'] = pd.cut(df['LastPurchaseDaysAgo'], bins=[-20, 0, 30, 60, 90, 200], labels=['Negatif', '0-30 Gün', '30-60 Gün', '60-90 Gün', '90+ Gün'])

print("\n--- Recency Grubu × PurchaseStatus ---")
print(df.groupby('RecencyGroup', observed=True)['PurchaseStatus'].mean().round(3))

print("\n--- CustomerSatisfaction × PurchaseStatus ---")
print(df.groupby('CustomerSatisfaction')['PurchaseStatus'].mean().round(3))

print("\n--- CustomerSatisfaction × TotalSpent ---")
print(df.groupby('CustomerSatisfaction')['TotalSpent'].mean().round(2))

print("\n--- Segment × PurchaseStatus ---")
print(df.groupby('CustomerSegment')['PurchaseStatus'].mean().round(3))

print("\n--- LoyaltyProgram × TotalSpent × PurchaseStatus ---")
print(df.groupby(['LoyaltyProgram', 'PurchaseStatus'])['TotalSpent'].mean().round(2))

print("\n" + "=" * 60)
print("8. KATEGORİ BAZLI TAHMİNİ SATIŞ DAĞILIMI")
print("=" * 60)

# Her müşteri için favori kategoriden minimum satın alma tahmini
df['min_favorite_purchases'] = (df['NumberOfPurchases'] // 5) + 1

print("\n--- Kategori Bazlı Toplam Minimum Satış Tahmini ---")
category_min_sales = df.groupby('ProductCategory')['min_favorite_purchases'].sum()
total_min = category_min_sales.sum()
category_ratio = pd.DataFrame({
    'Toplam Min Satış': category_min_sales,
    'Oran (%)': (category_min_sales / total_min * 100).round(2)
})
print(category_ratio)

print("\n--- Kategori × Cinsiyet Bazlı Min Satış ---")
cat_gender = df.groupby(['ProductCategory', 'Gender'])['min_favorite_purchases'].sum().unstack()
cat_gender['Toplam'] = cat_gender.sum(axis=1)
cat_gender['Female (%)'] = (cat_gender['Female'] / cat_gender['Toplam'] * 100).round(2)
cat_gender['Male (%)'] = (cat_gender['Male'] / cat_gender['Toplam'] * 100).round(2)
print(cat_gender)

print("\n--- Kategori × Segment Bazlı Min Satış ---")
cat_segment = df.groupby(['ProductCategory', 'CustomerSegment'])['min_favorite_purchases'].sum().unstack()
print(cat_segment)

print("\n--- Kategori × Region Bazlı Min Satış ---")
cat_region = df.groupby(['ProductCategory', 'Region'])['min_favorite_purchases'].sum().unstack()
print(cat_region)

print("\n--- Kategori × Kanal Bazlı Min Satış ---")
cat_kanal = df.groupby(['ProductCategory', 'ReferralSource'])['min_favorite_purchases'].sum().unstack()
print(cat_kanal)

print("\n--- Yaş Grubu × Kategori Bazlı Min Satış ---")
cat_age = df.groupby(['AgeGroup', 'ProductCategory'], observed=True)['min_favorite_purchases'].sum().unstack()
print(cat_age)