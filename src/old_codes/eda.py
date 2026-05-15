import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi yükle
df = pd.read_csv('data/clean_customer_data.csv')

print("=" * 60)
print("GENEL BİLGİ")
print("=" * 60)
print("Shape:", df.shape)
print("\nKolon Tipleri:")
print(df.dtypes)

print("\n" + "=" * 60)
print("İLK 5 SATIR")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("EKSİK DEĞERLER")
print("=" * 60)
print(df.isnull().sum())

print("\n" + "=" * 60)
print("TEMEL İSTATİSTİKLER - SAYISAL KOLONLAR")
print("=" * 60)
print(df.describe())

print("\n" + "=" * 60)
print("NEGATİF DEĞER ANALİZİ")
print("=" * 60)
sayisal_kolonlar = df.select_dtypes(include=[np.number]).columns
for kolon in sayisal_kolonlar:
    negatif = df[df[kolon] < 0].shape[0]
    if negatif > 0:
        print(f"\n{kolon}: {negatif} adet negatif değer ({negatif/len(df)*100:.2f}%)")
        print(df[df[kolon] < 0][kolon].describe())

print("\n" + "=" * 60)
print("NEGATİF LastPurchaseDaysAgo ANALİZİ")
print("=" * 60)
neg_last = df[df['LastPurchaseDaysAgo'] < 0]
print(f"Adet: {len(neg_last)}")
print(neg_last[['LastPurchaseDaysAgo', 'NumberOfPurchases', 'PurchaseStatus', 'CustomerSegment']].describe())
print("\nPurchaseStatus dağılımı:")
print(neg_last['PurchaseStatus'].value_counts())
print("\nCustomerSegment dağılımı:")
print(neg_last['CustomerSegment'].value_counts())

print("\n" + "=" * 60)
print("NEGATİF TimeSpentOnWebsite ANALİZİ")
print("=" * 60)
neg_time = df[df['TimeSpentOnWebsite'] < 0]
print(f"Adet: {len(neg_time)}")
if len(neg_time) > 0:
    print(neg_time[['TimeSpentOnWebsite', 'PurchaseStatus', 'CustomerSegment']].describe())
    print("\nPurchaseStatus dağılımı:")
    print(neg_time['PurchaseStatus'].value_counts())

print("\n" + "=" * 60)
print("KATEGORİK KOLON ANALİZİ")
print("=" * 60)
kategorik_kolonlar = df.select_dtypes(include=['object']).columns
for kolon in kategorik_kolonlar:
    print(f"\n{kolon}:")
    print(df[kolon].value_counts())
    print(f"Unique değer sayısı: {df[kolon].nunique()}")

print("\n" + "=" * 60)
print("SAYISAL KOLON DAĞILIMLARI")
print("=" * 60)
for kolon in sayisal_kolonlar:
    q1 = df[kolon].quantile(0.25)
    q3 = df[kolon].quantile(0.75)
    iqr = q3 - q1
    alt_sinir = q1 - 1.5 * iqr
    ust_sinir = q3 + 1.5 * iqr
    aykiri = df[(df[kolon] < alt_sinir) | (df[kolon] > ust_sinir)].shape[0]
    print(f"\n{kolon}:")
    print(f"  Min: {df[kolon].min():.2f} | Max: {df[kolon].max():.2f} | Ort: {df[kolon].mean():.2f}")
    print(f"  Aykırı değer (IQR): {aykiri} adet ({aykiri/len(df)*100:.2f}%)")

print("\n" + "=" * 60)
print("KORELASYON MATRİSİ")
print("=" * 60)
print(df[sayisal_kolonlar].corr().round(2))

print("\n" + "=" * 60)
print("HEDEF DEĞİŞKEN - PurchaseStatus")
print("=" * 60)
print(df['PurchaseStatus'].value_counts())
print(f"Satın alma oranı: {df['PurchaseStatus'].mean()*100:.1f}%")

print("\n" + "=" * 60)
print("SEGMENT BAZLI ÖZET")
print("=" * 60)
print(df.groupby('CustomerSegment')[['NumberOfPurchases', 'TotalSpent', 'CustomerSatisfaction']].mean().round(2))