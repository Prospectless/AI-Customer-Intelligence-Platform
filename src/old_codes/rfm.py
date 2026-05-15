import pandas as pd
import numpy as np

# Veriyi yükle, iadeleri filtrele
df = pd.read_csv('data/customerData_500k_v2.csv')
df = df[df['NumberOfPurchases'] > 0].copy()
print("Temiz veri:", df.shape)

# RFM metrikleri
df['Recency'] = df['LastPurchaseDaysAgo']
df['Frequency'] = df['NumberOfPurchases']
df['Monetary'] = df['TotalSpent']

# RFM skorları (1-5)
df['R_Score'] = pd.qcut(df['Recency'], q=5, labels=[5,4,3,2,1])
df['F_Score'] = pd.qcut(df['Frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5])
df['M_Score'] = pd.qcut(df['Monetary'], q=5, labels=[1,2,3,4,5])

# 10 Segment fonksiyonu
def rfm_segment(row):
    r = int(row['R_Score'])
    f = int(row['F_Score'])
    m = int(row['M_Score'])

    if r >= 4 and f >= 4 and m >= 4:
        return 'champions'
    elif r >= 4 and f <= 2:
        return 'new_customers'
    elif r >= 3 and f >= 3:
        return 'loyal_customers'
    elif r >= 3 and f <= 2 and m >= 3:
        return 'potential_loyalists'
    elif r >= 3 and f <= 2:
        return 'promising'
    elif r <= 2 and f != 3 and m >= 3:
        return 'cant_loose'
    elif r <= 2 and f >= 3 and m >= 3:
        return 'at_risk'
    elif r <= 2 and f >= 3 and m <= 2:
        return 'need_attention'
    elif r == 1 and f <= 2 and m <= 2:
        return 'hibernating'
    elif r <= 3 and f <= 2 and m <= 2:
        return 'about_to_sleep'
    else:
        return 'hibernating'

df['RFM_Segment'] = df.apply(rfm_segment, axis=1)

# Kaydet
df.to_csv('data/customerData_rfm.csv', index=False)
print("RFM tamamlandı!")
print(df['RFM_Segment'].value_counts())

# Toplam Hesaplanan Müşteri
print("\n--- Toplam Hesaplanan Müşteri ---")
print("Segmentlerdeki toplam:", df['RFM_Segment'].value_counts().sum())
print("DataFrame toplam:", len(df))
print("Fark:", len(df) - df['RFM_Segment'].value_counts().sum())