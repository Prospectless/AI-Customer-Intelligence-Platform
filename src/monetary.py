import pandas as pd
import numpy as np

# Veriyi yükle
df = pd.read_csv('data/customerData_500k.csv')
print("Veri yüklendi:", df.shape)

# Kategori bazlı ortalama sepet tutarı (USD)
category_basket = {
    'Electronics': 150,
    'Furniture': 200,
    'Fashion': 60,
    'Groceries': 30,
    'Sports': 80,
    'Kitchen': 50
}

# Segment çarpanı
segment_multiplier = {
    'Regular': 1.0,
    'Premium': 1.3,
    'VIP': 1.7
}

# Hesaplama
df['avg_basket'] = df['ProductCategory'].map(category_basket).fillna(70)
df['seg_mult'] = df['CustomerSegment'].map(segment_multiplier)
df['discount_factor'] = (1 - (df['DiscountsAvailed'] * 0.05)).clip(0.6, 1.0)
df['loyalty_bonus'] = df['LoyaltyProgram'].map({1: 1.1, 0: 1.0})

df['TotalSpent'] = (
    df['avg_basket']
    * df['seg_mult']
    * df['discount_factor']
    * df['loyalty_bonus']
    * df['NumberOfPurchases']
).round(2)

# Ara kolonları sil
df.drop(columns=['avg_basket', 'seg_mult', 'discount_factor', 'loyalty_bonus'], inplace=True)

# Kaydet
df.to_csv('data/customerData_500k_v2.csv', index=False)
print("Tamamlandı! Yeni dosya: data/customerData_500k_v2.csv")
print(df[['CustomerSegment', 'ProductCategory', 'NumberOfPurchases', 'TotalSpent']].head(10))