import os
import time
import pandas as pd
from dotenv import load_dotenv
import cohere

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

RFM_PATH = "data/RFM_Output/customer_rfm_final.csv"
CHURN_PATH = "data/Churn_Output/segment_churn_analysis.csv"
OUTPUT_DIR = "data/Persona_Output"

df = pd.read_csv(RFM_PATH)
churn_df = pd.read_csv(CHURN_PATH)

segment_profiles = df.groupby("RFM_Segment").agg(
    Musteri_Sayisi=("CustomerID", "count"),
    Ort_Recency=("recency", "mean"),
    Ort_Frequency=("frequency", "mean"),
    Ort_Monetary=("monetary", "mean"),
    Ort_Yas=("Age", "mean"),
    Ort_Gelir=("AnnualIncome", "mean"),
    Satin_Alma_Orani=("PurchaseStatus", "mean"),
    Loyalty_Orani=("LoyaltyProgram", "mean"),
    Ort_Memnuniyet=("CustomerSatisfaction", "mean"),
    Ort_Discount=("DiscountsAvailed", "mean"),
    Ort_TimeSpent=("TimeSpentOnWebsite", "mean"),
    En_Cok_Kategori=("ProductCategory", lambda x: x.mode()[0]),
    En_Cok_Bolge=("Region", lambda x: x.mode()[0]),
    En_Cok_Kanal=("ReferralSource", lambda x: x.mode()[0]),
    En_Cok_Cinsiyet=("Gender", lambda x: x.mode()[0]),
    En_Cok_Cihaz=("PreferredDevice", lambda x: x.mode()[0]),
).round(2).reset_index()

segment_profiles = segment_profiles.merge(
    churn_df[["RFM_Segment", "Gercek_Churn_Orani"]],
    on="RFM_Segment",
    how="left"
)

row = segment_profiles[segment_profiles["RFM_Segment"] == "promising"].iloc[0]

co = cohere.ClientV2(COHERE_API_KEY)

prompt = f"""Sen bir e-ticaret müşteri analitiği uzmanısın.
Aşağıdaki RFM segment verilerine göre detaylı bir müşteri personası oluştur.

SEGMENT ADI: promising

SEGMENT VERİLERİ:
- Müşteri Sayısı: {int(row['Musteri_Sayisi'])}
- Ortalama Son Alışveriş: {row['Ort_Recency']:.0f} gün önce
- Ortalama Alışveriş Sayısı: {row['Ort_Frequency']:.1f}
- Ortalama Harcama: ${row['Ort_Monetary']:.0f}
- Satın Alma Oranı: %{row['Satin_Alma_Orani']*100:.1f}
- Loyalty Üyelik Oranı: %{row['Loyalty_Orani']*100:.1f}
- Ortalama Memnuniyet: {row['Ort_Memnuniyet']:.1f}/5
- Ortalama İndirim Kullanımı: {row['Ort_Discount']:.1f}
- Ortalama Yaş: {row['Ort_Yas']:.0f}
- Ortalama Gelir: ${row['Ort_Gelir']:.0f}
- Favori Kategori: {row['En_Cok_Kategori']}
- Baskın Bölge: {row['En_Cok_Bolge']}
- Baskın Kanal: {row['En_Cok_Kanal']}
- Baskın Cinsiyet: {row['En_Cok_Cinsiyet']}
- Tercih Edilen Cihaz: {row['En_Cok_Cihaz']}

Lütfen aşağıdaki başlıkları kullanarak Türkçe bir müşteri personası yaz:

1. PERSONA ADI VE DEMOGRAFİ
2. DAVRANIŞ PROFİLİ
3. MOTİVASYONLAR VE İHTİYAÇLAR
4. RİSK VE FIRSATLAR
5. ÖNERİLEN KAMPANYA STRATEJİSİ

Her başlık için 2-3 cümle. Kısa ve aksiyon odaklı yaz."""

print("promising segmenti için persona üretiliyor...")

try:
    response = co.chat(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": prompt}],
    )

    persona_text = response.message.content[0].text.strip()
    print(f"✅ Persona üretildi ({len(persona_text)} karakter)")

    # txt olarak kaydet
    with open(f"{OUTPUT_DIR}/promising_persona.txt", "w", encoding="utf-8") as f:
        f.write(f"SEGMENT: promising\n")
        f.write(f"Müşteri Sayısı: {int(row['Musteri_Sayisi'])}\n")
        f.write(f"Satın Alma Oranı: %{row['Satin_Alma_Orani']*100:.1f}\n")
        f.write(f"Ort. Harcama: ${row['Ort_Monetary']:.0f}\n")
        f.write("=" * 60 + "\n\n")
        f.write(persona_text)

    # CSV'ye ekle
    personas_df = pd.read_csv(f"{OUTPUT_DIR}/segment_personas.csv")
    promising_row = pd.DataFrame([{
        "RFM_Segment": "promising",
        "Musteri_Sayisi": int(row["Musteri_Sayisi"]),
        "Satin_Alma_Orani": row["Satin_Alma_Orani"],
        "Ort_Monetary": row["Ort_Monetary"],
        "Ort_Recency": row["Ort_Recency"],
        "Persona": persona_text
    }])

    # Eski hatalı satırı güncelle
    personas_df = personas_df[personas_df["RFM_Segment"] != "promising"]
    personas_df = pd.concat([personas_df, promising_row], ignore_index=True)
    personas_df.to_csv(f"{OUTPUT_DIR}/segment_personas.csv", index=False)

    print("✅ CSV güncellendi.")
    print("\nPersona:")
    print(persona_text)

except Exception as e:
    print(f"❌ Hata: {e}")