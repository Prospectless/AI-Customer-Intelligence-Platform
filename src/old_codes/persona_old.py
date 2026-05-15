import os
import time
import pandas as pd
from dotenv import load_dotenv
import cohere

pd.set_option("display.max_columns", None)

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

#############################################
# 1. PATHS
#############################################

RFM_PATH = "data/RFM_Output/customer_rfm_final.csv"
CHURN_PATH = "data/Churn_Output/segment_churn_analysis.csv"
OUTPUT_DIR = "data/Persona_Output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

#############################################
# 2. LOAD DATA
#############################################

df = pd.read_csv(RFM_PATH)
churn_df = pd.read_csv(CHURN_PATH)

print("=" * 60)
print("LLM PERSONA ENGINE BAŞLADI")
print("=" * 60)

#############################################
# 3. SEGMENT PROFİLLERİ HAZIRLA
#############################################

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
).round(2).reset_index()  # ← reset_index eklendi

segment_profiles = segment_profiles.merge(
    churn_df[["RFM_Segment", "Gercek_Churn_Orani"]],
    on="RFM_Segment",
    how="left"
)

print("Segment profilleri hazırlandı:")
print(segment_profiles[["RFM_Segment", "Musteri_Sayisi", "Ort_Recency", "Satin_Alma_Orani"]])

#############################################
# 4. COHERE CLIENT
#############################################

co = cohere.ClientV2(COHERE_API_KEY)  # ← V2 kullanıyoruz

#############################################
# 5. PERSONA PROMPT
#############################################

def build_prompt(segment_name, profile):
    return f"""Sen bir e-ticaret müşteri analitiği uzmanısın.
Aşağıdaki RFM segment verilerine göre detaylı bir müşteri personası oluştur.

SEGMENT ADI: {segment_name}

SEGMENT VERİLERİ:
- Müşteri Sayısı: {int(profile['Musteri_Sayisi'])}
- Ortalama Son Alışveriş: {profile['Ort_Recency']:.0f} gün önce
- Ortalama Alışveriş Sayısı: {profile['Ort_Frequency']:.1f}
- Ortalama Harcama: ${profile['Ort_Monetary']:.0f}
- Satın Alma Oranı: %{profile['Satin_Alma_Orani']*100:.1f}
- Loyalty Üyelik Oranı: %{profile['Loyalty_Orani']*100:.1f}
- Ortalama Memnuniyet: {profile['Ort_Memnuniyet']:.1f}/5
- Ortalama İndirim Kullanımı: {profile['Ort_Discount']:.1f}
- Ortalama Yaş: {profile['Ort_Yas']:.0f}
- Ortalama Gelir: ${profile['Ort_Gelir']:.0f}
- Favori Kategori: {profile['En_Cok_Kategori']}
- Baskın Bölge: {profile['En_Cok_Bolge']}
- Baskın Kanal: {profile['En_Cok_Kanal']}
- Baskın Cinsiyet: {profile['En_Cok_Cinsiyet']}
- Tercih Edilen Cihaz: {profile['En_Cok_Cihaz']}

Lütfen aşağıdaki başlıkları kullanarak Türkçe bir müşteri personası yaz:

1. PERSONA ADI VE DEMOGRAFİ
2. DAVRANIŞ PROFİLİ
3. MOTİVASYONLAR VE İHTİYAÇLAR
4. RİSK VE FIRSATLAR
5. ÖNERİLEN KAMPANYA STRATEJİSİ

Her başlık için 2-3 cümle. Kısa ve aksiyon odaklı yaz."""

#############################################
# 6. PERSONA ÜRETİMİ
#############################################

personas = []

print(f"\n{len(segment_profiles)} segment için persona üretiliyor...")
print("Trial key: dakikada 5 istek limiti, segmentler arasında 15sn bekleniyor.\n")

for _, row in segment_profiles.iterrows():
    segment = row["RFM_Segment"]
    print(f"--- {segment} ---")

    prompt = build_prompt(segment, row)

    try:
        response = co.chat(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": prompt}],
        )

        persona_text = response.message.content[0].text.strip()
        print(f"✅ Persona üretildi ({len(persona_text)} karakter)")

        personas.append({
            "RFM_Segment": segment,
            "Musteri_Sayisi": int(row["Musteri_Sayisi"]),
            "Satin_Alma_Orani": row["Satin_Alma_Orani"],
            "Ort_Monetary": row["Ort_Monetary"],
            "Ort_Recency": row["Ort_Recency"],
            "Persona": persona_text
        })

    except Exception as e:
        print(f"❌ Hata: {e}")
        personas.append({
            "RFM_Segment": segment,
            "Musteri_Sayisi": int(row["Musteri_Sayisi"]),
            "Satin_Alma_Orani": row["Satin_Alma_Orani"],
            "Ort_Monetary": row["Ort_Monetary"],
            "Ort_Recency": row["Ort_Recency"],
            "Persona": f"Hata: {str(e)}"
        })

    # Trial key rate limit: 15sn bekle
    print("15sn bekleniyor...")
    time.sleep(15)

#############################################
# 7. KAYDET
#############################################

personas_df = pd.DataFrame(personas)
personas_df.to_csv(f"{OUTPUT_DIR}/segment_personas.csv", index=False)

for _, row in personas_df.iterrows():
    segment_name = str(row["RFM_Segment"]).replace(" ", "_")
    with open(f"{OUTPUT_DIR}/{segment_name}_persona.txt", "w", encoding="utf-8") as f:
        f.write(f"SEGMENT: {row['RFM_Segment']}\n")
        f.write(f"Müşteri Sayısı: {row['Musteri_Sayisi']}\n")
        f.write(f"Satın Alma Oranı: %{row['Satin_Alma_Orani']*100:.1f}\n")
        f.write(f"Ort. Harcama: ${row['Ort_Monetary']:.0f}\n")
        f.write("=" * 60 + "\n\n")
        f.write(str(row["Persona"]))

print("\n" + "=" * 60)
print("LLM PERSONA ENGINE TAMAMLANDI")
print("=" * 60)
print(f"Toplam {len(personas)} persona üretildi.")
print(f"\nKaydedilen dosyalar:")
for f_name in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  ✅ {OUTPUT_DIR}/{f_name}")