import os
import time
import pandas as pd
from dotenv import load_dotenv
import cohere

pd.set_option("display.max_columns", None)

# =========================================================
# ENV
# =========================================================

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# =========================================================
# PATHS
# =========================================================

RFM_PATH = "data/RFM_Output/customer_rfm_final.csv"
CHURN_PATH = "data/Churn_Output/segment_churn_analysis.csv"

OUTPUT_DIR = "data/Persona_Output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(RFM_PATH)
churn_df = pd.read_csv(CHURN_PATH)

print("=" * 60)
print("AI PERSONA ENGINE STARTED")
print("=" * 60)

# =========================================================
# SEGMENT PROFILE GENERATION
# =========================================================

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

# =========================================================
# MERGE CHURN INFO
# =========================================================

segment_profiles = segment_profiles.merge(
    churn_df[["RFM_Segment", "Gercek_Churn_Orani"]],
    on="RFM_Segment",
    how="left"
)

print("\nSegment profiles created:")
print(
    segment_profiles[
        [
            "RFM_Segment",
            "Musteri_Sayisi",
            "Ort_Recency",
            "Satin_Alma_Orani"
        ]
    ]
)

# =========================================================
# COHERE CLIENT
# =========================================================

co = cohere.ClientV2(COHERE_API_KEY)

# =========================================================
# PROMPT BUILDER
# =========================================================

def build_prompt(segment_name, profile):

    return f"""
Sen bir e-ticaret müşteri analitiği ve CRM strateji uzmanısın.

Aşağıdaki RFM segment verilerine göre sunumda gösterilecek,
kısa, net ve aksiyon odaklı bir müşteri personası oluştur.

KESİN KURALLAR:
- HTML etiketi kullanma.
- Markdown tablo kullanma.
- Kod bloğu kullanma.
- Sadece düz metin kullan.
- Başlıkları sade yaz.
- Her bölüm maksimum 2 cümle olsun.
- Cevabı Türkçe üret.
- Business ve CRM ekiplerine yönelik yaz.

SEGMENT ADI:
{segment_name}

SEGMENT VERİLERİ:

- Müşteri Sayısı: {int(profile['Musteri_Sayisi'])}

- Ortalama Son Alışveriş:
{profile['Ort_Recency']:.0f} gün önce

- Ortalama Alışveriş Sayısı:
{profile['Ort_Frequency']:.1f}

- Ortalama Harcama:
${profile['Ort_Monetary']:.0f}

- Satın Alma Oranı:
%{profile['Satin_Alma_Orani']*100:.1f}

- Churn Oranı:
%{profile['Gercek_Churn_Orani']*100:.1f}

- Loyalty Üyelik Oranı:
%{profile['Loyalty_Orani']*100:.1f}

- Ortalama Memnuniyet:
{profile['Ort_Memnuniyet']:.1f}/5

- Ortalama İndirim Kullanımı:
{profile['Ort_Discount']:.1f}

- Ortalama Yaş:
{profile['Ort_Yas']:.0f}

- Ortalama Gelir:
${profile['Ort_Gelir']:.0f}

- Favori Kategori:
{profile['En_Cok_Kategori']}

- Baskın Bölge:
{profile['En_Cok_Bolge']}

- Baskın Kanal:
{profile['En_Cok_Kanal']}

- Baskın Cinsiyet:
{profile['En_Cok_Cinsiyet']}

- Tercih Edilen Cihaz:
{profile['En_Cok_Cihaz']}

ÇIKTI FORMATI:

Persona Adı:
[Kısa persona adı]

Kısa Tanım:
[Bu müşteri grubunu açıkla]

Davranış Profili:
[Satın alma davranışı]

Risk / Fırsat:
[Bu segmentin temel riski veya fırsatı]

Önerilen Kampanya:
[Önerilen kampanya stratejisi]

CRM Aksiyonu:
[Business tarafında alınacak aksiyon]
"""

# =========================================================
# PERSONA GENERATION
# =========================================================

personas = []

print(f"\n{len(segment_profiles)} segment için persona üretiliyor...")
print("Rate limit nedeniyle segmentler arasında 15sn bekleniyor.\n")

for _, row in segment_profiles.iterrows():

    segment = row["RFM_Segment"]

    print(f"--- {segment} ---")

    prompt = build_prompt(segment, row)

    try:

        response = co.chat(
            model="command-a-03-2025",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        persona_text = response.message.content[0].text.strip()

        # -------------------------------------------------
        # CLEAN OUTPUT
        # -------------------------------------------------

        persona_text = (
            persona_text
            .replace("```", "")
            .replace("<", "")
            .replace(">", "")
            .strip()
        )

        print(f"✅ Persona generated ({len(persona_text)} chars)")

        personas.append({

            "RFM_Segment": segment,

            "Musteri_Sayisi":
            int(row["Musteri_Sayisi"]),

            "Satin_Alma_Orani":
            row["Satin_Alma_Orani"],

            "Ort_Monetary":
            row["Ort_Monetary"],

            "Ort_Recency":
            row["Ort_Recency"],

            "Persona":
            persona_text
        })

    except Exception as e:

        print(f"❌ Error: {e}")

        personas.append({

            "RFM_Segment": segment,

            "Musteri_Sayisi":
            int(row["Musteri_Sayisi"]),

            "Satin_Alma_Orani":
            row["Satin_Alma_Orani"],

            "Ort_Monetary":
            row["Ort_Monetary"],

            "Ort_Recency":
            row["Ort_Recency"],

            "Persona":
            f"Hata: {str(e)}"
        })

    # =====================================================
    # RATE LIMIT WAIT
    # =====================================================

    print("15 seconds waiting...")
    time.sleep(15)

# =========================================================
# SAVE CSV
# =========================================================

personas_df = pd.DataFrame(personas)

csv_output_path = f"{OUTPUT_DIR}/segment_personas.csv"

personas_df.to_csv(
    csv_output_path,
    index=False
)

print(f"\n✅ CSV saved: {csv_output_path}")

# =========================================================
# SAVE TXT FILES
# =========================================================

for _, row in personas_df.iterrows():

    segment_name = (
        str(row["RFM_Segment"])
        .replace(" ", "_")
    )

    txt_path = (
        f"{OUTPUT_DIR}/{segment_name}_persona.txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"SEGMENT: {row['RFM_Segment']}\n"
        )

        f.write(
            f"Müşteri Sayısı: {row['Musteri_Sayisi']}\n"
        )

        f.write(
            f"Satın Alma Oranı: %{row['Satin_Alma_Orani']*100:.1f}\n"
        )

        f.write(
            f"Ort. Harcama: ${row['Ort_Monetary']:.0f}\n"
        )

        f.write("=" * 60 + "\n\n")

        f.write(str(row["Persona"]))

print("\n" + "=" * 60)
print("AI PERSONA ENGINE COMPLETED")
print("=" * 60)

print(f"Toplam {len(personas)} persona üretildi.\n")

print("Generated files:")

for f_name in sorted(os.listdir(OUTPUT_DIR)):
    print(f"✅ {OUTPUT_DIR}/{f_name}")