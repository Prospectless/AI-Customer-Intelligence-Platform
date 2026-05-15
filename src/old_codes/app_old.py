import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Customer Analytics & RFM Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# PATHS
# =========================================================

CLEAN_PATH = "data/clean_customer_data.csv"
EDA_DIR = "data/Eda_Output"
RFM_DIR = "data/RFM_Output"


# =========================================================
# HELPERS
# =========================================================

@st.cache_data
def read_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def missing_file(path):
    st.warning(f"Dosya bulunamadı: {path}")


def format_number(x):
    return f"{x:,.0f}"


def format_money(x):
    return f"${x:,.0f}"

def find_col(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def standardize_summary_df(df, category_col=None, value_col="Count"):
    if df is None:
        return None

    df = df.copy()

    # Unnamed index kolonlarını temizle
    unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    df = df.drop(columns=unnamed_cols, errors="ignore")

    if category_col is not None and category_col not in df.columns:
        df = df.rename(columns={df.columns[0]: category_col})

    possible_value_cols = [
        value_col,
        "Count", "count",
        "Musteri_Sayisi", "Müşteri_Sayısı",
        "customer_count", "Customer_Count",
        "Toplam", "Total", "total",
        "n"
    ]

    found_value_col = find_col(df, possible_value_cols)

    if found_value_col is not None and found_value_col != value_col:
        df = df.rename(columns={found_value_col: value_col})

    return df


def wide_to_long_if_needed(df, id_col, name_col, value_col="Musteri_Sayisi"):
    if df is None:
        return None

    df = df.copy()

    unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    df = df.drop(columns=unnamed_cols, errors="ignore")

    # Zaten long format ise dokunma
    if id_col in df.columns and name_col in df.columns and value_col in df.columns:
        return df

    # Ratio ve oran kolonlarını grafikten çıkar
    drop_words = ["ratio", "oran", "rate", "percent"]
    drop_cols = [
        col for col in df.columns
        if any(word in str(col).lower() for word in drop_words)
    ]

    total_cols = ["Toplam", "Total", "total"]
    drop_cols += [col for col in total_cols if col in df.columns]

    value_cols = [
        col for col in df.columns
        if col not in [id_col, name_col] + drop_cols
    ]

    if id_col in df.columns and name_col not in df.columns:
        return df.melt(
            id_vars=id_col,
            value_vars=value_cols,
            var_name=name_col,
            value_name=value_col
        )

    return df

def normalize_count_columns(df, category_col):
    """
    Hazır summary CSV kolon adları farklı olsa bile dashboard uyumlu hale getirir.
    Beklenen çıktı: [category_col, Count]
    """
    if df is None:
        return None

    df = df.copy()

    if category_col not in df.columns:
        df = df.rename(columns={df.columns[0]: category_col})

    possible_count_cols = [
        "Count", "count", "Musteri_Sayisi", "Customer_Count",
        "customer_count", "Total", "total", "n"
    ]

    count_col = None
    for col in possible_count_cols:
        if col in df.columns:
            count_col = col
            break

    if count_col is None:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != category_col]
        if numeric_cols:
            count_col = numeric_cols[0]

    if count_col is not None and count_col != "Count":
        df = df.rename(columns={count_col: "Count"})

    return df


def get_category_summary(precomputed_df, raw_df, column):
    precomputed_df = normalize_count_columns(precomputed_df, column)

    if precomputed_df is not None and column in precomputed_df.columns and "Count" in precomputed_df.columns:
        return precomputed_df[[column, "Count"]]

    data = raw_df[column].value_counts().reset_index()
    data.columns = [column, "Count"]
    return data


def get_target_summary(precomputed_df, raw_df, selected_cat):
    """
    Hazır target summary varsa onu kullanır.
    Yoksa groupby fallback yapar.
    """
    if precomputed_df is not None:
        df = precomputed_df.copy()

        if selected_cat not in df.columns:
            df = df.rename(columns={df.columns[0]: selected_cat})

        rename_map = {}

        for col in df.columns:
            lower = col.lower()

            if lower in ["count", "customer_count", "musteri_sayisi", "müşteri_sayısı"]:
                rename_map[col] = "Musteri_Sayisi"

            if lower in ["purchase_rate", "satin_alma_orani", "satın_alma_oranı", "purchasestatus_mean"]:
                rename_map[col] = "Satin_Alma_Orani"

            if lower in ["avg_totalspent", "average_totalspent", "ortalama_harcama", "totalspent_mean"]:
                rename_map[col] = "Ortalama_Harcama"

        df = df.rename(columns=rename_map)

        required = [selected_cat, "Musteri_Sayisi", "Satin_Alma_Orani", "Ortalama_Harcama"]

        if all(col in df.columns for col in required):
            return df[required]

    return raw_df.groupby(selected_cat).agg(
        Musteri_Sayisi=("CustomerID", "count"),
        Satin_Alma_Orani=("PurchaseStatus", "mean"),
        Ortalama_Harcama=("TotalSpent", "mean")
    ).reset_index()


def get_rfm_segment_distribution(precomputed_df, rfm_df):
    df = normalize_count_columns(precomputed_df, "RFM_Segment")

    if df is not None and "RFM_Segment" in df.columns and "Count" in df.columns:
        return df[["RFM_Segment", "Count"]]

    seg_counts = rfm_df["RFM_Segment"].value_counts().reset_index()
    seg_counts.columns = ["RFM_Segment", "Count"]
    return seg_counts


def get_segment_summary_row(precomputed_df, rfm_df, selected_segment):
    """
    Hazır rfm_segment_summary varsa seçilen segmentin özetini döner.
    Yoksa None döner.
    """
    if precomputed_df is None:
        return None

    df = precomputed_df.copy()

    if "RFM_Segment" not in df.columns:
        df = df.rename(columns={df.columns[0]: "RFM_Segment"})

    selected = df[df["RFM_Segment"] == selected_segment]

    if selected.empty:
        return None

    return selected.iloc[0]


def safe_get(row, possible_cols, default=np.nan):
    if row is None:
        return default

    for col in possible_cols:
        if col in row.index:
            return row[col]

    return default


# =========================================================
# LOAD DATA
# =========================================================

clean_df = read_csv(CLEAN_PATH)
rfm_df = read_csv(f"{RFM_DIR}/customer_rfm_final.csv")
campaign_df = read_csv(f"{RFM_DIR}/campaign_customer_list.csv")

# EDA PRECOMPUTED OUTPUTS
eda_summary = read_csv(f"{EDA_DIR}/eda_summary.csv")
target_summary_general = read_csv(f"{EDA_DIR}/target_summary.csv")
correlation_matrix = read_csv(f"{EDA_DIR}/correlation_matrix.csv")

cat_product = read_csv(f"{EDA_DIR}/cat_summary_ProductCategory.csv")
cat_gender = read_csv(f"{EDA_DIR}/cat_summary_Gender.csv")
cat_region = read_csv(f"{EDA_DIR}/cat_summary_Region.csv")
cat_referral = read_csv(f"{EDA_DIR}/cat_summary_ReferralSource.csv")

outlier_summary = read_csv(f"{EDA_DIR}/outlier_summary.csv")

target_cat_files = {
    "Gender": f"{EDA_DIR}/target_cat_summary_Gender.csv",
    "ProductCategory": f"{EDA_DIR}/target_cat_summary_ProductCategory.csv",
    "PreferredDevice": f"{EDA_DIR}/target_cat_summary_PreferredDevice.csv",
    "Region": f"{EDA_DIR}/target_cat_summary_Region.csv",
    "ReferralSource": f"{EDA_DIR}/target_cat_summary_ReferralSource.csv",
    "CustomerSegment": f"{EDA_DIR}/target_cat_summary_CustomerSegment.csv",
    "LoyaltyProgram": f"{EDA_DIR}/target_cat_summary_LoyaltyProgram.csv",
    "CustomerSatisfaction": f"{EDA_DIR}/target_cat_summary_CustomerSatisfaction.csv"
}

# RFM PRECOMPUTED OUTPUTS
rfm_segment_summary = read_csv(f"{RFM_DIR}/rfm_segment_summary.csv")
rfm_segment_distribution = read_csv(f"{RFM_DIR}/rfm_segment_distribution.csv")
rfm_category_summary = read_csv(f"{RFM_DIR}/rfm_category_summary.csv")
rfm_region_summary = read_csv(f"{RFM_DIR}/rfm_region_summary.csv")
rfm_gender_summary = read_csv(f"{RFM_DIR}/rfm_gender_summary.csv")
rfm_source_summary = read_csv(f"{RFM_DIR}/rfm_source_summary.csv")
rfm_discount_summary = read_csv(f"{RFM_DIR}/rfm_discount_summary.csv")


# =========================================================
# PAGEBAR
# =========================================================

st.markdown(
    """
    <style>
    div[data-testid="stHorizontalBlock"] button {
        height: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Customer Analytics Dashboard")

page = st.radio(
    label="Sayfa Seç",
    options=[
        "1. Overview",
        "2. EDA Dashboard",
        "3. RFM Dashboard",
        "4. Campaign Dashboard",
        "5. Customer Explorer",
        "6. Outlier Playground"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()

# =========================================================
# PAGE 1 - OVERVIEW
# =========================================================

if page == "1. Overview":
    st.title("📌 Customer Analytics Overview")

    if clean_df is None:
        missing_file(CLEAN_PATH)
        st.stop()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Toplam Müşteri", format_number(clean_df["CustomerID"].nunique()))
    col2.metric("Toplam Satır", format_number(clean_df.shape[0]))
    col3.metric("Satın Alma Oranı", f"{clean_df['PurchaseStatus'].mean() * 100:.2f}%")
    col4.metric("Toplam Harcama", format_money(clean_df["TotalSpent"].sum()))

    st.divider()

    col5, col6, col7, col8 = st.columns(4)

    col5.metric("Ortalama Harcama", format_money(clean_df["TotalSpent"].mean()))
    col6.metric("Ortalama Purchase", f"{clean_df['NumberOfPurchases'].mean():.2f}")
    col7.metric("Ortalama Recency", f"{clean_df['LastPurchaseDaysAgo'].mean():.2f} gün")
    col8.metric("Ortalama Discount", f"{clean_df['DiscountsAvailed'].mean():.2f}")

    st.divider()

    st.subheader("📚 Dataset Bilgisi")

    with st.expander("Kaggle Dataset Açıklaması", expanded=True):
        st.markdown("""
        **Dataset:** Customer Purchase Behavior Dataset (E-Commerce)  
        **Kaynak:** Kaggle  
        **Kayıt Sayısı:** 500.000 müşteri davranış kaydı  
        **Kullanım Alanları:**
        - PurchaseStatus tahmini
        - Müşteri segmentasyonu
        - EDA
        - Davranışsal analiz
        - Pazarlama stratejisi geliştirme

        Bu veri seti, bir e-ticaret platformundaki müşteri satın alma davranışlarını analiz etmek için kullanılmaktadır.
        """)

    with st.expander("Kolon Açıklamaları"):
        data_dictionary = pd.DataFrame({
            "Kolon": [
                "Age", "AnnualIncome", "NumberOfPurchases", "TimeSpentOnWebsite",
                "CustomerTenureYears", "LastPurchaseDaysAgo", "Gender",
                "ProductCategory", "PreferredDevice", "Region", "ReferralSource",
                "CustomerSegment", "LoyaltyProgram", "DiscountsAvailed",
                "SessionCount", "CustomerSatisfaction", "PurchaseStatus",
                "TotalSpent", "min_favorite_purchases"
            ],
            "Açıklama": [
                "Müşterinin yaşı",
                "Müşterinin yıllık geliri",
                "Müşterinin toplam satın alma sayısı",
                "Web sitesinde geçirilen ortalama süre",
                "Müşterinin platformdaki üyelik süresi",
                "Son satın almadan bu yana geçen gün sayısı",
                "Müşterinin cinsiyeti",
                "Müşterinin en sık satın aldığı / favori ürün kategorisi",
                "Müşterinin tercih ettiği cihaz",
                "Müşterinin bölgesi",
                "Müşterinin platforma geldiği kaynak",
                "İş tarafından tanımlanan müşteri segmenti",
                "Loyalty programına üyelik durumu",
                "Kullanılan indirim / kupon sayısı",
                "Oturum / ziyaret sayısı",
                "Müşteri memnuniyet skoru",
                "Satın alma hedef değişkeni",
                "Feature engineering ile oluşturulan toplam harcama",
                "Favori kategoriden minimum satın alma tahmini"
            ]
        })

        st.dataframe(data_dictionary, use_container_width=True)

    with st.expander("Proje Kapsamında Yapılan Veri Zenginleştirme"):
        st.markdown("""
        Orijinal Kaggle veri setinde **TotalSpent** değişkeni bulunmadığı için proje kapsamında üretildi.

        **TotalSpent hesaplama mantığı:**

        - ProductCategory bazlı ortalama sepet tutarı
        - CustomerSegment bazlı çarpan
        - DiscountsAvailed bazlı indirim etkisi
        - LoyaltyProgram bazlı bonus çarpan
        - NumberOfPurchases

        kullanılarak müşteri bazlı toplam harcama tahmini oluşturuldu.

        Ayrıca:
        - `CustomerID` eklendi
        - `min_favorite_purchases` feature'ı üretildi
        - negatif değerler temizlendi
        - RFM için final temiz veri hazırlandı
        """)

    st.subheader("📄 Temiz Veri İlk 10 Satır")
    st.dataframe(clean_df.head(10), use_container_width=True)

    st.subheader("📌 Veri Kolonları")
    dtype_df = pd.DataFrame({
        "Column": clean_df.columns,
        "Dtype": clean_df.dtypes.astype(str).values,
        "Null Count": clean_df.isnull().sum().values
    })
    st.dataframe(dtype_df, use_container_width=True)


# =========================================================
# PAGE 2 - EDA DASHBOARD
# =========================================================

elif page == "2. EDA Dashboard":
    st.title("🔎 EDA Dashboard")

    if clean_df is None:
        missing_file(CLEAN_PATH)
        st.stop()

    # -----------------------------------------------------
    # EDA LOCAL HELPERS
    # -----------------------------------------------------

    def clean_summary_columns(df):
        if df is None:
            return None

        df = df.copy()
        unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
        df = df.drop(columns=unnamed_cols, errors="ignore")
        return df

    def prepare_count_df(df, category_col, fallback_df):
        df = clean_summary_columns(df)

        if df is not None:
            if category_col not in df.columns:
                df = df.rename(columns={df.columns[0]: category_col})

            possible_count_cols = [
                "Count", "count", "Musteri_Sayisi", "Müşteri_Sayısı",
                "customer_count", "Customer_Count", "Toplam", "Total", "total", "n"
            ]

            count_col = None
            for col in possible_count_cols:
                if col in df.columns:
                    count_col = col
                    break

            if count_col is None:
                numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
                numeric_cols = [c for c in numeric_cols if c != category_col]
                if numeric_cols:
                    count_col = numeric_cols[0]

            if count_col is not None:
                df = df.rename(columns={count_col: "Count"})
                return df[[category_col, "Count"]]

        fallback = fallback_df[category_col].value_counts().reset_index()
        fallback.columns = [category_col, "Count"]
        return fallback

    def prepare_target_df(df, selected_cat):
        df = clean_summary_columns(df)

        if df is not None:
            if selected_cat not in df.columns:
                df = df.rename(columns={df.columns[0]: selected_cat})

            rename_map = {}

            for col in df.columns:
                lower = str(col).lower()

                if lower in [
                    "count", "customer_count", "musteri_sayisi",
                    "müşteri_sayısı", "toplam", "total", "n"
                ]:
                    rename_map[col] = "Musteri_Sayisi"

                elif lower in [
                    "purchase_rate", "satin_alma_orani", "satın_alma_oranı",
                    "purchasestatus_mean", "purchase_status_mean",
                    "mean_purchase", "target_rate"
                ]:
                    rename_map[col] = "Satin_Alma_Orani"

                elif lower in [
                    "avg_totalspent", "average_totalspent", "ortalama_harcama",
                    "totalspent_mean", "mean_totalspent", "avg_spent"
                ]:
                    rename_map[col] = "Ortalama_Harcama"

            df = df.rename(columns=rename_map)

            required_cols = [
                selected_cat,
                "Musteri_Sayisi",
                "Satin_Alma_Orani",
                "Ortalama_Harcama"
            ]

            if all(col in df.columns for col in required_cols):
                return df[required_cols]

        return clean_df.groupby(selected_cat).agg(
            Musteri_Sayisi=("CustomerID", "count"),
            Satin_Alma_Orani=("PurchaseStatus", "mean"),
            Ortalama_Harcama=("TotalSpent", "mean")
        ).reset_index()

    def prepare_cross_df(df, id_col, name_col):
        df = clean_summary_columns(df)

        if df is not None:
            # Zaten long format ise
            if id_col in df.columns and name_col in df.columns:
                possible_value_cols = [
                    "Musteri_Sayisi", "Müşteri_Sayısı", "Count",
                    "count", "customer_count", "Customer_Count",
                    "Toplam", "Total", "total", "n"
                ]

                value_col = None
                for col in possible_value_cols:
                    if col in df.columns:
                        value_col = col
                        break

                if value_col is not None:
                    df = df.rename(columns={value_col: "Musteri_Sayisi"})
                    return df[[id_col, name_col, "Musteri_Sayisi"]]

            # Wide format ise melt et
            if id_col in df.columns and name_col not in df.columns:
                drop_words = ["ratio", "oran", "rate", "percent"]
                drop_cols = [
                    col for col in df.columns
                    if any(word in str(col).lower() for word in drop_words)
                ]

                drop_cols += [
                    col for col in ["Toplam", "Total", "total"]
                    if col in df.columns
                ]

                value_cols = [
                    col for col in df.columns
                    if col not in [id_col] + drop_cols
                ]

                return df.melt(
                    id_vars=id_col,
                    value_vars=value_cols,
                    var_name=name_col,
                    value_name="Musteri_Sayisi"
                )

        return None

    def prepare_corr_df(df):
        df = clean_summary_columns(df)

        if df is not None:
            first_col = str(df.columns[0]).lower()

            if first_col in ["index", "column", "variable"] or "unnamed" in first_col:
                df = df.set_index(df.columns[0])

            return df

        numeric_df = clean_df.select_dtypes(include=["int64", "float64"])
        return numeric_df.corr()

    # -----------------------------------------------------
    # TABS
    # -----------------------------------------------------

    tab1, tab2, tab3, tab4 = st.tabs([
        "Genel Dağılımlar",
        "Target Analizi",
        "Kategori Analizleri",
        "Korelasyon"
    ])

    # -----------------------------------------------------
    # TAB 1 - GENEL DAĞILIMLAR
    # -----------------------------------------------------

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("ProductCategory Dağılımı")

            data = prepare_count_df(
                cat_product,
                "ProductCategory",
                clean_df
            )

            fig = px.bar(
                data,
                x="ProductCategory",
                y="Count"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Gender Dağılımı")

            data = prepare_count_df(
                cat_gender,
                "Gender",
                clean_df
            )

            fig = px.pie(
                data,
                names="Gender",
                values="Count",
                hole=0.35
            )
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Region Dağılımı")

            data = prepare_count_df(
                cat_region,
                "Region",
                clean_df
            )

            fig = px.bar(
                data,
                x="Region",
                y="Count"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.subheader("ReferralSource Dağılımı")

            data = prepare_count_df(
                cat_referral,
                "ReferralSource",
                clean_df
            )

            fig = px.bar(
                data,
                x="ReferralSource",
                y="Count"
            )
            st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # TAB 2 - TARGET ANALİZİ
    # -----------------------------------------------------

    with tab2:
        st.subheader("Tüm Değişkenlerin Target ile Analizi")

        selected_cat = st.selectbox(
            "Kategorik Değişken Seç",
            [
                "Gender",
                "ProductCategory",
                "PreferredDevice",
                "Region",
                "ReferralSource",
                "CustomerSegment",
                "LoyaltyProgram",
                "CustomerSatisfaction"
            ]
        )

        precomputed_target = read_csv(target_cat_files[selected_cat])

        target_summary = prepare_target_df(
            precomputed_target,
            selected_cat
        )

        st.dataframe(target_summary, use_container_width=True)

        fig = px.bar(
            target_summary,
            x=selected_cat,
            y="Satin_Alma_Orani",
            text="Satin_Alma_Orani",
            title=f"{selected_cat} Bazında Satın Alma Oranı"
        )
        fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # TAB 3 - KATEGORİ ANALİZLERİ
    # -----------------------------------------------------

    with tab3:
        st.subheader("Kategori Bazlı Satış Hikayesi")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Gender × ProductCategory**")

            source_file = read_csv(f"{EDA_DIR}/category_gender_min_sales.csv")

            gender_cat = prepare_cross_df(
                source_file,
                id_col="ProductCategory",
                name_col="Gender"
            )

            if gender_cat is None:
                gender_cat = clean_df.groupby(["Gender", "ProductCategory"]).agg(
                    Musteri_Sayisi=("CustomerID", "count")
                ).reset_index()

            fig = px.bar(
                gender_cat,
                x="ProductCategory",
                y="Musteri_Sayisi",
                color="Gender",
                barmode="group"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Region × ProductCategory**")

            source_file = read_csv(f"{EDA_DIR}/category_region_min_sales.csv")

            region_cat = prepare_cross_df(
                source_file,
                id_col="ProductCategory",
                name_col="Region"
            )

            if region_cat is None:
                region_cat = clean_df.groupby(["Region", "ProductCategory"]).agg(
                    Musteri_Sayisi=("CustomerID", "count")
                ).reset_index()

            fig = px.bar(
                region_cat,
                x="ProductCategory",
                y="Musteri_Sayisi",
                color="Region",
                barmode="group"
            )
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**ReferralSource × ProductCategory**")

            source_file = read_csv(f"{EDA_DIR}/category_referral_min_sales.csv")

            source_cat = prepare_cross_df(
                source_file,
                id_col="ProductCategory",
                name_col="ReferralSource"
            )

            if source_cat is None:
                source_cat = clean_df.groupby(["ReferralSource", "ProductCategory"]).agg(
                    Musteri_Sayisi=("CustomerID", "count")
                ).reset_index()

            fig = px.bar(
                source_cat,
                x="ProductCategory",
                y="Musteri_Sayisi",
                color="ReferralSource",
                barmode="group"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown("**CustomerSegment × ProductCategory**")

            source_file = read_csv(f"{EDA_DIR}/category_segment_min_sales.csv")

            segment_cat = prepare_cross_df(
                source_file,
                id_col="ProductCategory",
                name_col="CustomerSegment"
            )

            if segment_cat is None:
                segment_cat = clean_df.groupby(["CustomerSegment", "ProductCategory"]).agg(
                    Musteri_Sayisi=("CustomerID", "count")
                ).reset_index()

            fig = px.bar(
                segment_cat,
                x="ProductCategory",
                y="Musteri_Sayisi",
                color="CustomerSegment",
                barmode="group"
            )
            st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # TAB 4 - KORELASYON
    # -----------------------------------------------------

    with tab4:
        st.subheader("Korelasyon Matrisi")

        corr = prepare_corr_df(correlation_matrix)

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# PAGE 3 - RFM DASHBOARD
# =========================================================

elif page == "3. RFM Dashboard":
    st.title("🧩 RFM Dashboard")

    if rfm_df is None:
        missing_file(f"{RFM_DIR}/customer_rfm_final.csv")
        st.stop()

    segment_list = sorted(rfm_df["RFM_Segment"].dropna().unique())

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.metric(
            "Toplam Müşteri",
            format_number(rfm_df["CustomerID"].nunique())
        )

    with col2:
        st.metric(
            "Toplam Segment",
            rfm_df["RFM_Segment"].nunique()
        )

    with col3:
        selected_segment = st.selectbox(
            "Segment Seçin",
            segment_list
        )

    filtered = rfm_df[
        rfm_df["RFM_Segment"] == selected_segment
    ]

    selected_summary = get_segment_summary_row(
        rfm_segment_summary,
        rfm_df,
        selected_segment
    )

    st.divider()

    left, right = st.columns([1, 2])

    # =====================================================
    # LEFT METRICS
    # =====================================================

    with left:
        st.subheader(f"📊 {selected_segment.upper()}")

        if selected_summary is not None:

            customer_count = safe_get(
                selected_summary,
                ["customer_count", "Customer_Count", "Musteri_Sayisi", "Count"],
                len(filtered)
            )

            avg_recency = safe_get(
                selected_summary,
                ["avg_recency", "recency_mean", "Ortalama_Recency", "recency"],
                filtered["recency"].mean()
            )

            avg_frequency = safe_get(
                selected_summary,
                ["avg_frequency", "frequency_mean", "Ortalama_Frequency", "frequency"],
                filtered["frequency"].mean()
            )

            avg_monetary = safe_get(
                selected_summary,
                ["avg_monetary", "monetary_mean", "Ortalama_Monetary", "monetary"],
                filtered["monetary"].mean()
            )

            purchase_rate = safe_get(
                selected_summary,
                ["purchase_rate", "Purchase_Rate", "Satin_Alma_Orani", "PurchaseStatus"],
                filtered["PurchaseStatus"].mean()
            )

        else:
            customer_count = len(filtered)
            avg_recency = filtered["recency"].mean()
            avg_frequency = filtered["frequency"].mean()
            avg_monetary = filtered["monetary"].mean()
            purchase_rate = filtered["PurchaseStatus"].mean()

        st.metric(
            "Müşteri Sayısı",
            format_number(customer_count)
        )

        st.metric(
            "Ort. Recency (Ham Gün)",
            f"{avg_recency:.2f} gün"
        )

        st.metric(
            "Ort. Frequency (Ham Purchase)",
            f"{avg_frequency:.2f}"
        )

        st.metric(
            "Ort. Monetary (Ham Harcama)",
            format_money(avg_monetary)
        )

        st.metric(
            "Satın Alma Oranı",
            f"{purchase_rate * 100:.2f}%"
        )

    # =====================================================
    # RIGHT PIE
    # =====================================================

    with right:

        seg_counts = get_rfm_segment_distribution(
            rfm_segment_distribution,
            rfm_df
        )

        fig = px.pie(
            seg_counts,
            values="Count",
            names="RFM_Segment",
            title="RFM Segment Distribution",
            hole=0.35
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.info(
        "Raw Recency / Frequency / Monetary sekmeleri gerçek ham müşteri davranış "
        "değerlerini gösterir. "
        "RFM Score Comparison ve Heatmap sekmeleri ise 1-5 arası oluşturulan "
        "RFM skorlarını gösterir. "
        "Ham recency düşük olduğunda müşteri daha iyidir. "
        "recency_score yüksek olduğunda müşteri daha iyidir."
    )

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Segment Count",
        "Raw Recency",
        "Raw Frequency",
        "Raw Monetary",
        "RFM Score Comparison",
        "RFM Score Heatmap"
    ])

    # =====================================================
    # TAB 1 - SEGMENT COUNT
    # =====================================================

    with tab1:

        st.subheader("Segment Bazlı Müşteri Sayısı")

        counts = get_rfm_segment_distribution(
            rfm_segment_distribution,
            rfm_df
        )

        counts["Highlight"] = np.where(
            counts["RFM_Segment"] == selected_segment,
            "Seçili Segment",
            "Diğer"
        )

        fig = px.bar(
            counts,
            y="RFM_Segment",
            x="Count",
            color="Highlight",
            orientation="h",
            title=f"{selected_segment} Segmentinin Genel İçindeki Yeri"
        )

        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # TAB 2 - RAW RECENCY
    # =====================================================

    with tab2:

        col_l, col_r = st.columns(2)

        with col_l:

            st.markdown("### Genel - Tüm Müşteriler")

            fig = px.histogram(
                rfm_df,
                x="recency",
                nbins=40,
                title="Raw Recency Distribution - All Customers"
            )

            st.plotly_chart(fig, use_container_width=True)

        with col_r:

            st.markdown(f"### {selected_segment.upper()}")

            fig = px.histogram(
                filtered,
                x="recency",
                nbins=40,
                title=f"Raw Recency Distribution - {selected_segment}"
            )

            st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # TAB 3 - RAW FREQUENCY
    # =====================================================

    with tab3:

        col_l, col_r = st.columns(2)

        with col_l:

            st.markdown("### Genel - Tüm Müşteriler")

            fig = px.histogram(
                rfm_df,
                x="frequency",
                nbins=40,
                title="Raw Frequency Distribution - All Customers"
            )

            st.plotly_chart(fig, use_container_width=True)

        with col_r:

            st.markdown(f"### {selected_segment.upper()}")

            fig = px.histogram(
                filtered,
                x="frequency",
                nbins=40,
                title=f"Raw Frequency Distribution - {selected_segment}"
            )

            st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # TAB 4 - RAW MONETARY
    # =====================================================

    with tab4:

        col_l, col_r = st.columns(2)

        with col_l:

            st.markdown("### Genel - Tüm Müşteriler")

            fig = px.histogram(
                rfm_df,
                x="monetary",
                nbins=40,
                title="Raw Monetary Distribution - All Customers"
            )

            st.plotly_chart(fig, use_container_width=True)

        with col_r:

            st.markdown(f"### {selected_segment.upper()}")

            fig = px.histogram(
                filtered,
                x="monetary",
                nbins=40,
                title=f"Raw Monetary Distribution - {selected_segment}"
            )

            st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # TAB 5 - RFM SCORE COMPARISON
    # =====================================================

    with tab5:

        st.subheader("Segment Bazlı Ortalama RFM Score Karşılaştırması")

        score_cols = [
            "recency_score",
            "frequency_score",
            "monetary_score"
        ]

        score_avg = rfm_df.groupby("RFM_Segment")[
            score_cols
        ].mean().round(2).reset_index()

        fig = px.bar(
            score_avg,
            x="RFM_Segment",
            y=score_cols,
            barmode="group",
            title="Average RFM Scores by Segment"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader(
            f"{selected_segment} Segmenti Ortalama RFM Score Profili"
        )

        selected_score_avg = filtered[
            score_cols
        ].mean().round(2).reset_index()

        selected_score_avg.columns = [
            "Metric",
            "Score"
        ]

        fig = px.bar(
            selected_score_avg,
            x="Metric",
            y="Score",
            text="Score",
            title=f"{selected_segment} Ortalama R/F/M Score Profili",
            range_y=[0, 5]
        )

        fig.update_traces(textposition="outside")

        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # TAB 6 - RFM SCORE HEATMAP
    # =====================================================

    with tab6:

        st.subheader("Segment Bazlı Ortalama RFM Score Heatmap")

        score_cols = [
            "recency_score",
            "frequency_score",
            "monetary_score"
        ]

        if (
            rfm_segment_summary is not None
            and all(col in rfm_segment_summary.columns for col in score_cols)
        ):

            heatmap_data = rfm_segment_summary.copy()

            if "RFM_Segment" not in heatmap_data.columns:
                heatmap_data = heatmap_data.rename(
                    columns={heatmap_data.columns[0]: "RFM_Segment"}
                )

            heatmap_data = heatmap_data.set_index(
                "RFM_Segment"
            )[score_cols].round(2)

        else:

            heatmap_data = rfm_df.groupby(
                "RFM_Segment"
            )[score_cols].mean().round(2)

        fig = px.imshow(
            heatmap_data,
            text_auto=True,
            aspect="auto",
            title="Average RFM Scores Heatmap"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader(
            f"{selected_segment} RF Score Grid → Ortalama Monetary Score"
        )

        pivot = filtered.groupby(
            ["recency_score", "frequency_score"]
        )["monetary_score"].mean().reset_index()

        pivot_table = pivot.pivot(
            index="recency_score",
            columns="frequency_score",
            values="monetary_score"
        )

        fig = px.imshow(
            pivot_table,
            text_auto=".2f",
            aspect="auto",
            title=f"{selected_segment} RF Score Heatmap"
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# PAGE 4 - CAMPAIGN DASHBOARD
# =========================================================

elif page == "4. Campaign Dashboard":
    st.title("🎯 Campaign Dashboard")

    if campaign_df is None:
        missing_file(f"{RFM_DIR}/campaign_customer_list.csv")
        st.stop()

    discount_df = read_csv(f"{RFM_DIR}/discount_campaign_customer_list.csv")
    high_priority_df = read_csv(f"{RFM_DIR}/high_priority_campaign_customer_list.csv")
    discount_summary = read_csv(f"{RFM_DIR}/rfm_discount_summary.csv")

    col1, col2, col3 = st.columns(3)

    col1.metric("Tüm Kampanya Müşteri Sayısı", format_number(campaign_df.shape[0]))
    col2.metric("High Priority Müşteri", format_number(0 if high_priority_df is None else high_priority_df.shape[0]))
    col3.metric("Discount Campaign Müşteri", format_number(0 if discount_df is None else discount_df.shape[0]))

    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "Discount Hikayesi",
        "Kampanya Stratejileri",
        "N8N Export"
    ])

    with tab1:
        st.subheader("RFM Segmentine Göre Discount Kullanımı")

        if discount_summary is not None:
            st.dataframe(discount_summary, use_container_width=True)

            ds = discount_summary.copy()

            if "RFM_Segment" not in ds.columns:
                ds = ds.rename(columns={ds.columns[0]: "RFM_Segment"})

            if "Ortalama_Discount" not in ds.columns:
                possible_cols = ["avg_discount", "discount_mean", "DiscountsAvailed", "mean_discount"]
                for col in possible_cols:
                    if col in ds.columns:
                        ds = ds.rename(columns={col: "Ortalama_Discount"})
                        break

            if "Ortalama_Discount" in ds.columns:
                fig = px.bar(
                    ds,
                    x="RFM_Segment",
                    y="Ortalama_Discount",
                    title="Segment Bazlı Ortalama Discount"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Discount summary içinde Ortalama_Discount kolonu bulunamadı.")
        else:
            missing_file(f"{RFM_DIR}/rfm_discount_summary.csv")

    with tab2:
        st.subheader("Kampanya Stratejisi Listesi")

        segment_filter = st.multiselect(
            "Segment Seç",
            sorted(campaign_df["RFM_Segment"].dropna().unique()),
            default=sorted(campaign_df["RFM_Segment"].dropna().unique())[:5]
        )

        filtered_campaign = campaign_df[campaign_df["RFM_Segment"].isin(segment_filter)]

        st.dataframe(filtered_campaign.head(1000), use_container_width=True)

    with tab3:
        st.subheader("N8N İçin Customer Export")

        st.write("Bu tablo otomasyon sistemine gönderilecek kampanya müşteri listesidir.")

        st.dataframe(campaign_df.head(1000), use_container_width=True)

        csv = campaign_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Campaign Customer List İndir",
            data=csv,
            file_name="campaign_customer_list.csv",
            mime="text/csv"
        )


# =========================================================
# PAGE 5 - CUSTOMER EXPLORER
# =========================================================

elif page == "5. Customer Explorer":
    st.title("👤 Customer Explorer")

    if rfm_df is None:
        missing_file(f"{RFM_DIR}/customer_rfm_final.csv")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_segments = st.multiselect(
            "RFM Segment",
            sorted(rfm_df["RFM_Segment"].dropna().unique()),
            default=sorted(rfm_df["RFM_Segment"].dropna().unique())
        )

    with col2:
        selected_regions = st.multiselect(
            "Region",
            sorted(rfm_df["Region"].dropna().unique()),
            default=sorted(rfm_df["Region"].dropna().unique())
        )

    with col3:
        selected_categories = st.multiselect(
            "ProductCategory",
            sorted(rfm_df["ProductCategory"].dropna().unique()),
            default=sorted(rfm_df["ProductCategory"].dropna().unique())
        )

    selected_gender = st.multiselect(
        "Gender",
        sorted(rfm_df["Gender"].dropna().unique()),
        default=sorted(rfm_df["Gender"].dropna().unique())
    )

    filtered = rfm_df[
        (rfm_df["RFM_Segment"].isin(selected_segments)) &
        (rfm_df["Region"].isin(selected_regions)) &
        (rfm_df["ProductCategory"].isin(selected_categories)) &
        (rfm_df["Gender"].isin(selected_gender))
    ]

    st.metric("Filtrelenmiş Müşteri Sayısı", format_number(filtered.shape[0]))

    st.dataframe(filtered.head(2000), use_container_width=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Filtrelenmiş Customer List İndir",
        data=csv,
        file_name="customer_explorer_filtered.csv",
        mime="text/csv"
    )


# =========================================================
# PAGE 6 - OUTLIER PLAYGROUND
# =========================================================

elif page == "6. Outlier Playground":
    st.title("📈 Outlier Playground")

    if clean_df is None:
        missing_file(CLEAN_PATH)
        st.stop()

    numeric_cols = clean_df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    selected_num = st.selectbox(
        "Sayısal değişken seç",
        numeric_cols,
        index=numeric_cols.index("TotalSpent") if "TotalSpent" in numeric_cols else 0
    )

    min_val = float(clean_df[selected_num].min())
    max_val = float(clean_df[selected_num].max())

    if outlier_summary is not None and selected_num in outlier_summary.astype(str).values:
        outlier_row = outlier_summary[
            outlier_summary.astype(str).apply(lambda row: selected_num in row.values, axis=1)
        ]

        if not outlier_row.empty:
            outlier_row = outlier_row.iloc[0]

            q1 = safe_get(outlier_row, ["q1", "Q1", "quantile_25"], clean_df[selected_num].quantile(0.25))
            q3 = safe_get(outlier_row, ["q3", "Q3", "quantile_75"], clean_df[selected_num].quantile(0.75))
            lower_limit = safe_get(outlier_row, ["lower_limit", "Lower_Limit", "IQR_Alt_Sınır"], q1 - 1.5 * (q3 - q1))
            upper_limit = safe_get(outlier_row, ["upper_limit", "Upper_Limit", "IQR_Üst_Sınır"], q3 + 1.5 * (q3 - q1))
        else:
            q1 = clean_df[selected_num].quantile(0.25)
            q3 = clean_df[selected_num].quantile(0.75)
            iqr = q3 - q1
            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr
    else:
        q1 = clean_df[selected_num].quantile(0.25)
        q3 = clean_df[selected_num].quantile(0.75)
        iqr = q3 - q1
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Min", f"{min_val:.2f}")
    col2.metric("Max", f"{max_val:.2f}")
    col3.metric("IQR Alt Sınır", f"{lower_limit:.2f}")
    col4.metric("IQR Üst Sınır", f"{upper_limit:.2f}")

    selected_range = st.slider(
        "Filtre aralığı seç",
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val)
    )

    filtered_outlier = clean_df[
        (clean_df[selected_num] >= selected_range[0]) &
        (clean_df[selected_num] <= selected_range[1])
    ]

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Orijinal Dağılım")
        fig = px.histogram(clean_df, x=selected_num, nbins=50)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Filtrelenmiş Dağılım")
        fig = px.histogram(filtered_outlier, x=selected_num, nbins=50)
        st.plotly_chart(fig, use_container_width=True)

    st.metric("Orijinal Satır Sayısı", format_number(clean_df.shape[0]))
    st.metric("Filtrelenmiş Satır Sayısı", format_number(filtered_outlier.shape[0]))

    if st.button("UYGULA"):
        st.success("Filtre uygulandı.")
        st.dataframe(filtered_outlier.head(2000), use_container_width=True)

        csv = filtered_outlier.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Filtrelenmiş Veriyi İndir",
            data=csv,
            file_name=f"filtered_{selected_num}.csv",
            mime="text/csv"
        )