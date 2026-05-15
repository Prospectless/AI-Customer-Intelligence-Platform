import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

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
KMEANS_DIR = "data/KMeans_Output"
HC_DIR = "data/HC_Output"
CHURN_DIR = "data/Churn_Output"
PERSONA_DIR = "data/Persona_Output"

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

def normalize_count_columns(df, category_col):
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

def get_rfm_segment_distribution(precomputed_df, rfm_df):
    df = normalize_count_columns(precomputed_df, "RFM_Segment")
    if df is not None and "RFM_Segment" in df.columns and "Count" in df.columns:
        return df[["RFM_Segment", "Count"]]
    seg_counts = rfm_df["RFM_Segment"].value_counts().reset_index()
    seg_counts.columns = ["RFM_Segment", "Count"]
    return seg_counts

def get_segment_summary_row(precomputed_df, rfm_df, selected_segment):
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
            if lower in ["count", "customer_count", "musteri_sayisi", "müşteri_sayısı", "toplam", "total", "n"]:
                rename_map[col] = "Musteri_Sayisi"
            elif lower in ["purchase_rate", "satin_alma_orani", "satın_alma_oranı", "purchasestatus_mean", "purchase_status_mean", "mean_purchase", "target_rate"]:
                rename_map[col] = "Satin_Alma_Orani"
            elif lower in ["avg_totalspent", "average_totalspent", "ortalama_harcama", "totalspent_mean", "mean_totalspent", "avg_spent"]:
                rename_map[col] = "Ortalama_Harcama"
        df = df.rename(columns=rename_map)
        required_cols = [selected_cat, "Musteri_Sayisi", "Satin_Alma_Orani", "Ortalama_Harcama"]
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
        if id_col in df.columns and name_col in df.columns:
            possible_value_cols = ["Musteri_Sayisi", "Müşteri_Sayısı", "Count", "count", "customer_count", "Customer_Count", "Toplam", "Total", "total", "n"]
            value_col = None
            for col in possible_value_cols:
                if col in df.columns:
                    value_col = col
                    break
            if value_col is not None:
                df = df.rename(columns={value_col: "Musteri_Sayisi"})
                return df[[id_col, name_col, "Musteri_Sayisi"]]
        if id_col in df.columns and name_col not in df.columns:
            drop_words = ["ratio", "oran", "rate", "percent"]
            drop_cols = [col for col in df.columns if any(word in str(col).lower() for word in drop_words)]
            drop_cols += [col for col in ["Toplam", "Total", "total"] if col in df.columns]
            value_cols = [col for col in df.columns if col not in [id_col] + drop_cols]
            return df.melt(id_vars=id_col, value_vars=value_cols, var_name=name_col, value_name="Musteri_Sayisi")
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

# =========================================================
# LOAD DATA
# =========================================================

clean_df = read_csv(CLEAN_PATH)
rfm_df = read_csv(f"{RFM_DIR}/customer_rfm_final.csv")
campaign_df = read_csv(f"{RFM_DIR}/campaign_customer_list.csv")

# EDA
eda_summary = read_csv(f"{EDA_DIR}/eda_summary.csv")
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

# RFM
rfm_segment_summary = read_csv(f"{RFM_DIR}/rfm_segment_summary.csv")
rfm_segment_distribution = read_csv(f"{RFM_DIR}/rfm_segment_distribution.csv")
rfm_discount_summary = read_csv(f"{RFM_DIR}/rfm_discount_summary.csv")

# KMeans
kmeans_df = read_csv(f"{KMEANS_DIR}/customer_kmeans_rfm_final.csv")
kmeans_profile = read_csv(f"{KMEANS_DIR}/kmeans_cluster_profile_rfm.csv")
rfm_kmeans_cross = read_csv(f"{KMEANS_DIR}/rfm_kmeans_cross.csv")
elbow_results = read_csv(f"{KMEANS_DIR}/elbow_results_rfm.csv")

# HC
hc_profile = read_csv(f"{HC_DIR}/hc_cluster_profile.csv")
kmeans_hc_cross = read_csv(f"{HC_DIR}/kmeans_hc_cross.csv")
rfm_hc_cross = read_csv(f"{HC_DIR}/rfm_hc_cross.csv")

# Churn
model_comparison = read_csv(f"{CHURN_DIR}/model_comparison.csv")
feature_importance = read_csv(f"{CHURN_DIR}/feature_importance.csv")
segment_churn = read_csv(f"{CHURN_DIR}/segment_churn_analysis.csv")
churn_predictions = read_csv(f"{CHURN_DIR}/churn_predictions.csv")

# Persona
personas_df = read_csv(f"{PERSONA_DIR}/segment_personas.csv")

# =========================================================
# NAVIGATION
# =========================================================

st.title("📊 Customer Analytics Dashboard")

page = st.radio(
    label="Sayfa Seç",
    options=[
        "1. Overview",
        "2. EDA Dashboard",
        "3. RFM Dashboard",
        "4. Clustering",
        "5. Churn Prediction",
        "6. Persona Engine",
        "7. Customer Intelligence"
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
        **Kullanım Alanları:** PurchaseStatus tahmini, Müşteri segmentasyonu, EDA, Davranışsal analiz
        """)

    with st.expander("Kolon Açıklamaları"):
        data_dictionary = pd.DataFrame({
            "Kolon": ["Age", "AnnualIncome", "NumberOfPurchases", "TimeSpentOnWebsite",
                      "CustomerTenureYears", "LastPurchaseDaysAgo", "Gender", "ProductCategory",
                      "PreferredDevice", "Region", "ReferralSource", "CustomerSegment",
                      "LoyaltyProgram", "DiscountsAvailed", "SessionCount", "CustomerSatisfaction",
                      "PurchaseStatus", "TotalSpent", "min_favorite_purchases"],
            "Açıklama": ["Müşterinin yaşı", "Müşterinin yıllık geliri", "Toplam satın alma sayısı",
                         "Web sitesinde geçirilen süre", "Platformdaki üyelik süresi",
                         "Son satın almadan bu yana geçen gün", "Cinsiyet", "Favori ürün kategorisi",
                         "Tercih edilen cihaz", "Bölge", "Platforma gelinen kaynak",
                         "İş tarafından tanımlanan segment", "Loyalty programı üyeliği",
                         "Kullanılan indirim sayısı", "Oturum sayısı", "Memnuniyet skoru",
                         "Satın alma hedef değişkeni", "Feature engineering ile üretilen harcama",
                         "Favori kategori minimum satın alma tahmini"]
        })
        st.dataframe(data_dictionary, use_container_width=True)

    with st.expander("Proje Pipeline"):
        st.markdown("""
        **1. Raw Data** → customerData_500k.csv  
        **2. Feature Enrichment** → TotalSpent üretildi  
        **3. EDA** → Kapsamlı veri analizi  
        **4. Data Cleaning** → 491.705 temiz müşteri  
        **5. RFM Analysis** → 10 segment  
        **6. K-Means Clustering** → 6 cluster (Silhouette: 0.38)  
        **7. Hierarchical Clustering** → 7 cluster, 50K örneklem validasyonu  
        **8. Churn Prediction** → XGBoost AUC: 0.98  
        **9. LLM Persona Engine** → 10 segment personası  
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

    tab1, tab2, tab3, tab4 = st.tabs([
        "Genel Dağılımlar",
        "Target Analizi",
        "Kategori Analizleri",
        "Korelasyon"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ProductCategory Dağılımı")
            data = prepare_count_df(cat_product, "ProductCategory", clean_df)
            fig = px.bar(data, x="ProductCategory", y="Count")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Gender Dağılımı")
            data = prepare_count_df(cat_gender, "Gender", clean_df)
            fig = px.pie(data, names="Gender", values="Count", hole=0.35)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Region Dağılımı")
            data = prepare_count_df(cat_region, "Region", clean_df)
            fig = px.bar(data, x="Region", y="Count")
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            st.subheader("ReferralSource Dağılımı")
            data = prepare_count_df(cat_referral, "ReferralSource", clean_df)
            fig = px.bar(data, x="ReferralSource", y="Count")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Tüm Değişkenlerin Target ile Analizi")
        selected_cat = st.selectbox("Kategorik Değişken Seç", list(target_cat_files.keys()))
        precomputed_target = read_csv(target_cat_files[selected_cat])
        target_summary = prepare_target_df(precomputed_target, selected_cat)
        st.dataframe(target_summary, use_container_width=True)
        fig = px.bar(target_summary, x=selected_cat, y="Satin_Alma_Orani",
                     text="Satin_Alma_Orani", title=f"{selected_cat} Bazında Satın Alma Oranı")
        fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Kategori Bazlı Satış Hikayesi")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Gender × ProductCategory**")
            source_file = read_csv(f"{EDA_DIR}/category_gender_min_sales.csv")
            gender_cat = prepare_cross_df(source_file, id_col="ProductCategory", name_col="Gender")
            if gender_cat is None:
                gender_cat = clean_df.groupby(["Gender", "ProductCategory"]).agg(Musteri_Sayisi=("CustomerID", "count")).reset_index()
            fig = px.bar(gender_cat, x="ProductCategory", y="Musteri_Sayisi", color="Gender", barmode="group")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**Region × ProductCategory**")
            source_file = read_csv(f"{EDA_DIR}/category_region_min_sales.csv")
            region_cat = prepare_cross_df(source_file, id_col="ProductCategory", name_col="Region")
            if region_cat is None:
                region_cat = clean_df.groupby(["Region", "ProductCategory"]).agg(Musteri_Sayisi=("CustomerID", "count")).reset_index()
            fig = px.bar(region_cat, x="ProductCategory", y="Musteri_Sayisi", color="Region", barmode="group")
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**ReferralSource × ProductCategory**")
            source_file = read_csv(f"{EDA_DIR}/category_referral_min_sales.csv")
            source_cat = prepare_cross_df(source_file, id_col="ProductCategory", name_col="ReferralSource")
            if source_cat is None:
                source_cat = clean_df.groupby(["ReferralSource", "ProductCategory"]).agg(Musteri_Sayisi=("CustomerID", "count")).reset_index()
            fig = px.bar(source_cat, x="ProductCategory", y="Musteri_Sayisi", color="ReferralSource", barmode="group")
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            st.markdown("**CustomerSegment × ProductCategory**")
            source_file = read_csv(f"{EDA_DIR}/category_segment_min_sales.csv")
            segment_cat = prepare_cross_df(source_file, id_col="ProductCategory", name_col="CustomerSegment")
            if segment_cat is None:
                segment_cat = clean_df.groupby(["CustomerSegment", "ProductCategory"]).agg(Musteri_Sayisi=("CustomerID", "count")).reset_index()
            fig = px.bar(segment_cat, x="ProductCategory", y="Musteri_Sayisi", color="CustomerSegment", barmode="group")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Korelasyon Matrisi")
        corr = prepare_corr_df(correlation_matrix)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation Heatmap")
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
        st.metric("Toplam Müşteri", format_number(rfm_df["CustomerID"].nunique()))
    with col2:
        st.metric("Toplam Segment", rfm_df["RFM_Segment"].nunique())
    with col3:
        selected_segment = st.selectbox("Segment Seçin", segment_list)

    filtered = rfm_df[rfm_df["RFM_Segment"] == selected_segment]
    selected_summary = get_segment_summary_row(rfm_segment_summary, rfm_df, selected_segment)

    st.divider()

    left, right = st.columns([1, 2])

    with left:
        st.subheader(f"📊 {selected_segment.upper()}")
        customer_count = len(filtered)
        avg_recency = filtered["recency"].mean()
        avg_frequency = filtered["frequency"].mean()
        avg_monetary = filtered["monetary"].mean()
        purchase_rate = filtered["PurchaseStatus"].mean()

        st.metric("Müşteri Sayısı", format_number(customer_count))
        st.metric("Ort. Recency", f"{avg_recency:.2f} gün")
        st.metric("Ort. Frequency", f"{avg_frequency:.2f}")
        st.metric("Ort. Monetary", format_money(avg_monetary))
        st.metric("Satın Alma Oranı", f"{purchase_rate * 100:.2f}%")

    with right:
        seg_counts = get_rfm_segment_distribution(rfm_segment_distribution, rfm_df)
        fig = px.pie(seg_counts, values="Count", names="RFM_Segment",
                     title="RFM Segment Distribution", hole=0.35)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.info("Raw sekmeler ham davranış değerlerini, Score sekmeleri 1-5 arası skorları gösterir. Ham recency düşük = iyi, recency_score yüksek = iyi.")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Segment Count", "Raw Recency", "Raw Frequency",
        "Raw Monetary", "RFM Score Comparison", "RFM Score Heatmap"
    ])

    with tab1:
        counts = get_rfm_segment_distribution(rfm_segment_distribution, rfm_df)
        counts["Highlight"] = np.where(counts["RFM_Segment"] == selected_segment, "Seçili Segment", "Diğer")
        fig = px.bar(counts, y="RFM_Segment", x="Count", color="Highlight",
                     orientation="h", title=f"{selected_segment} Segmentinin Genel İçindeki Yeri")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("### Genel")
            fig = px.histogram(rfm_df, x="recency", nbins=40, title="Raw Recency - All")
            st.plotly_chart(fig, use_container_width=True)
        with col_r:
            st.markdown(f"### {selected_segment.upper()}")
            fig = px.histogram(filtered, x="recency", nbins=40, title=f"Raw Recency - {selected_segment}")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("### Genel")
            fig = px.histogram(rfm_df, x="frequency", nbins=40, title="Raw Frequency - All")
            st.plotly_chart(fig, use_container_width=True)
        with col_r:
            st.markdown(f"### {selected_segment.upper()}")
            fig = px.histogram(filtered, x="frequency", nbins=40, title=f"Raw Frequency - {selected_segment}")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("### Genel")
            fig = px.histogram(rfm_df, x="monetary", nbins=40, title="Raw Monetary - All")
            st.plotly_chart(fig, use_container_width=True)
        with col_r:
            st.markdown(f"### {selected_segment.upper()}")
            fig = px.histogram(filtered, x="monetary", nbins=40, title=f"Raw Monetary - {selected_segment}")
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        score_cols = ["recency_score", "frequency_score", "monetary_score"]
        score_avg = rfm_df.groupby("RFM_Segment")[score_cols].mean().round(2).reset_index()
        fig = px.bar(score_avg, x="RFM_Segment", y=score_cols, barmode="group",
                     title="Average RFM Scores by Segment")
        st.plotly_chart(fig, use_container_width=True)

        selected_score_avg = filtered[score_cols].mean().round(2).reset_index()
        selected_score_avg.columns = ["Metric", "Score"]
        fig = px.bar(selected_score_avg, x="Metric", y="Score", text="Score",
                     title=f"{selected_segment} R/F/M Score Profili", range_y=[0, 5])
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with tab6:
        score_cols = ["recency_score", "frequency_score", "monetary_score"]
        heatmap_data = rfm_df.groupby("RFM_Segment")[score_cols].mean().round(2)
        fig = px.imshow(heatmap_data, text_auto=True, aspect="auto",
                        title="Average RFM Scores Heatmap")
        st.plotly_chart(fig, use_container_width=True)

        pivot = filtered.groupby(["recency_score", "frequency_score"])["monetary_score"].mean().reset_index()
        pivot_table = pivot.pivot(index="recency_score", columns="frequency_score", values="monetary_score")
        fig = px.imshow(pivot_table, text_auto=".2f", aspect="auto",
                        title=f"{selected_segment} RF Score Heatmap")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# PAGE 4 - CLUSTERING
# =========================================================

elif page == "4. Clustering":
    st.title("🔬 Clustering Analizi")

    tab1, tab2, tab3 = st.tabs([
        "K-Means",
        "Hierarchical Clustering",
        "Yöntem Karşılaştırması"
    ])

    with tab1:
        st.subheader("K-Means Clustering (RFM Only)")

        col1, col2, col3 = st.columns(3)
        col1.metric("Optimal K", "6")
        col2.metric("Silhouette Score", "0.3794")
        col3.metric("Kullanılan Özellik", "Recency, Frequency, Monetary")

        st.divider()

        if elbow_results is not None:
            col_l, col_r = st.columns(2)
            with col_l:
                fig = px.line(elbow_results, x="K", y="Inertia", markers=True,
                              title="Elbow Method")
                st.plotly_chart(fig, use_container_width=True)
            with col_r:
                fig = px.line(elbow_results, x="K", y="Silhouette_Score", markers=True,
                              title="Silhouette Scores")
                st.plotly_chart(fig, use_container_width=True)

        if kmeans_profile is not None:
            st.subheader("Cluster Profilleri")
            profile = kmeans_profile.copy()
            if "RFM_Segment" in profile.columns:
                profile = profile.drop(columns=["RFM_Segment"], errors="ignore")
            st.dataframe(profile, use_container_width=True)

            if "Cluster_Adi" in profile.columns and "Ort_Recency" in profile.columns:
                fig = px.scatter(
                    profile,
                    x="Ort_Recency",
                    y="Ort_Monetary",
                    size="Musteri_Sayisi",
                    color="Cluster_Adi",
                    hover_name="Cluster_Adi",
                    title="Cluster Profili: Recency vs Monetary"
                )
                st.plotly_chart(fig, use_container_width=True)

        if rfm_kmeans_cross is not None:
            st.subheader("RFM Segment × K-Means Cluster Karşılaştırması")
            fig = px.bar(
                rfm_kmeans_cross.sort_values("Musteri_Sayisi", ascending=False).head(20),
                x="Musteri_Sayisi",
                y="RFM_Segment",
                color="KMeans_Cluster_Adi",
                orientation="h",
                title="RFM Segmentlerinin K-Means Clusterlarına Dağılımı"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Hierarchical Clustering (50K Örneklem, Ward Linkage)")

        col1, col2, col3 = st.columns(3)
        col1.metric("Optimal K", "7")
        col2.metric("Silhouette Score", "0.3233")
        col3.metric("Örneklem", "50.000")

        st.info("500K veri için HC hesaplama maliyeti çok yüksek. 50K temsili örneklem üzerinde çalıştırıldı.")

        st.divider()

        dendrogram_path = f"{HC_DIR}/dendrogram.png"
        hc_sil_path = f"{HC_DIR}/hc_silhouette.png"

        col_l, col_r = st.columns(2)
        with col_l:
            if os.path.exists(dendrogram_path):
                st.image(dendrogram_path, caption="Dendrogram (Ward, 5K Örneklem)", use_column_width=True)
        with col_r:
            if os.path.exists(hc_sil_path):
                st.image(hc_sil_path, caption="HC Silhouette Scores", use_column_width=True)

        if hc_profile is not None:
            st.subheader("HC Cluster Profilleri")
            profile = hc_profile.copy()
            unnamed = [c for c in profile.columns if "unnamed" in c.lower()]
            profile = profile.drop(columns=unnamed, errors="ignore")
            st.dataframe(profile, use_container_width=True)

        if rfm_hc_cross is not None:
            st.subheader("RFM Segment × HC Cluster Karşılaştırması")
            fig = px.bar(
                rfm_hc_cross.sort_values("Musteri_Sayisi", ascending=False).head(15),
                x="Musteri_Sayisi",
                y="RFM_Segment",
                color="HC_Cluster_Adi",
                orientation="h",
                title="RFM Segmentlerinin HC Clusterlarına Dağılımı"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("K-Means × HC Validasyonu")

        st.markdown("""
        **Üç Yöntem Karşılaştırması:**
        - RFM → kural bazlı segmentasyon (10 segment)
        - K-Means → matematiksel kümeleme (6 cluster, Silhouette: 0.38)
        - HC → hiyerarşik kümeleme (7 cluster, Silhouette: 0.32)
        """)

        comparison_data = pd.DataFrame({
            "Yöntem": ["RFM", "K-Means", "Hierarchical Clustering"],
            "Segment/Cluster Sayısı": [10, 6, 7],
            "Silhouette Score": ["-", "0.3794", "0.3233"],
            "Veri Boyutu": ["491.705", "491.705", "50.000 (örneklem)"],
            "Yaklaşım": ["Kural bazlı", "Matematiksel", "Hiyerarşik"]
        })
        st.dataframe(comparison_data, use_container_width=True)

        if kmeans_hc_cross is not None:
            st.subheader("K-Means × HC Tutarlılık Analizi")
            fig = px.bar(
                kmeans_hc_cross.sort_values("Musteri_Sayisi", ascending=False),
                x="Musteri_Sayisi",
                y="KMeans_Cluster_Adi",
                color="HC_Cluster_Adi",
                orientation="h",
                title="K-Means Clusterlarının HC'deki Karşılıkları"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.success("✅ K-Means ve HC tutarlı sonuçlar verdi: Aktif müşteriler → Aktif, Pasif müşteriler → Pasif")

# =========================================================
# PAGE 5 - CHURN PREDICTION
# =========================================================

elif page == "5. Churn Prediction":
    st.title("🎯 Churn Prediction")

    tab1, tab2, tab3 = st.tabs([
        "Model Sonuçları",
        "Feature Importance",
        "Segment Bazlı Analiz"
    ])

    with tab1:
        st.subheader("Model Karşılaştırması")

        if model_comparison is not None:
            col1, col2, col3 = st.columns(3)
            best = model_comparison.loc[model_comparison["AUC"].idxmax()]
            col1.metric("En İyi Model", best["Model"])
            col2.metric("AUC", f"{best['AUC']:.4f}")
            col3.metric("F1 Score", f"{best['F1_Score']:.4f}")

            st.divider()
            st.dataframe(model_comparison, use_container_width=True)

            fig = px.bar(
                model_comparison,
                x="Model",
                y=["AUC", "F1_Score", "Accuracy"],
                barmode="group",
                title="Model Karşılaştırması"
            )
            st.plotly_chart(fig, use_container_width=True)

        roc_path = f"{CHURN_DIR}/roc_curve.png"
        cm_path = f"{CHURN_DIR}/confusion_matrix.png"

        col_l, col_r = st.columns(2)
        with col_l:
            if os.path.exists(roc_path):
                st.image(roc_path, caption="ROC Curve", use_column_width=True)
        with col_r:
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Confusion Matrix (XGBoost)", use_column_width=True)

    with tab2:
        st.subheader("Feature Importance (XGBoost)")

        if feature_importance is not None:
            fi_path = f"{CHURN_DIR}/feature_importance.png"
            if os.path.exists(fi_path):
                st.image(fi_path, caption="Feature Importance", use_column_width=True)

            st.dataframe(feature_importance.head(15), use_container_width=True)

            st.info("LastPurchaseDaysAgo modelin %62.7'sini açıklıyor. Bu EDA bulgularıyla örtüşüyor (korelasyon: -0.70). Recency dominant feature.")

    with tab3:
        st.subheader("Segment Bazlı Churn Analizi")

        if segment_churn is not None:
            fig = px.bar(
                segment_churn.sort_values("Gercek_Churn_Orani", ascending=True),
                x="Gercek_Churn_Orani",
                y="RFM_Segment",
                orientation="h",
                title="Segment Bazlı Satın Alma Oranı",
                color="Gercek_Churn_Orani",
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(segment_churn, use_container_width=True)

# =========================================================
# PAGE 6 - PERSONA ENGINE
# =========================================================

elif page == "6. Persona Engine":
    st.title("🧠 LLM Persona Engine")

    st.markdown("Her RFM segmenti için Cohere API ile üretilen müşteri personaları.")

    if personas_df is None:
        missing_file(f"{PERSONA_DIR}/segment_personas.csv")
        st.stop()

    # Segment seçimi
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Toplam Persona", len(personas_df))

    with col2:
        selected_persona_segment = st.selectbox(
            "Segment Seç",
            sorted(personas_df["RFM_Segment"].dropna().unique())
        )

    persona_row = personas_df[personas_df["RFM_Segment"] == selected_persona_segment]

    if not persona_row.empty:
        row = persona_row.iloc[0]

        with col3:
            st.metric("Müşteri Sayısı", format_number(row["Musteri_Sayisi"]))

        st.divider()

        col_l, col_r = st.columns([1, 2])

        with col_l:
            st.subheader("📊 Segment Metrikleri")
            st.metric("Satın Alma Oranı", f"{row['Satin_Alma_Orani']*100:.1f}%")
            st.metric("Ort. Harcama", format_money(row["Ort_Monetary"]))
            st.metric("Ort. Recency", f"{row['Ort_Recency']:.1f} gün")

            st.divider()

            st.subheader("📋 Tüm Segmentler")
            summary = personas_df[["RFM_Segment", "Musteri_Sayisi", "Satin_Alma_Orani", "Ort_Monetary"]].copy()
            summary["Satin_Alma_Orani"] = (summary["Satin_Alma_Orani"] * 100).round(1).astype(str) + "%"
            summary["Ort_Monetary"] = summary["Ort_Monetary"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(summary, use_container_width=True)

        with col_r:
            st.subheader(f"🤖 {selected_persona_segment.upper()} Personası")
            persona_text = str(row["Persona"])
            if persona_text.startswith("Hata:"):
                st.error("Bu segment için persona üretilemedi.")
            else:
                st.markdown(persona_text)

# =========================================================
# PAGE 7 - CUSTOMER INTELLIGENCE
# =========================================================

elif page == "7. Customer Intelligence":
    st.title("👤 Customer Intelligence")

    if rfm_df is None:
        missing_file(f"{RFM_DIR}/customer_rfm_final.csv")
        st.stop()

    # Filtreler
    col1, col2, col3, col4 = st.columns(4)

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

    with col4:
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

    # Özet metrikler
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filtrelenmiş Müşteri", format_number(filtered.shape[0]))
    col2.metric("Ort. Harcama", format_money(filtered["TotalSpent"].mean()) if len(filtered) > 0 else "$0")
    col3.metric("Satın Alma Oranı", f"{filtered['PurchaseStatus'].mean()*100:.1f}%" if len(filtered) > 0 else "0%")
    col4.metric("Ort. Recency", f"{filtered['LastPurchaseDaysAgo'].mean():.1f} gün" if len(filtered) > 0 else "0")

    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "Müşteri Listesi",
        "Kampanya Export",
        "Dağılım Analizi"
    ])

    with tab1:
        st.subheader("Filtrelenmiş Müşteri Listesi")
        st.dataframe(filtered.head(2000), use_container_width=True)
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Müşteri Listesi İndir", data=csv,
                           file_name="filtered_customers.csv", mime="text/csv")

    with tab2:
        st.subheader("Kampanya Export (N8N)")

        if campaign_df is not None:
            filtered_campaign = campaign_df[campaign_df["RFM_Segment"].isin(selected_segments)]

            col1, col2, col3 = st.columns(3)
            col1.metric("Kampanya Müşteri", format_number(filtered_campaign.shape[0]))

            high_priority_df = read_csv(f"{RFM_DIR}/high_priority_campaign_customer_list.csv")
            discount_df = read_csv(f"{RFM_DIR}/discount_campaign_customer_list.csv")

            col2.metric("High Priority", format_number(0 if high_priority_df is None else
                        high_priority_df[high_priority_df["RFM_Segment"].isin(selected_segments)].shape[0]))
            col3.metric("Discount Hedef", format_number(0 if discount_df is None else
                        discount_df[discount_df["RFM_Segment"].isin(selected_segments)].shape[0]))

            st.divider()

            if rfm_discount_summary is not None:
                ds = rfm_discount_summary.copy()
                unnamed = [c for c in ds.columns if "unnamed" in c.lower()]
                ds = ds.drop(columns=unnamed, errors="ignore")
                if "RFM_Segment" not in ds.columns:
                    ds = ds.rename(columns={ds.columns[0]: "RFM_Segment"})
                if "Ortalama_Discount" not in ds.columns:
                    for col in ["avg_discount", "discount_mean", "DiscountsAvailed"]:
                        if col in ds.columns:
                            ds = ds.rename(columns={col: "Ortalama_Discount"})
                            break
                if "Ortalama_Discount" in ds.columns:
                    fig = px.bar(ds, x="RFM_Segment", y="Ortalama_Discount",
                                 title="Segment Bazlı Ortalama Discount Kullanımı")
                    st.plotly_chart(fig, use_container_width=True)

            st.dataframe(filtered_campaign.head(1000), use_container_width=True)
            csv = filtered_campaign.to_csv(index=False).encode("utf-8")
            st.download_button("Kampanya Listesi İndir", data=csv,
                               file_name="campaign_export.csv", mime="text/csv")

    with tab3:
        st.subheader("Dağılım Analizi")

        if len(filtered) > 0:
            numeric_cols = ["TotalSpent", "LastPurchaseDaysAgo", "NumberOfPurchases",
                            "TimeSpentOnWebsite", "CustomerTenureYears", "DiscountsAvailed"]
            numeric_cols = [c for c in numeric_cols if c in filtered.columns]

            selected_num = st.selectbox("Değişken Seç", numeric_cols)

            q1 = filtered[selected_num].quantile(0.25)
            q3 = filtered[selected_num].quantile(0.75)
            iqr = q3 - q1

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Min", f"{filtered[selected_num].min():.2f}")
            col2.metric("Max", f"{filtered[selected_num].max():.2f}")
            col3.metric("IQR Alt", f"{q1 - 1.5*iqr:.2f}")
            col4.metric("IQR Üst", f"{q3 + 1.5*iqr:.2f}")

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**Tüm Veri**")
                if rfm_df is not None and selected_num in rfm_df.columns:
                    fig = px.histogram(rfm_df, x=selected_num, nbins=50, title="Genel Dağılım")
                    st.plotly_chart(fig, use_container_width=True)
            with col_r:
                st.markdown("**Filtrelenmiş**")
                fig = px.histogram(filtered, x=selected_num, nbins=50, title="Filtrelenmiş Dağılım")
                st.plotly_chart(fig, use_container_width=True)