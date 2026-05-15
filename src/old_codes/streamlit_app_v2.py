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

CLEAN_PATH = "data/Cleaned_data/clean_customer_data.csv"
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


# =========================================================
# LOAD DATA
# =========================================================

clean_df = read_csv(CLEAN_PATH)
rfm_df = read_csv(f"{RFM_DIR}/customer_rfm_final.csv")
campaign_df = read_csv(f"{RFM_DIR}/campaign_customer_list.csv")


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
            data = clean_df["ProductCategory"].value_counts().reset_index()
            data.columns = ["ProductCategory", "Count"]
            fig = px.bar(data, x="ProductCategory", y="Count")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Gender Dağılımı")
            fig = px.pie(clean_df, names="Gender", hole=0.35)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Region Dağılımı")
            data = clean_df["Region"].value_counts().reset_index()
            data.columns = ["Region", "Count"]
            fig = px.bar(data, x="Region", y="Count")
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.subheader("ReferralSource Dağılımı")
            data = clean_df["ReferralSource"].value_counts().reset_index()
            data.columns = ["ReferralSource", "Count"]
            fig = px.bar(data, x="ReferralSource", y="Count")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Tüm Değişkenlerin Target ile Analizi")

        selected_cat = st.selectbox(
            "Kategorik Değişken Seç",
            ["Gender", "ProductCategory", "PreferredDevice", "Region", "ReferralSource", "CustomerSegment", "LoyaltyProgram", "CustomerSatisfaction"]
        )

        target_summary = clean_df.groupby(selected_cat).agg(
            Musteri_Sayisi=("CustomerID", "count"),
            Satin_Alma_Orani=("PurchaseStatus", "mean"),
            Ortalama_Harcama=("TotalSpent", "mean")
        ).reset_index()

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

    with tab3:
        st.subheader("Kategori Bazlı Satış Hikayesi")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Gender × ProductCategory**")
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

    with tab4:
        st.subheader("Korelasyon Matrisi")

        numeric_df = clean_df.select_dtypes(include=["int64", "float64"])
        corr = numeric_df.corr()

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
        st.metric("Toplam Müşteri", format_number(rfm_df["CustomerID"].nunique()))

    with col2:
        st.metric("Toplam Segment", rfm_df["RFM_Segment"].nunique())

    with col3:
        selected_segment = st.selectbox("Segment Seçin", segment_list)

    filtered = rfm_df[rfm_df["RFM_Segment"] == selected_segment]

    st.divider()

    left, right = st.columns([1, 2])

    with left:
        st.subheader(f"📊 {selected_segment.upper()}")

        st.metric("Müşteri Sayısı", format_number(len(filtered)))
        st.metric("Ort. Recency", f"{filtered['recency'].mean():.2f} gün")
        st.metric("Ort. Frequency", f"{filtered['frequency'].mean():.2f}")
        st.metric("Ort. Monetary", format_money(filtered["monetary"].mean()))
        st.metric("Satın Alma Oranı", f"{filtered['PurchaseStatus'].mean() * 100:.2f}%")

    with right:
        seg_counts = rfm_df["RFM_Segment"].value_counts().reset_index()
        seg_counts.columns = ["RFM_Segment", "Count"]

        fig = px.pie(
            seg_counts,
            values="Count",
            names="RFM_Segment",
            title="Segmentlerin Dağılımı",
            hole=0.35
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Segment Dağılımı",
        "Recency",
        "Frequency",
        "Monetary",
        "R/F/M Karşılaştırma",
        "Heatmap"
    ])

    with tab1:
        st.subheader("Segment Bazlı Müşteri Sayısı")

        counts = rfm_df["RFM_Segment"].value_counts().reset_index()
        counts.columns = ["RFM_Segment", "Count"]
        counts["Highlight"] = np.where(counts["RFM_Segment"] == selected_segment, "Seçili Segment", "Diğer")

        fig = px.bar(
            counts,
            y="RFM_Segment",
            x="Count",
            color="Highlight",
            orientation="h",
            title=f"{selected_segment} Segmentinin Genel İçindeki Yeri"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Genel - Tüm Müşteriler**")
            fig = px.histogram(rfm_df, x="recency", nbins=40, title="Genel Recency Dağılımı")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown(f"**Segment - {selected_segment.upper()}**")
            fig = px.histogram(filtered, x="recency", nbins=40, title=f"{selected_segment} Recency Dağılımı")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Genel - Tüm Müşteriler**")
            fig = px.histogram(rfm_df, x="frequency", nbins=40, title="Genel Frequency Dağılımı")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown(f"**Segment - {selected_segment.upper()}**")
            fig = px.histogram(filtered, x="frequency", nbins=40, title=f"{selected_segment} Frequency Dağılımı")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Genel - Tüm Müşteriler**")
            fig = px.histogram(rfm_df, x="monetary", nbins=40, title="Genel Monetary Dağılımı")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown(f"**Segment - {selected_segment.upper()}**")
            fig = px.histogram(filtered, x="monetary", nbins=40, title=f"{selected_segment} Monetary Dağılımı")
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Segment Bazlı Normalize R/F/M Karşılaştırma")

        rfm_avg = rfm_df.groupby("RFM_Segment")[["recency", "frequency", "monetary"]].mean()
        rfm_norm = (rfm_avg - rfm_avg.min()) / (rfm_avg.max() - rfm_avg.min())
        rfm_norm = rfm_norm.reset_index()

        fig = px.bar(
            rfm_norm,
            x="RFM_Segment",
            y=["recency", "frequency", "monetary"],
            barmode="group",
            title="Normalize R/F/M Karşılaştırması"
        )
        st.plotly_chart(fig, use_container_width=True)

        selected_vals = filtered[["recency", "frequency", "monetary"]].mean()
        global_min = rfm_df[["recency", "frequency", "monetary"]].min()
        global_max = rfm_df[["recency", "frequency", "monetary"]].max()
        selected_norm = ((selected_vals - global_min) / (global_max - global_min)).reset_index()
        selected_norm.columns = ["Metric", "Value"]

        fig = px.bar(
            selected_norm,
            x="Metric",
            y="Value",
            title=f"{selected_segment} Normalize R/F/M Profili",
            range_y=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab6:
        st.subheader("Segment Bazlı Ortalama RFM Skorları")

        score_cols = ["recency_score", "frequency_score", "monetary_score"]
        heatmap_data = rfm_df.groupby("RFM_Segment")[score_cols].mean().round(2)

        fig = px.imshow(
            heatmap_data,
            text_auto=True,
            aspect="auto",
            title="Segment Bazlı Ortalama RFM Skorları"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"{selected_segment} - R/F Skorlarına Göre Ortalama M Skoru")

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
            title=f"{selected_segment} RF Heatmap"
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

            ds = discount_summary.reset_index()
            if "RFM_Segment" not in ds.columns:
                ds = ds.rename(columns={ds.columns[0]: "RFM_Segment"})

            fig = px.bar(
                ds,
                x="RFM_Segment",
                y="Ortalama_Discount",
                title="Segment Bazlı Ortalama Discount"
            )
            st.plotly_chart(fig, use_container_width=True)
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