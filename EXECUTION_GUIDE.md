# 🚀 Execution Order Guide - Raw Data → Production Dashboard

**Raw customerData_500k.csv → 500K Customer Intelligence Platform**

---

## 📋 Step-by-Step Execution

### **PHASE 0: FEATURE ENGINEERING** ⚙️

#### Step 0️⃣: Monetary Feature Creation
```bash
python src/monetary.py
```
**Input**: `data/customerData_500k.csv`  
**Output**: `data/customerData_500k_v2.csv`  
**Time**: ~1 minute  
**What it does**:
- Creates TotalSpent feature based on:
  - Product category average basket
  - Customer segment multiplier
  - Discount impact
  - Loyalty program bonus
- Formula: `avg_basket × seg_mult × discount_factor × loyalty_bonus × NumberOfPurchases`
- Example:
  - Electronics + VIP + Loyalty + 10 purchases = higher TotalSpent
  - Groceries + Regular + No loyalty + 2 purchases = lower TotalSpent

---

### **PHASE 1: DATA PREPARATION** 🔧

#### Step 1️⃣: Data Cleaning
```bash
python src/data_prep.py
```
**Input**: `data/customerData_500k_v2.csv` (output from Step 0)  
**Output**: `data/clean_customer_data.csv`  
**Time**: ~2 minutes  
**What it does**: 
- Removes duplicates
- Handles negative values
- Standardizes data types
- Creates clean dataset (500K rows, ready for analysis)

---

#### Step 2️⃣: Exploratory Data Analysis
```bash
python src/eda_final.py
```
**Input**: `data/clean_customer_data.csv`  
**Output**: `data/Eda_Output/` (20+ visualizations + reports)  
**Time**: ~5 minutes  
**What it does**:
- Numerical variable analysis
- Categorical variable distribution
- Correlation analysis (LastPurchaseDaysAgo dominant!)
- Feature importance insights
- Target variable (PurchaseStatus) analysis

---

### **PHASE 2: SEGMENTATION** 📊

#### Step 3️⃣: RFM Analysis
```bash
python src/rfm_analysis.py
```
**Input**: `data/clean_customer_data.csv`  
**Output**: 
- `data/RFM_Output/customer_rfm_final.csv` (491,705 customers with RFM)
- `data/RFM_Output/rfm_segment_distribution.csv`
- `data/RFM_Output/campaign_customer_list.csv`

**Time**: ~3 minutes  
**What it does**:
- Calculates Recency, Frequency, Monetary scores (using TotalSpent from Step 0)
- Creates 10 RFM segments:
  - Champions (2.9% churn)
  - Loyal Customers (35.1% churn)
  - Potential Loyalists (13.9% churn)
  - New Customers (8.6% churn)
  - Promising (31.0% churn)
  - Need Attention (62.6% churn)
  - About to Sleep (70.2% churn)
  - At Risk (99.9% churn)
  - Can't Lose (99.9% churn)
  - Hibernating (100% churn)
- Generates segment profiles
- Creates campaign-ready customer lists

---

### **PHASE 3: CLUSTERING VALIDATION** 🎯

#### Step 4️⃣: KMeans Clustering
```bash
python src/kmeans.py
```
**Input**: `data/RFM_Output/customer_rfm_final.csv`  
**Output**:
- `data/KMeans_Output/kmeans_cluster_profile_rfm.csv`
- `data/KMeans_Output/rfm_kmeans_cross.csv`
- `data/KMeans_Output/elbow_silhouette_rfm.png`

**Time**: ~2 minutes  
**What it does**:
- Finds optimal K using Elbow Method + Silhouette Score
- Result: K=6 optimal clusters
- Silhouette Score: 0.3794
- Validates RFM segments mathematically
- Creates KMeans cluster profiles

---

#### Step 5️⃣: Hierarchical Clustering
```bash
python src/hierarchical.py
```
**Input**: `data/RFM_Output/customer_rfm_final.csv`  
**Output**:
- `data/HC_Output/hc_cluster_profile.csv`
- `data/HC_Output/rfm_hc_cross.csv`
- `data/HC_Output/dendrogram.png`
- `data/HC_Output/hc_silhouette.png`

**Time**: ~10 minutes  
**What it does**:
- Stratified sampling (50K records from 500K)
- Ward linkage hierarchical clustering
- Result: K=7 optimal clusters
- Silhouette Score: 0.3233
- Three-method comparison validation

---

### **PHASE 4: CHURN PREDICTION** 🔮

#### Step 6️⃣: ML Churn Models
```bash
python src/churn_ml.py
```
**Input**: `data/RFM_Output/customer_rfm_final.csv`  
**Output**:
- `data/Churn_Output/model_comparison.csv`
- `data/Churn_Output/feature_importance.csv`
- `data/Churn_Output/churn_predictions.csv` (491K rows with predictions)
- `data/Churn_Output/segment_churn_analysis.csv`
- `data/Churn_Output/roc_curve.png`
- `data/Churn_Output/confusion_matrix.png`

**Time**: ~5 minutes  
**What it does**:
- Trains 3 models:
  - Logistic Regression (AUC: 0.9804) ⭐
  - Random Forest (AUC: 0.9782)
  - XGBoost (AUC: 0.9804) ⭐
- Selects best model (Logistic Regression)
- Feature Importance ranking:
  1. LastPurchaseDaysAgo: 62.7% (dominant!)
  2. LoyaltyProgram: 8.0%
  3. CustomerSatisfaction: 7.9%
  4. CustomerSegment: 4.6%
- Predicts churn probability for all 500K customers
- Validates against RFM segments (perfect alignment ✅)

---

### **PHASE 5: BUSINESS INTELLIGENCE** 💼

#### Step 7️⃣: Segment Intelligence & KPI
```bash
python src/segment_intelligence.py
```
**Input**: 
- `data/RFM_Output/customer_rfm_final.csv`
- `data/Churn_Output/churn_predictions.csv`

**Output**: `data/Intelligence_Output/`  
**Time**: ~2 minutes  
**What it does**:
- Enriches segments with behavioral metrics:
  - Avg Recency, Frequency, Monetary
  - Purchase Rate %
  - Churn Risk %
  - Dominant Product Category
  - Dominant Region
  - Dominant Age Group
  - Dominant Device
  - Discount Affinity
  - Loyalty Score

---

#### Step 8️⃣: Financial Impact Calculation
```bash
python src/business_kpi_layer_ml.py
```
**Input**: 
- `data/RFM_Output/customer_rfm_final.csv`
- `data/Churn_Output/churn_predictions.csv`

**Output**:
- `data/Business_KPI_Output/rfm_business_kpi.csv`
- `data/Business_KPI_Output/kmeans_business_kpi.csv`
- `data/Business_KPI_Output/hc_business_kpi.csv`
- `data/Business_KPI_Output/overview_kpi_summary.csv`
- `data/Business_KPI_Output/risk_distribution.csv`

**Time**: ~2 minutes  
**What it does**:
- Calculates per-segment financial metrics:
  - Revenue at Risk = TotalSpent × Churn Probability
  - Recoverable Revenue = Revenue at Risk × Recovery Rate
  - Potential Retention Gain = Recoverable Revenue × Campaign ROI
  - High Value Customer Exposure
- Summary KPIs:
  - **Total Revenue at Risk**: $319.1M
  - **Recoverable Revenue**: $29.4M
  - **Recovery Rate**: 9.2% (ML-based)
  - **Risk Distribution**: 
    - 🟢 Healthy: 198,720 customers
    - 🟡 Warning: 214,816 customers
    - 🔴 Critical: 78,169 customers

---

### **PHASE 6: AI PERSONAS** 👥

#### Step 9️⃣: LLM Persona Generation
```bash
python src/persona_v2.py
```
**Input**: 
- `data/RFM_Output/customer_rfm_final.csv`
- `data/Churn_Output/churn_predictions.csv`
- Segment intelligence data

**Output**:
- `data/Persona_Output/segment_personas.csv`
- `data/Persona_Output/{segment}_persona.txt` (10 files)

**Time**: ~5 minutes (includes Cohere API calls)  
**What it does**:
- Uses Cohere API to generate 10 personas:
  1. Champions (Sadık Moda Tutkunları)
  2. Loyal Customers (Sadık Müşteriler)
  3. Potential Loyalists (Potansiyel Sadık)
  4. New Customers (Yeni Müşteriler)
  5. Promising (Umut Veren)
  6. Need Attention (Dikkat Gereken)
  7. About to Sleep (Uyumaya Hazır)
  8. At Risk (Risk Altında)
  9. Can't Lose (Kaybetmemeliyiz)
  10. Hibernating (Hibernasyonda)

- Each persona includes:
  - Behavioral profile
  - Motivations & needs
  - Risk assessment
  - Recommended campaign strategy
  - CRM action items

**Requirements**: `.env` file with `COHERE_API_KEY`

---

### **PHASE 7: DASHBOARD** 🎨

#### Step 🔟: Run Streamlit Dashboard
```bash
streamlit run src/presentation_app.py
```
**Input**: All CSV files from steps 0-9  
**Output**: 7-page interactive dashboard  
**Time**: Real-time (instant)  
**What it does**:
1. **Hero** - Overview + KPI cards (500K customers, $319.1M at risk)
2. **Problem** - Business challenges + risk distribution
3. **RFM Intelligence** - 10 segments + loyalty journey
4. **Clustering** - KMeans + HC validation (3-method proof)
5. **Churn** - ML predictions + feature importance (62.7% LastPurchaseDaysAgo)
6. **Personas** - 10 AI personas + campaign strategies
7. **Action** - Campaign automation + n8n payload

---

## ⚡ Quick Summary Table

| # | Script | Input | Output | Time | Purpose |
|---|--------|-------|--------|------|---------|
| 0 | `monetary.py` | Raw 500K | TotalSpent feature | 1m | Feature Engineering |
| 1 | `data_prep.py` | Raw + TotalSpent | Clean 500K | 2m | Data Cleaning |
| 2 | `eda_final.py` | Clean data | EDA reports | 5m | Exploratory Analysis |
| 3 | `rfm_analysis.py` | Clean data | 10 RFM segments | 3m | RFM Segmentation |
| 4 | `kmeans.py` | RFM scores | 6 KMeans clusters | 2m | Clustering Validation |
| 5 | `hierarchical.py` | RFM scores | 7 HC clusters | 10m | Hierarchical Validation |
| 6 | `churn_ml.py` | RFM data | Churn predictions | 5m | ML Churn Modeling |
| 7 | `segment_intelligence.py` | RFM + Churn | Intelligence profiles | 2m | Segment Enrichment |
| 8 | `business_kpi_layer_ml.py` | RFM + Churn | Business KPIs | 2m | Financial Impact |
| 9 | `persona_v2.py` | RFM + Churn | LLM Personas | 5m | AI Persona Generation |
| 10 | `presentation_app.py` | All CSVs | 7-page Dashboard | Real-time | Interactive Dashboard |

**Total Execution Time**: ~37 minutes (one-time setup)

---

## 🔄 Full Automation Script

Save as `run_pipeline.sh`:

```bash
#!/bin/bash

echo "🚀 Starting full AI Customer Intelligence pipeline..."

echo "0️⃣ Feature Engineering (TotalSpent)..."
python src/monetary.py

echo "1️⃣ Data Preparation..."
python src/data_prep.py

echo "2️⃣ EDA Analysis..."
python src/eda_final.py

echo "3️⃣ RFM Segmentation..."
python src/rfm_analysis.py

echo "4️⃣ KMeans Clustering..."
python src/kmeans.py

echo "5️⃣ Hierarchical Clustering..."
python src/hierarchical.py

echo "6️⃣ Churn Prediction..."
python src/churn_ml.py

echo "7️⃣ Segment Intelligence..."
python src/segment_intelligence.py

echo "8️⃣ Business KPI Layer..."
python src/business_kpi_layer_ml.py

echo "9️⃣ LLM Persona Generation..."
python src/persona_v2.py

echo "✅ Pipeline complete!"
echo "🎨 Starting dashboard..."
streamlit run src/presentation_app.py
```

Execute:
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

---

## 📥 Input Data Required

Place raw data:
data/customerData_500k.csv

---

## 🔑 Environment Setup

Create `.env`:
COHERE_API_KEY=your_cohere_api_key_here
N8N_WEBHOOK_URL=your_n8n_webhook_url

---

## ✅ Verification Checklist

```bash
# After Step 0
ls -lh data/customerData_500k_v2.csv

# After Step 1
ls -lh data/clean_customer_data.csv

# After Step 3
ls -lh data/RFM_Output/customer_rfm_final.csv

# After Step 6
ls -lh data/Churn_Output/churn_predictions.csv

# After Step 8
ls -lh data/Business_KPI_Output/overview_kpi_summary.csv

# After Step 9
ls -lh data/Persona_Output/
```

---

## 🚀 Dashboard Launch

```bash
streamlit run src/presentation_app.py
```

Navigate to: `http://localhost:8501`

---

## 📊 Expected Final Results

✅ **500,000 customers** analyzed  
✅ **10 RFM segments** + profiles  
✅ **6 KMeans clusters** validated  
✅ **7 Hierarchical clusters** confirmed  
✅ **3 ML models** (Logistic Regression, Random Forest, XGBoost)  
✅ **Churn AUC: 0.9804** (98% accuracy)  
✅ **$319.1M revenue at risk** identified  
✅ **$29.4M recoverable revenue** calculated  
✅ **10 AI personas** generated  
✅ **Campaign-ready audiences** exported  
✅ **7-page interactive dashboard** live  

---

**Status**: ✅ Production Ready

**Last Updated**: May 16, 2024

