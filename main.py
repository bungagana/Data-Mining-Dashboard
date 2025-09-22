import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Cohort LOS Dashboard", layout="wide")

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    df_raw = pd.read_csv("Mali_Cohort_Study.csv")   # <-- ganti dengan data rawmu
    df_cleaned = pd.read_csv("cleaned_dataset.csv")  # hasil preprocessing
    results = pd.read_csv("model_evaluation_results.csv")  # hasil modeling dari notebook
    return df_raw, df_cleaned, results

df_raw, df_cleaned, model_results = load_data()

# ======================
# SIDEBAR
# ======================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["EDA (Raw Data)", "Preprocessing", "Feature Selection (LOS)", "Modeling"])

# ======================
# HELPER: SUMMARY CARDS
# ======================
def summary_card(title, value, subtitle, color):
    st.markdown(
        f"""
        <div style='background-color:{color}; padding:20px; border-radius:10px; height:130px; text-align:center;'>
            <h2 style='margin:0; color:black;'>{value}</h2>
            <p style='margin:0; color:black; font-size:14px;'>{title}</p>
            <p style='margin:0; font-size:12px; color:gray;'>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ======================
# PAGE: EDA
# ======================
if page == "EDA (Raw Data)":
    st.title("🔍 Exploratory Data Analysis (Raw Data)")

    # Summary cards
    col1, col2, col3 = st.columns(3)
    with col1: summary_card("Rows", df_raw.shape[0], "Jumlah baris", "#E3F2FD")
    with col2: summary_card("Columns", df_raw.shape[1], "Jumlah kolom", "#E3F2FD")
    with col3: summary_card("Columns with Missing", df_raw.isna().sum().gt(0).sum(), "Jumlah kolom dengan missing", "#E3F2FD")

    st.subheader("Preview Data")
    st.dataframe(df_raw.head())

    st.subheader("Missing Values per Column")
    missing = df_raw.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        st.bar_chart(missing)

    st.subheader("Distribusi Fitur Numerik")
    num_cols = df_raw.select_dtypes(include=np.number).columns
    for col in num_cols[:5]:  # tampilkan 5 contoh
        fig, ax = plt.subplots()
        sns.histplot(df_raw[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribusi {col}")
        st.pyplot(fig)

    st.subheader("Distribusi Fitur Kategorikal")
    cat_cols = df_raw.select_dtypes(exclude=np.number).columns
    for col in cat_cols[:3]:  # tampilkan 3 contoh
        fig, ax = plt.subplots()
        df_raw[col].value_counts().head(10).plot(kind="bar", ax=ax)
        ax.set_title(f"Top 10 {col}")
        st.pyplot(fig)

# ======================
# PAGE: PREPROCESSING
# ======================
elif page == "Preprocessing":
    st.title("🧹 Data Preprocessing")

    # Summary cards
    col1, col2, col3 = st.columns(3)
    with col1: summary_card("Rows", df_cleaned.shape[0], "Setelah preprocessing", "#E8F5E9")
    with col2: summary_card("Columns", df_cleaned.shape[1], "Jumlah kolom", "#E8F5E9")
    with col3: summary_card("Columns with Missing", df_cleaned.isna().sum().gt(0).sum(), "Kolom masih missing", "#E8F5E9")

    st.subheader("Apa yang dilakukan di preprocessing?")
    st.markdown("""
    - 🔄 Replace `inf` dengan NaN  
    - 🧹 Imputasi missing value: numeric (mean), categorical (mode)  
    - 🗑️ Drop kolom irrelevan: `Anon_ID`, `record_id`, dll  
    - 🔢 Konversi tipe data ke float64 untuk numeric  
    - 📊 Standardisasi fitur numerik  
    """)

    st.subheader("Missing Values Before vs After")
    before = df_raw.isnull().sum().sum()
    after = df_cleaned.isnull().sum().sum()
    st.write(f"**Before:** {before} missing values | **After:** {after} missing values")

    st.subheader("Preview Cleaned Data")
    st.dataframe(df_cleaned.head())

# ======================
# PAGE: FEATURE SELECTION
# ======================
elif page == "Feature Selection (LOS)":
    st.title("🎯 Feature Selection for LOS Prediction")

    corr_method = st.selectbox("Pilih metode korelasi", ["pearson", "spearman"])
    numeric_cols = df_cleaned.select_dtypes(include=np.number)

    if "LOS" in numeric_cols:
        corr = numeric_cols.corr(method=corr_method)["LOS"].sort_values(ascending=False)
        st.subheader(f"Korelasi dengan LOS ({corr_method})")
        st.dataframe(corr)

        fig, ax = plt.subplots()
        corr.head(10).plot(kind="bar", ax=ax, color="skyblue")
        ax.set_title("Top 10 Korelasi dengan LOS")
        st.pyplot(fig)

    # RandomForest Importance
    st.subheader("Feature Importance (RandomForest)")
    X = numeric_cols.drop(columns=["LOS"])
    y = numeric_cols["LOS"]
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X, y)
    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.dataframe(importance.head(15))

    fig, ax = plt.subplots()
    importance.head(15).plot(kind="bar", ax=ax, color="orange")
    ax.set_title("RandomForest Feature Importance")
    st.pyplot(fig)

# ======================
# PAGE: MODELING
# ======================
elif page == "Modeling":
    st.title("🤖 Modeling Scenarios & Results")

    st.subheader("Hasil Evaluasi Model (dari Notebook)")
    st.dataframe(model_results)

    st.subheader("Perbandingan Metrik")
    metrics = ["MSE", "RMSE", "MAE", "R2"]
    for metric in metrics:
        if metric in model_results.columns:
            fig, ax = plt.subplots()
            sns.barplot(data=model_results, x="Scenario", y=metric, ax=ax)
            ax.set_title(f"Perbandingan {metric} antar Scenario")
            st.pyplot(fig)
