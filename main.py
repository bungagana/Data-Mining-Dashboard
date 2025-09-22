# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Mali Cohort Study — Data Analyst Dashboard",
                   layout="wide", initial_sidebar_state="expanded")

sns.set_style("whitegrid")

# -----------------------
# Utilities & Caching
# -----------------------
@st.cache_data
def load_data(path="Mali_Cohort_Study.csv"):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Error loading CSV at `{path}`: {e}")
        return None
    return df

@st.cache_data
def preprocess_pipeline(df_raw):
    df = df_raw.copy()

    # Columns to drop that were in your notebook
    drop_candidates = ['in_treatment', 'adm_WaSt_category',
                       'adm_WHZltneg3WAZgtneg3', 'adm_MUAClt110',
                       'Screened_hf', 'adm_WHZgtneg3WAZgtneg3', 'Cared_hws']

    # also drop obvious ID-like columns (e.g., anon_id, id, patient_id) if present
    id_like = [c for c in df.columns if any(x in c.lower() for x in ['anon', 'id', 'patient']) and df[c].nunique() == df.shape[0]]
    # remove any duplicates in lists
    drop_list = list(dict.fromkeys(drop_candidates + id_like))

    existing_drop = [c for c in drop_list if c in df.columns]
    df = df.drop(columns=existing_drop, errors='ignore')

    # Replace inf with nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Detect types (use broader numeric selection)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Impute numeric with mean, categorical with mode
    if numeric_cols:
        num_imputer = SimpleImputer(strategy="mean")
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # Convert numeric columns to float64 explicitly
    for c in numeric_cols:
        try:
            df[c] = df[c].astype('float64')
        except Exception:
            pass

    # Standardize numeric columns (create scaled copy)
    scaler = StandardScaler()
    df_scaled = df.copy()
    if numeric_cols:
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

    return df, df_scaled, numeric_cols, cat_cols, existing_drop, id_like

def get_missing_table(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    miss_df = pd.DataFrame({
        "Missing Values": missing,
        "Percentage": 100 * missing / len(df)
    }).sort_values("Percentage", ascending=False)
    return miss_df

def to_download_bytes(df, name="cleaned.csv"):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

# -----------------------
# Load data
# -----------------------
df_raw = load_data()
if df_raw is None:
    st.stop()

df_cleaned, df_scaled, numeric_cols, cat_cols, dropped_columns, id_like_cols = preprocess_pipeline(df_raw)

# -----------------------
# Helper: HTML Metric Card
# -----------------------
def metric_card(title, value, bg="#f7fbff"):
    return f"""
    <div style="background:{bg};padding:14px;border-radius:10px;text-align:center;">
        <div style="font-size:14px;color:#333;">{title}</div>
        <div style="font-size:28px;font-weight:700;color:#111;margin-top:6px;">{value}</div>
    </div>
    """

# -----------------------
# Sidebar navigation
# -----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA (Raw Data)", "Preprocessing", "Feature Selection (LOS)", "Modeling"])

# Global options
st.sidebar.markdown("---")
st.sidebar.markdown("Display options")
palette = st.sidebar.selectbox("Seaborn palette", ["viridis", "Set2", "mako", "coolwarm"], index=0)
sns.set_palette(palette)

# -----------------------
# Top-level header + metrics (Raw & Cleaned)
# -----------------------
st.title("📊 Mali Cohort Study — Data Analyst Dashboard")
st.markdown("Transparansi penuh: tampilkan kondisi **raw** (sebelum preprocessing) dan kondisi **cleaned** (setelah preprocessing).")

# Raw metrics
raw_rows, raw_cols = df_raw.shape
raw_missing_cols = (df_raw.isnull().sum() > 0).sum()

clean_rows, clean_cols = df_cleaned.shape
clean_missing_cols = (df_cleaned.isnull().sum() > 0).sum()

col1, col2 = st.columns([1,1])
with col1:
    st.markdown("<h4 style='margin:0;'>Raw Data Summary</h4>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_card("Rows", raw_rows, bg="#fff7e6"), unsafe_allow_html=True)
    c2.markdown(metric_card("Columns", raw_cols, bg="#fff7e6"), unsafe_allow_html=True)
    c3.markdown(metric_card("Columns with Missing", raw_missing_cols, bg="#fff7e6"), unsafe_allow_html=True)

with col2:
    st.markdown("<h4 style='margin:0;'>Cleaned Data Summary</h4>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_card("Rows", clean_rows, bg="#e8fff2"), unsafe_allow_html=True)
    c2.markdown(metric_card("Columns", clean_cols, bg="#e8fff2"), unsafe_allow_html=True)
    c3.markdown(metric_card("Columns with Missing", clean_missing_cols, bg="#e8fff2"), unsafe_allow_html=True)

st.markdown("---")

# -----------------------
# 1) EDA (Raw Data)
# -----------------------
if page == "EDA (Raw Data)":
    st.header("🔎 Exploratory Data Analysis — Raw Data (Before Preprocessing)")

    tab_overview, tab_missing, tab_dist, tab_outcome, tab_insights = st.tabs(["📋 Overview", "❓ Missing Values", "📊 Distributions", "🏥 Outcome", "💡 Insights"])

    with tab_overview:
        st.subheader("Preview (first 20 rows)")
        st.dataframe(df_raw.head(20))
        st.markdown("**Columns & dtypes**")
        dtype_table = pd.DataFrame(df_raw.dtypes, columns=["dtype"]).reset_index().rename(columns={"index":"column"})
        st.dataframe(dtype_table)

    with tab_missing:
        st.subheader("Missing values overview")
        miss_raw = get_missing_table(df_raw)
        if miss_raw.empty:
            st.success("No missing values detected in raw (unexpected).")
        else:
            st.dataframe(miss_raw.head(30))
            fig, ax = plt.subplots(figsize=(10,4))
            sns.barplot(x=miss_raw.reset_index()["index"], y="Percentage", data=miss_raw.reset_index(), palette=palette, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_ylabel("Missing %")
            st.pyplot(fig)

    with tab_dist:
        st.subheader("Numeric distributions (choose variable)")
        num_options = df_raw.select_dtypes(include=np.number).columns.tolist()
        if not num_options:
            st.info("No numeric columns found in raw data.")
        else:
            sel_num = st.selectbox("Select numeric column", num_options, index=0)
            bins = st.slider("Bins", 5, 100, 30)
            fig, ax = plt.subplots(figsize=(8,4))
            sns.histplot(df_raw[sel_num].dropna(), bins=bins, kde=True, ax=ax)
            ax.set_title(f"{sel_num} (raw)")
            st.pyplot(fig)

    with tab_outcome:
        st.subheader("Outcome distribution (if 'status' exists)")
        if "status" in df_raw.columns:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.countplot(x="status", data=df_raw, palette=palette, order=df_raw['status'].value_counts().index, ax=ax)
            ax.set_title("status (raw)")
            st.pyplot(fig)
        else:
            st.info("'status' column not present in raw data.")

    with tab_insights:
        st.subheader("Key findings (raw)")
        st.markdown("""
        - Beberapa kolom full-empty / mostly empty (ditampilkan di tab Missing).  
        - Outcome cenderung imbalanced (mayoritas `cured`).  
        - Ada nilai ekstrem & nilai negatif pada beberapa metrik klinis.  
        - Terdapat kolom ID/anon yang tidak relevan untuk modeling (akan di-drop).
        """)
        with st.expander("Show top 10 rows raw data (expanded)"):
            st.dataframe(df_raw.head(100))

# -----------------------
# 2) Preprocessing
# -----------------------
elif page == "Preprocessing":
    st.header("🧹 Preprocessing (What we did & results)")

    tab_steps, tab_cleaned, tab_dropped, tab_compare = st.tabs(["📝 Steps", "📋 Cleaned Data", "🗑️ Dropped Features", "🔁 Missing Before vs After"])

    with tab_steps:
        st.subheader("Preprocessing steps (applied automatically)")
        st.markdown("""
        1. Drop obviously-empty / irrelevant columns (e.g., `in_treatment`, `adm_WaSt_category`, some screening columns).  
        2. Drop ID-like columns (e.g., `anon_id`, `patient_id`) if they were detected.  
        3. Replace `inf` / `-inf` with `NaN`.  
        4. Impute missing values:
           - numeric → mean  
           - categorical → mode
        5. Convert numeric columns to `float64`.  
        6. Standardize numeric features with `StandardScaler` (keperluan modeling).
        """)

    with tab_cleaned:
        st.subheader("Cleaned dataset preview")
        st.dataframe(df_cleaned.head(20))
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df_cleaned.shape[0])
        c2.metric("Columns", df_cleaned.shape[1])
        c3.metric("Numeric cols", len(numeric_cols))

        st.markdown("Download cleaned dataset (CSV)")
        buf = to_download_bytes(df_cleaned, name="cleaned_mali_cohort.csv")
        st.download_button("Download cleaned CSV", data=buf, file_name="cleaned_mali_cohort.csv", mime="text/csv")

    with tab_dropped:
        st.subheader("Dropped / Irrelevant columns (auto-detected)")
        if dropped_columns:
            reasons = []
            for c in dropped_columns:
                reason = "explicit drop (notebook list)"
                if c in id_like_cols:
                    reason = "ID-like column (high cardinality unique per row) -> removed"
                reasons.append((c, reason))
            dropped_df = pd.DataFrame(reasons, columns=["column", "reason"])
            st.dataframe(dropped_df)
        else:
            st.info("No columns were dropped by the pipeline (none matched the drop list).")

        with st.expander("Notes on why columns are dropped"):
            st.write("""
            - ID columns (anon/patient/id) are dropped to avoid leakage and because they don't carry predictive signal.  
            - Columns with extremely high missingness or duplicates of other columns can be dropped to reduce noise.  
            - Screening flags (if mostly empty) also removed since they don't help LOS modelling in current dataset.
            """)

    with tab_compare:
        st.subheader("Missing values: Before vs After")
        miss_before = get_missing_table(df_raw)
        miss_after = get_missing_table(df_cleaned)
        # combine top 20 keys for comparison
        keys = list(pd.Index(miss_before.index).union(miss_after.index))[:40]
        comp = pd.DataFrame({
            "missing_before": miss_before["Percentage"].reindex(keys).fillna(0),
            "missing_after": miss_after["Percentage"].reindex(keys).fillna(0)
        }).fillna(0)
        st.dataframe(comp.sort_values("missing_before", ascending=False).head(40))

        fig, ax = plt.subplots(figsize=(10,6))
        comp_plot = comp.sort_values("missing_before", ascending=False).head(20)
        index = np.arange(len(comp_plot))
        width = 0.35
        ax.barh(index - width/2, comp_plot["missing_before"], height=width, label="Before", color="#FF6B6B")
        ax.barh(index + width/2, comp_plot["missing_after"], height=width, label="After", color="#4ECDC4")
        ax.set_yticks(index)
        ax.set_yticklabels(comp_plot.index)
        ax.invert_yaxis()
        ax.set_xlabel("Missing %")
        ax.legend()
        ax.set_title("Missing % Before vs After (top features)")
        st.pyplot(fig)

# -----------------------
# 3) Feature Selection (LOS-focused)
# -----------------------
elif page == "Feature Selection (LOS)":
    st.header("🔑 Feature Selection for LOS prediction")

    if "LOS" not in df_cleaned.columns:
        st.error("Kolom 'LOS' tidak ditemukan di dataset cleaned — pastikan dataset berisi kolom 'LOS'.")
    else:
        tab_corr, tab_rf, tab_scatter = st.tabs(["📊 Correlation (with LOS)", "🌲 RF Feature Importance", "📈 Scatterplots"])

        with tab_corr:
            st.subheader("Correlation of numeric features with LOS")
            corr_method = st.radio("Correlation method", ["pearson", "spearman"], horizontal=True)
            # use numeric only to avoid conversion errors
            numeric_only = df_cleaned.select_dtypes(include=[np.number])
            if numeric_only.shape[1] <= 1:
                st.info("Tidak cukup kolom numerik untuk korelasi.")
            else:
                corr_series = numeric_only.corr(method=corr_method)["LOS"].sort_values(ascending=False)
                st.dataframe(corr_series.to_frame("correlation").round(4))

                # top-k barplot
                top_k = st.slider("Top K features to plot", 5, min(30, len(corr_series)), value=12)
                top_corr = corr_series.drop(index="LOS").abs().sort_values(ascending=False).head(top_k)
                fig, ax = plt.subplots(figsize=(6, top_k * 0.4 + 1))
                sns.barplot(x=top_corr.values, y=top_corr.index, palette=palette, ax=ax)
                ax.set_xlabel(f"|Correlation with LOS| ({corr_method})")
                st.pyplot(fig)

                # Heatmap for top correlated features (including LOS)
                heat_k = min(12, len(numeric_only.columns))
                top_features = corr_series.abs().sort_values(ascending=False).head(heat_k).index.tolist()
                heat_df = numeric_only[top_features].corr()
                fig2, ax2 = plt.subplots(figsize=(8,6))
                sns.heatmap(heat_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
                ax2.set_title("Correlation matrix (top features incl. LOS)")
                st.pyplot(fig2)

        with tab_rf:
            st.subheader("Random Forest — Feature Importance (encoded categorical included)")
            # prepare X, y; encode categoricals with get_dummies
            X = df_cleaned.drop(columns=["LOS"])
            y = df_cleaned["LOS"]

            # encode categoricals for RF if present
            X_enc = pd.get_dummies(X, drop_first=True)
            # remove any non-numeric (safeguard)
            X_enc = X_enc.select_dtypes(include=[np.number]).fillna(0)

            if X_enc.shape[1] == 0:
                st.info("Tidak ada feature numeric/encoded untuk RandomForest.")
            else:
                try:
                    rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
                    rf.fit(X_enc, y)
                    imp = pd.Series(rf.feature_importances_, index=X_enc.columns).sort_values(ascending=False)
                    st.dataframe(imp.head(30).to_frame("importance").round(5))

                    fig, ax = plt.subplots(figsize=(8, min(12, len(imp.head(20))) * 0.45 + 1))
                    sns.barplot(x=imp.head(20).values, y=imp.head(20).index, palette=palette, ax=ax)
                    ax.set_title("Top 20 Feature Importances (Random Forest)")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"RandomForest failed: {e}")

        with tab_scatter:
            st.subheader("Interactive scatter: pick feature vs LOS")
            feat_options = [c for c in df_cleaned.columns if c != "LOS"]
            if not feat_options:
                st.info("Tidak ada fitur selain LOS.")
            else:
                sel_feat = st.selectbox("Select feature to plot against LOS", feat_options)
                fig, ax = plt.subplots(figsize=(7,4))
                # If feature is categorical, plot jittered stripplot
                if df_cleaned[sel_feat].dtype == 'object' or df_cleaned[sel_feat].nunique() < 12:
                    sns.stripplot(x=df_cleaned[sel_feat], y=df_cleaned["LOS"], jitter=True, ax=ax, palette=palette)
                    ax.set_xlabel(sel_feat)
                    ax.set_ylabel("LOS")
                else:
                    sns.scatterplot(x=df_cleaned[sel_feat], y=df_cleaned["LOS"], alpha=0.6, ax=ax)
                    ax.set_xlabel(sel_feat)
                    ax.set_ylabel("LOS")
                st.pyplot(fig)

# -----------------------
# 4) Modeling (Placeholder + quick demo)
# -----------------------
elif page == "Modeling":
    st.header("🤖 Modeling — Quick demo & next steps")
    st.markdown("""
    Saat ini tab ini bersifat *scaffold*. Jika ingin, saya bisa:
    - Tambahkan training pipeline (train/test split) untuk LOS (LinearRegression, RandomForest, XGBoost).
    - Tampilkan metric: RMSE, MAE, R².
    - Visualisasi Prediksi vs Aktual dan residual analysis.
    """)

    if "LOS" in df_cleaned.columns:
        st.info("Jika mau demo cepat, tekan tombol 'Run quick RF demo' (fit RF & evaluasi 80/20).")
        if st.button("Run quick RF demo"):
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

            X = df_cleaned.drop(columns=["LOS"])
            y = df_cleaned["LOS"]
            X_enc = pd.get_dummies(X, drop_first=True).select_dtypes(include=[np.number]).fillna(0)

            X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42)
            rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.metric("RMSE", f"{rmse:.4f}")
            st.metric("MAE", f"{mae:.4f}")
            st.metric("R²", f"{r2:.4f}")

            fig, ax = plt.subplots(figsize=(6,5))
            sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Actual LOS")
            ax.set_ylabel("Predicted LOS")
            ax.set_title("Actual vs Predicted (Quick RF demo)")
            st.pyplot(fig)
    else:
        st.info("Kolom LOS tidak tersedia → tidak bisa demo modeling.")

st.markdown("---")
st.caption("Dashboard by Senior Data Analyst — visualisasi Raw vs Cleaned; feature selection fokus ke LOS. Jika mau, saya tambahkan export PPTX/print-ready slide otomatis.")
