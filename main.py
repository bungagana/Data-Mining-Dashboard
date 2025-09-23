import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Mali Cohort Study — Data Analyst Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# GLOBAL STYLE
st.markdown(
    """
    <style>
    .metric-row { display:flex; gap:12px; }
    .metric-card { flex:1; background:#f7fbff; padding:14px; border-radius:10px; text-align:center; }
    .metric-card h4 { margin:0; font-size:14px; color:#333 }
    .metric-card p { margin:6px 0 0 0; font-size:28px; font-weight:700; color:#111 }
    .stDownloadButton>button{ background-color:#2d6cdf; color:white }
    </style>
    """,
    unsafe_allow_html=True,
)

sns.set_style("whitegrid")

# -----------------------
# Utilities & Caching
# -----------------------
@st.cache_data
def load_data_local(path: str):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Error loading CSV at `{path}`: {e}")
        return None
    return df

@st.cache_data
def preprocess_pipeline(df_raw: pd.DataFrame):
    df = df_raw.copy()

    drop_candidates = [
        'in_treatment', 'adm_WaSt_category',
        'adm_WHZltneg3WAZgtneg3', 'adm_MUAClt110',
        'Screened_hf', 'adm_WHZgtneg3WAZgtneg3', 'Cared_hws'
    ]

    # Detect ID-like columns
    id_like = [
        c for c in df.columns
        if any(x in c.lower() for x in ['anon', 'id', 'patient'])
        and df[c].nunique() == df.shape[0]
    ]

    drop_list = list(dict.fromkeys(drop_candidates + id_like))
    existing_drop = [c for c in drop_list if c in df.columns]
    df = df.drop(columns=existing_drop, errors='ignore')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    if numeric_cols:
        num_imputer = SimpleImputer(strategy="mean")
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    for c in numeric_cols:
        try:
            df[c] = df[c].astype('float64')
        except Exception:
            pass

    scaler = StandardScaler()
    df_scaled = df.copy()
    if numeric_cols:
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

    return df, df_scaled, numeric_cols, cat_cols, existing_drop, id_like

def get_missing_table(df: pd.DataFrame):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    miss_df = pd.DataFrame({
        "Missing Values": missing,
        "Percentage": 100 * missing / len(df)
    }).sort_values("Percentage", ascending=False)
    return miss_df

def to_download_bytes(df: pd.DataFrame, name: str = "cleaned.csv"):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

# -----------------------
# Load data (local)
# -----------------------
df_raw = load_data_local("Mali_Cohort_Study.csv")
if df_raw is None:
    st.error("Gagal load dataset `Mali_Cohort_Study.csv` dari lokal.")
    st.stop()

# Preprocess
df_cleaned, df_scaled, numeric_cols, cat_cols, dropped_columns, id_like_cols = preprocess_pipeline(df_raw)

# -----------------------
# Header + Metrics
# -----------------------
st.title("Mali Cohort Study — Data Analyst Dashboard")
st.markdown("Tampilkan kondisi **raw** (sebelum preprocessing) dan kondisi **cleaned** (setelah preprocessing).")

raw_rows, raw_cols = df_raw.shape
raw_missing_cols = (df_raw.isnull().sum() > 0).sum()

clean_rows, clean_cols = df_cleaned.shape
clean_missing_cols = (df_cleaned.isnull().sum() > 0).sum()

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("<div class='metric-row'>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-card'><h4>Raw Data — Rows</h4><p>{raw_rows}</p></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-card'><h4>Raw Data — Columns</h4><p>{raw_cols}</p></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-card'><h4>Raw — Columns with Missing</h4><p>{raw_missing_cols}</p></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-row'>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-card' style='background:#e8fff2'><h4>Cleaned — Rows</h4><p>{clean_rows}</p></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-card' style='background:#e8fff2'><h4>Cleaned — Columns</h4><p>{clean_cols}</p></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-card' style='background:#e8fff2'><h4>Cleaned — Columns with Missing</h4><p>{clean_missing_cols}</p></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# -----------------------
# Sidebar navigation
# -----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Pendahuluan",  # <- tambahkan ini
    "EDA (Raw Data)",
    "Preprocessing",
    "Feature Selection (LOS)",
    "Modeling"
])


st.sidebar.markdown("---")
st.sidebar.markdown("Display options")
palette = st.sidebar.selectbox("Seaborn palette", ["viridis", "Set2", "mako", "coolwarm"], index=0)
sns.set_palette(palette)

import streamlit as st
import pandas as pd
import base64

def show_pdf(file_path, width=600, height=800):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'''
    <iframe 
        src="data:application/pdf;base64,{base64_pdf}" 
        width="{width}" 
        height="{height}" 
        type="application/pdf" 
        style="border:none;">
    </iframe>'''
    return pdf_display

if page == "Pendahuluan":
    st.title("Pendahuluan")
    
    # Header utama
    st.markdown("""
    ### Preprocessing & Exploratory Data Analysis (EDA) pada Dataset LOS & Status Gizi Anak  
    **Sumber Dataset:** *Malnutrition Cohort Study (2019–2024)*

    **Anggota Tim:**  
    - Anggun Dwi Rizkika  
    - Bunga Laelatul Muna  
    - Wan Sabrina Mayzura

    **Deskripsi:**  
    Notebook ini mencakup proses preprocessing (pembersihan data, imputasi missing values, normalisasi fitur, dan deteksi outlier) serta EDA (analisis distribusi LOS, demografi, status gizi, missed visits, dan korelasi variabel).  
    Tujuan utama analisis adalah memahami pola **lama rawat inap (Length of Stay / LOS)** dan faktor-faktor **klinis serta administratif** yang memengaruhinya.
    """)

    st.markdown("---")

    # Dataset & Notebook
    st.markdown("### Sumber Dataset & Notebook")
    st.markdown("""
    - Dataset Zenodo: [https://zenodo.org/records/7343363](https://zenodo.org/records/7343363)  
    - Notebook Eksplorasi Awal (Kaggle): [https://www.kaggle.com/code/bungalaelatulmuna/cohor](https://www.kaggle.com/code/bungalaelatulmuna/cohor)
    """)

    st.markdown("---")

    # Referensi artikel: 2 kolom sejajar
    st.markdown("### Referensi Artikel Penelitian")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Artikel Utama")
        st.markdown("*Implementing Digital Community‐Based Management of Malnutrition*  \n_Journal of Biomedical Informatics_")
        st.markdown(show_pdf("main.pdf", width=500, height=650), unsafe_allow_html=True)

    with col2:
        st.subheader("Artikel Pembanding")
        st.markdown("*Children Living with Disabilities Are Absent From Severe Malnutrition Guidelines*  \n_MDPI Nutrients_")
        st.markdown(show_pdf("nutrients.pdf", width=500, height=650), unsafe_allow_html=True)

    st.markdown("---")

    # Dokumentasi kolom dataset
    st.markdown("### Dokumentasi Kolom Dataset")

    data_pendahuluan = [
        ("AnonID", "Identifikasi unik untuk setiap peserta (anonim)."),
        ("adm_category", "Kategori penerimaan, menggambarkan tingkat keparahan atau klasifikasi malnutrisi saat penerimaan."),
        ("adm_referral", "Metode rujukan, menunjukkan bagaimana anak dirujuk untuk perawatan."),
        ("adm_site", "Tempat penerimaan perawatan."),
        ("adm_sex", "Jenis kelamin anak."),
        ("adm_age", "Usia anak pada saat penerimaan (bulan)."),
        ("age_cat_1", "Kategorisasi usia berdasarkan rentang tertentu."),
        ("age_cat_2", "Kategorisasi usia berdasarkan rentang usia lainnya."),
        ("adm_muac", "Lingkar lengan atas tengah (MUAC) saat penerimaan (mm)."),
        ("adm_kg", "Berat anak saat penerimaan (kg)."),
        ("missed_1_visit", "Anak melewatkan satu kunjungan."),
        ("missed_2_more_visit", "Anak melewatkan lebih dari satu kunjungan."),
        ("missed_1_visit_cured", "Anak melewatkan satu kunjungan namun sembuh."),
        ("missed_2_more_visit_cured", "Anak melewatkan >2 kunjungan namun sembuh."),
        ("Screened_hf", "Disaring di fasilitas kesehatan."),
        ("Screened_chw_chv", "Disaring oleh petugas komunitas (CHW/CHV)."),
        ("Screened_cg_fm", "Disaring oleh pengasuh/keluarga."),
        ("Cared_hf", "Dirawat di fasilitas kesehatan."),
        ("Cared_hws", "Dirawat oleh petugas non-formal."),
        ("Child's length or height at admission", "Panjang/tinggi anak saat masuk (cm)."),
        ("Weight-for-height Z-score (WHZ)", "Z-score berat terhadap tinggi."),
        ("Height-for-age Z-score (HAZ)", "Z-score tinggi terhadap usia."),
        ("Weight-for-age Z-score (WAZ)", "Z-score berat terhadap usia."),
        ("Treatment status at exit", "Status anak saat keluar."),
        ("Treatment site categorization", "Kategori tempat perawatan."),
        ("Nutritional status details (edema presence)", "Adanya edema saat masuk."),
        ("Treatment regimen details (e.g., RUTF consumption)", "Detil rejimen perawatan seperti RUTF."),
        ("Anthropometric measurements throughout treatment", "Data ukuran tubuh selama perawatan."),
        ("Subgroup tracking", "Klasifikasi subkelompok (misalnya usia <24 bulan)."),
        ("Community-based care indicators", "Indikator perawatan komunitas.")
    ]

    df_pendahuluan = pd.DataFrame(data_pendahuluan, columns=["Kolom", "Penjelasan"])
    st.dataframe(df_pendahuluan, use_container_width=True)

# -----------------------
# 1) EDA (Raw Data)
# -----------------------
if page == "EDA (Raw Data)":
    st.header("Exploratory Data Analysis — Raw Data (Before Preprocessing)")

    tab_overview, tab_missing, tab_dist, tab_bivariate, tab_outcome = st.tabs([
        "Overview", "Missing Values", "Univariate Distributions",
        "Bivariate / vs (Notebook-like)", "Outcome"
    ])

    with tab_overview:
        st.subheader("Preview (first 20 rows)")
        st.dataframe(df_raw.head(20))
        st.markdown("**Columns & dtypes**")
        dtype_table = (
            pd.DataFrame(df_raw.dtypes, columns=["dtype"])
            .reset_index()
            .rename(columns={"index":"column"})
        )
        st.dataframe(dtype_table)

    with tab_missing:
        st.subheader("Missing values overview")
        miss_raw = get_missing_table(df_raw)
        if miss_raw.empty:
            st.success("No missing values detected in raw data.")
        else:
            st.dataframe(miss_raw.head(200))
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(
                x=miss_raw.reset_index()["index"],
                y="Percentage",
                data=miss_raw.reset_index(),
                palette=palette,
                ax=ax
            )
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
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df_raw[sel_num].dropna(), bins=bins, kde=True, ax=ax)
            ax.set_title(f"{sel_num} (raw)")
            st.pyplot(fig)

    with tab_bivariate:
        st.subheader("Bivariate plots: Column vs Column")
        num_options = df_raw.select_dtypes(include=np.number).columns.tolist()
        if len(num_options) < 2:
            st.info("Tidak cukup kolom numerik untuk bivariate plot.")
        else:
            col_x = st.selectbox("X (numeric)", num_options, index=0)
            col_y = st.selectbox("Y (numeric)", [c for c in num_options if c != col_x], index=0)
            kind = st.selectbox("Plot type", ["scatter", "hex", "kde"], index=0)

            fig, ax = plt.subplots(figsize=(7, 5))
            if kind == 'scatter':
                sns.scatterplot(x=df_raw[col_x], y=df_raw[col_y], alpha=0.6, ax=ax)
            elif kind == 'hex':
                ax.hexbin(
                    df_raw[col_x].dropna(),
                    df_raw[col_y].dropna(),
                    gridsize=40,
                    cmap="Blues"
                )
            else:  # kde
                try:
                    sns.kdeplot(
                        x=df_raw[col_x].dropna(),
                        y=df_raw[col_y].dropna(),
                        cmap='Blues',
                        shade=True,
                        ax=ax
                    )
                except Exception:
                    st.info("KDE plot gagal — coba scatter atau hex.")
            ax.set_xlabel(col_x)
            ax.set_ylabel(col_y)
            st.pyplot(fig)

    with tab_outcome:
        st.subheader("Outcome distribution (if 'status' exists)")
        if "status" in df_raw.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(
                x="status",
                data=df_raw,
                order=df_raw['status'].value_counts().index,
                ax=ax
            )
            ax.set_title("status (raw)")
            st.pyplot(fig)
        else:
            st.info("'status' column tidak ditemukan di raw data.")

# -----------------------
# 2) Preprocessing
# -----------------------
elif page == "Preprocessing":
    st.header("Preprocessing (What was done & results)")

    tab_steps, tab_cleaned, tab_dropped, tab_compare = st.tabs([
        "Steps", "Cleaned Data", "Dropped Features", "Missing Before vs After"
    ])

    with tab_steps:
        st.subheader("Preprocessing steps applied automatically")
        st.markdown("""
        1. Drop kolom yang secara manual ditentukan (irrelevan / kosong banyak).  
        2. Drop kolom seperti ID / anon / patient jika tiap baris unik.  
        3. Ganti `inf` / `-inf` ke `NaN`.  
        4. Imputasi nilai hilang:  
           - numerik → rata-rata  
           - kategorikal → modus  
        5. Konversi numerik menjadi `float64`.  
        6. Standarisasi kolom numerik dengan `StandardScaler`.
        """)

    with tab_cleaned:
        st.subheader("Preview cleaned dataset")
        st.dataframe(df_cleaned.head(20))
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df_cleaned.shape[0])
        c2.metric("Columns", df_cleaned.shape[1])
        c3.metric("Numeric cols", len(numeric_cols))

        st.markdown("Download cleaned dataset (CSV)")
        buf = to_download_bytes(df_cleaned, name="cleaned_mali_cohort.csv")
        st.download_button(
            "Download cleaned CSV",
            data=buf,
            file_name="cleaned_mali_cohort.csv",
            mime="text/csv"
        )

    with tab_dropped:
        st.subheader("Dropped / Irrelevant Features")
        if dropped_columns:
            reasons = []
            for c in dropped_columns:
                reason = "explicit drop"
                if c in id_like_cols:
                    reason = "ID-like column"
                reasons.append((c, reason))
            dropped_df = pd.DataFrame(reasons, columns=["column", "reason"])
            st.dataframe(dropped_df)
        else:
            st.info("Tidak ada kolom yang di-drop secara otomatis.")

    with tab_compare:
        st.subheader("Missing values: Before vs After")
        miss_before = get_missing_table(df_raw)
        miss_after = get_missing_table(df_cleaned)
        keys = list(pd.Index(miss_before.index).union(miss_after.index))[:40]
        comp = pd.DataFrame({
            "missing_before": miss_before["Percentage"].reindex(keys).fillna(0),
            "missing_after": miss_after["Percentage"].reindex(keys).fillna(0)
        }).fillna(0)
        st.dataframe(comp.sort_values("missing_before", ascending=False).head(40))

        fig, ax = plt.subplots(figsize=(10, 6))
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
        ax.set_title("Missing % Before vs After (Top Features)")
        st.pyplot(fig)


# -----------------------
# 3a) Feature Selection (LOS)
# -----------------------
elif page == "Feature Selection (LOS)":
    st.header("Feature Selection for LOS Prediction")

    if "LOS" not in df_cleaned.columns:
        st.error("Kolom `LOS` tidak ditemukan pada dataset yang sudah di-cleaned.")
    else:
        tab_corr, tab_rf, tab_scatter, tab_pca = st.tabs([
            "Correlation (with LOS)", 
            "RF Feature Importance", 
            "Scatterplot vs LOS", 
            "PCA (Dimensionality Reduction)"
        ])

        with tab_corr:
            st.subheader("Correlation of numeric features with LOS")
            corr_method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
            numeric_only = df_cleaned.select_dtypes(include=[np.number])
            if numeric_only.shape[1] <= 1:
                st.info("Tidak cukup fitur numerik untuk menghitung korelasi.")
            else:
                corr_series = numeric_only.corr(method=corr_method)["LOS"].sort_values(ascending=False)
                st.dataframe(corr_series.to_frame("correlation").round(4))

                top_k = st.slider("Top K features to show", 5, min(30, len(corr_series)), value=12)
                top_corr = corr_series.drop(index="LOS").abs().sort_values(ascending=False).head(top_k)
                fig, ax = plt.subplots(figsize=(6, top_k * 0.4 + 1))
                sns.barplot(x=top_corr.values, y=top_corr.index, ax=ax)
                ax.set_xlabel(f"|Correlation with LOS| ({corr_method})")
                st.pyplot(fig)

                heat_k = min(12, len(numeric_only.columns))
                top_feats = corr_series.abs().sort_values(ascending=False).head(heat_k).index.tolist()
                heat_df = numeric_only[top_feats].corr()
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                sns.heatmap(heat_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
                ax2.set_title("Correlation matrix (top features incl. LOS)")
                st.pyplot(fig2)

        with tab_rf:
            st.subheader("Random Forest — Feature Importance")
            X = df_cleaned.drop(columns=["LOS"])
            y = df_cleaned["LOS"]
            X_enc = pd.get_dummies(X, drop_first=True)
            X_enc = X_enc.select_dtypes(include=[np.number]).fillna(0)

            if X_enc.shape[1] == 0:
                st.info("Tidak ada fitur numeric/encoded untuk RandomForest.")
            else:
                rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
                rf.fit(X_enc, y)
                imp = pd.Series(rf.feature_importances_, index=X_enc.columns).sort_values(ascending=False)
                st.dataframe(imp.head(100).to_frame("importance").round(5))

                fig, ax = plt.subplots(figsize=(8, min(20, len(imp.head(20))) * 0.45 + 1))
                sns.barplot(x=imp.head(20).values, y=imp.head(20).index, ax=ax)
                ax.set_title("Top 20 Feature Importances (Random Forest)")
                st.pyplot(fig)

        with tab_scatter:
            st.subheader("Scatter / Stripplot: Feature vs LOS")
            feat_options = [c for c in df_cleaned.columns if c != "LOS"]
            if not feat_options:
                st.info("Tidak ada fitur selain LOS.")
            else:
                sel_feat = st.selectbox("Pilih fitur vs LOS", feat_options)
                fig, ax = plt.subplots(figsize=(7, 4))
                if df_cleaned[sel_feat].dtype == object or df_cleaned[sel_feat].nunique() < 12:
                    sns.stripplot(x=df_cleaned[sel_feat], y=df_cleaned["LOS"], jitter=True, ax=ax)
                else:
                    sns.scatterplot(x=df_cleaned[sel_feat], y=df_cleaned["LOS"], alpha=0.6, ax=ax)
                ax.set_xlabel(sel_feat)
                ax.set_ylabel("LOS")
                st.pyplot(fig)

# -----------------------
# 3b) PCA (Dimensionality Reduction)
# -----------------------
        with tab_pca:
            st.subheader("Principal Component Analysis (PCA)")

            from sklearn.decomposition import PCA

            exclude_cols = [
                "LOS", "in_treatment", "adm_WaSt_category",
                "adm_WHZltneg3WAZgtneg3", "adm_MUAClt110", 
                "Screened_hf", "adm_WHZgtneg3WAZgtneg3", 
                "Cared_hws", "AnonID"
            ]

            X = df_scaled.drop(columns=exclude_cols, errors="ignore")
            X = X.select_dtypes(include=[np.number])

            if X.shape[1] < 2:
                st.info("Tidak cukup fitur numerik untuk PCA.")
            else:
                n_comp = st.slider("Jumlah komponen PCA", 2, X.shape[1], 5)
                pca = PCA(n_components=n_comp, random_state=42)
                pcs = pca.fit_transform(X)

                var_ratio = pca.explained_variance_ratio_
                cum_var = np.cumsum(var_ratio)

                st.markdown("### Scree Plot — Cumulative Variance Explained")
                var_df = pd.DataFrame({
                    "PC": [f"PC{i+1}" for i in range(len(var_ratio))],
                    "Explained Variance": var_ratio,
                    "Cumulative Variance": cum_var
                })
                st.dataframe(var_df)

                fig, ax = plt.subplots(figsize=(7,4))
                ax.plot(range(1, len(cum_var)+1), cum_var, marker="o", linestyle="-", color="orange")
                ax.set_xticks(range(1, len(cum_var)+1))
                ax.set_xlabel("Number of Components")
                ax.set_ylabel("Cumulative Explained Variance")
                ax.set_title("Scree Plot (Cumulative Variance)")
                st.pyplot(fig)

                if "LOS" in df_scaled.columns:
                    fig2, ax2 = plt.subplots(figsize=(6,5))
                    sns.scatterplot(
                        x=pcs[:,0], y=pcs[:,1],
                        hue=df_scaled["LOS"],
                        palette="viridis", alpha=0.7, ax=ax2
                    )
                    ax2.set_xlabel("PC1")
                    ax2.set_ylabel("PC2")
                    ax2.set_title("PCA Projection (PC1 vs PC2, colored by LOS)")
                    st.pyplot(fig2)

                loadings = pd.DataFrame(
                    pca.components_.T,
                    columns=[f"PC{i+1}" for i in range(n_comp)],
                    index=X.columns
                )

                st.markdown("### Feature Loadings")
                st.dataframe(loadings.round(4))

                top_k = st.slider("Top K Features untuk Heatmap", 5, loadings.shape[0], 20)
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                sns.heatmap(loadings.iloc[:top_k], annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
                ax3.set_title(f"Heatmap of Feature Loadings (Top {top_k} Features)")
                st.pyplot(fig3)

# -----------------------
# 4) Modeling
# -----------------------
elif page == "Modeling":
    st.header("Modeling — Results Comparison")
    
    # Try to load model evaluation results locally
    model_df = load_data_local("model_evaluation_results.csv")
    if model_df is None:
        st.info("File `model_evaluation_results.csv` tidak ditemukan secara lokal. Silakan taruh di direktori yang sama dengan app.")
    else:
        st.subheader("Tabel hasil model")
        st.dataframe(model_df)

        if {'Scenario', 'Model'}.issubset(model_df.columns):
            pivot = (
                model_df
                .groupby(['Scenario', 'Model'])
                .agg({'RMSE':'mean', 'MAE':'mean', 'R2':'mean'})
                .reset_index()
            )
            st.subheader("Summary (mean metrics per Scenario & Model)")
            st.dataframe(pivot.round(4))

        # Pilih metric untuk dibandingkan
        possible_metrics = [
            c for c in model_df.columns if c.lower() in ['rmse','mae','r2','accuracy']
        ] + [
            c for c in model_df.columns if pd.api.types.is_numeric_dtype(model_df[c])
        ]
        possible_metrics = list(dict.fromkeys(possible_metrics))  # unique

        if possible_metrics:
            metric = st.selectbox("Metric to compare", possible_metrics, index=0)
            fig, ax = plt.subplots(figsize=(8, 4))
            if {'Scenario', 'Model'}.issubset(model_df.columns):
                sns.barplot(data=model_df, x='Scenario', y=metric, hue='Model', ax=ax)
                ax.set_title(f"Perbandingan {metric} per Scenario & Model")
                st.pyplot(fig)

        # Jika ada ≥2 metric numerik, tampilkan scatterplot perbandingan
        metrics_available = [
            c for c in model_df.columns if pd.api.types.is_numeric_dtype(model_df[c])
        ]
        if len(metrics_available) >= 2 and {'Scenario', 'Model'}.issubset(model_df.columns):
            st.subheader("Comparative charts")
            m1 = st.selectbox("Metric 1", metrics_available, index=0)
            m2 = st.selectbox("Metric 2", metrics_available, index=min(1, len(metrics_available)-1))
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.scatterplot(
                data=model_df, x=m1, y=m2,
                hue='Model', style='Scenario', s=100, ax=ax2
            )
            ax2.set_title(f"{m1} vs {m2} (per Model/Scenario)")
            st.pyplot(fig2)

        # Download model results
        summary_buf = to_download_bytes(model_df, name='model_results_summary.csv')
        st.download_button(
            "Download model results",
            data=summary_buf,
            file_name='model_results_summary.csv',
            mime="text/csv"
        )
