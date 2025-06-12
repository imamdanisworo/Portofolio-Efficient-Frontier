import streamlit as st

# Set page config before anything else
st.set_page_config(page_title="📈 Ringkasan Saham", layout="wide")

import pandas as pd
import os
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download, upload_file, delete_file
from scipy.optimize import minimize

# === CONFIG ===
REPO_ID = "imamdanisworo/dbf-storage"
HF_TOKEN = st.secrets["HF_TOKEN"]
api = HfApi()

st.markdown("<h1 style='text-align:center;'>📈 Ringkasan Saham</h1>", unsafe_allow_html=True)

# === Helper Functions ===
def get_date_from_filename(name):
    try:
        base = os.path.splitext(name)[0]
        date_part = base.split("-")[-1]
        return datetime.strptime(date_part, "%Y%m%d").date()
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_excel_from_hf(filename):
    try:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            token=HF_TOKEN,
            cache_dir="/tmp/huggingface"
        )
        return pd.read_excel(local_path)
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat: {filename} - {e}")
        return None

# === Load All Data ===
def load_data_from_hf():
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
    xlsx_files = [f for f in files if f.lower().endswith(".xlsx")]

    stock_by_date = {}
    index_by_date = {}
    filename_by_date = {}

    progress_bar = st.progress(0)
    status = st.empty()

    for i, file in enumerate(xlsx_files):
        status.text(f"📥 Memuat: {file}")
        df = load_excel_from_hf(file)
        if df is not None:
            date = get_date_from_filename(file)
            if date and 'Kode Saham' in df.columns and 'Penutupan' in df.columns:
                df_filtered = df[['Kode Saham', 'Penutupan']].copy()
                df_filtered['Tanggal'] = date
                if file.startswith("index-"):
                    index_by_date[date] = df_filtered
                else:
                    stock_by_date[date] = df_filtered
                    filename_by_date[date] = file
        progress_bar.progress((i + 1) / len(xlsx_files))

    status.success("✅ Semua file berhasil dimuat.")
    st.session_state.data_by_date = stock_by_date
    st.session_state.index_by_date = index_by_date
    st.session_state.filename_by_date = filename_by_date

# === Portfolio Optimization ===
def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)

    def max_return(weights):
        return -weights @ mean_returns

    def min_volatility(weights):
        return weights.T @ cov_matrix @ weights

    def neg_sharpe(weights):
        ret = weights @ mean_returns
        vol = (weights.T @ cov_matrix @ weights) ** 0.5
        return -(ret - risk_free_rate) / vol if vol != 0 else float("inf")

    constraints = {"type": "eq", "fun": lambda x: sum(x) - 1}
    bounds = [(0.0, 0.3)] * num_assets
    init_guess = [1 / num_assets] * num_assets

    max_ret = minimize(max_return, init_guess, bounds=bounds, constraints=constraints)
    min_risk = minimize(min_volatility, init_guess, bounds=bounds, constraints=constraints)
    opt_sharpe = minimize(neg_sharpe, init_guess, bounds=bounds, constraints=constraints)

    return max_ret.x, min_risk.x, opt_sharpe.x

# === Tabs ===
tab1, tab2 = st.tabs(["📂 Manajemen Data", "📊 Analisis Saham"])

# === Tab 1: Manajemen Data ===
with tab1:
    st.markdown("### 📂 Upload Data")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_stocks = st.file_uploader("Upload Data Saham (.xlsx)", type=["xlsx"], accept_multiple_files=True, key="stock")
        if uploaded_stocks:
            for file in uploaded_stocks:
                try:
                    upload_file(
                        path_or_fileobj=file,
                        path_in_repo=file.name,
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        token=HF_TOKEN
                    )
                    st.success(f"✅ Uploaded Saham: {file.name}")
                except Exception as e:
                    st.error(f"❌ Gagal Upload Saham: {file.name} - {e}")
            st.cache_resource.clear()
            st.rerun()

    with col2:
        uploaded_index = st.file_uploader("Upload Data Indeks Pasar (.xlsx)", type=["xlsx"], accept_multiple_files=True, key="index")
        if uploaded_index:
            for file in uploaded_index:
                try:
                    new_name = f"index-{file.name}"
                    upload_file(
                        path_or_fileobj=file,
                        path_in_repo=new_name,
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        token=HF_TOKEN
                    )
                    st.success(f"✅ Uploaded Indeks: {file.name}")
                except Exception as e:
                    st.error(f"❌ Gagal Upload Indeks: {file.name} - {e}")
            st.cache_resource.clear()
            st.rerun()

    if st.button("🧹 Hapus Semua Data"):
        try:
            all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
            for file in all_files:
                if file.lower().endswith(".xlsx"):
                    delete_file(file, REPO_ID, repo_type="dataset", token=HF_TOKEN)
            st.success("✅ Semua file berhasil dihapus.")
            st.cache_resource.clear()
            st.rerun()
        except Exception as e:
            st.error(str(e))

    if "data_by_date" not in st.session_state:
        with st.spinner("📦 Mengambil data dari Hugging Face..."):
            load_data_from_hf()

    data_by_date = st.session_state.get("data_by_date", {})
    filename_by_date = st.session_state.get("filename_by_date", {})

    if data_by_date:
        selected_date = st.selectbox("📆 Pilih Tanggal", sorted(data_by_date.keys(), reverse=True))
        df_show = data_by_date[selected_date].copy()
        df_show['Penutupan'] = df_show['Penutupan'].apply(lambda x: f"{x:,.0f}")
        st.dataframe(df_show, use_container_width=True)

        if st.button("🗑️ Hapus Data Ini"):
            delete_file(filename_by_date[selected_date], REPO_ID, repo_type="dataset", token=HF_TOKEN)
            st.success("✅ Dihapus.")
            st.cache_resource.clear()
            st.rerun()

# === Tab 2: Analisis Saham ===
with tab2:
    st.markdown("### 📊 Optimasi Portofolio Saham")
    data_by_date = st.session_state.get("data_by_date", {})

    if not data_by_date:
        st.warning("Belum ada data.")
    else:
        df_all = pd.concat(data_by_date.values(), ignore_index=True)
        df_all = df_all.sort_values(by="Tanggal", ascending=False)

        stocks = sorted(df_all["Kode Saham"].unique())
        selected_stocks = st.multiselect("Pilih Saham", stocks)
        period = st.selectbox("Pilih Periode", [20, 50, 100, 200, 500])
        risk_free_rate = st.number_input("Risk-Free Rate (per tahun, %)", value=0.0) / 100

        if selected_stocks and st.button("🔍 Analisis"):
            df_filtered = df_all[df_all['Kode Saham'].isin(selected_stocks)]
            recent_dates = sorted(df_filtered['Tanggal'].unique(), reverse=True)[:period]
            df_recent = df_filtered[df_filtered['Tanggal'].isin(recent_dates)]

            df_pivot = df_recent.pivot(index="Tanggal", columns="Kode Saham", values="Penutupan")
            df_returns = df_pivot.sort_index().pct_change().dropna()

            historical_returns = df_pivot.sort_index().iloc[-1] / df_pivot.sort_index().iloc[0] - 1
            mean_returns = df_returns.mean() * period
            cov_matrix = df_returns.cov() * period
            volatility = df_returns.std() * (period ** 0.5)
            sharpe_ratio = (mean_returns - risk_free_rate) / volatility

            st.markdown(f"#### 📈 Statistik Saham (Periode: {period} hari)")
            stats_df = pd.DataFrame({
                "Historical Return": historical_returns,
                "Expected Return": mean_returns,
                "Volatility (Risk)": volatility,
                "Sharpe Ratio": sharpe_ratio
            })
            st.dataframe(stats_df.style.format({
                "Historical Return": "{:.2%}",
                "Expected Return": "{:.2%}",
                "Volatility (Risk)": "{:.2%}",
                "Sharpe Ratio": "{:.2f}"
            }), use_container_width=True)

            st.markdown("#### 🔗 Korelasi Antar Saham")
            st.dataframe(df_returns.corr().style.format("{:.2f}"), use_container_width=True)

            w_max, w_min, w_opt = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate)
            alloc_df = pd.DataFrame({
                "Saham": selected_stocks,
                "📈 Maksimum Return": w_max,
                "🛡️ Minimum Risk": w_min,
                "⚖️ Optimum Return": w_opt
            }).set_index("Saham")
            sum_row = pd.DataFrame(alloc_df.sum()).T
            sum_row.index = ["TOTAL"]
            alloc_df = pd.concat([alloc_df, sum_row])
            st.markdown("#### 🧮 Alokasi Optimal Portofolio (Metode Historis)")
            st.dataframe(alloc_df.applymap(lambda x: f"{x:.2%}"), use_container_width=True)
