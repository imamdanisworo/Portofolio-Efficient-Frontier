import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download, upload_file, delete_file
from scipy.optimize import minimize

# CONFIGURATION
st.set_page_config(page_title="📈 Ringkasan Saham", layout="wide")
REPO_ID = "imamdanisworo/dbf-storage"
HF_TOKEN = st.secrets["HF_TOKEN"]
api = HfApi()

# HEADER
st.markdown("<h1 style='text-align:center;'>📈 Ringkasan Saham</h1>", unsafe_allow_html=True)

# Helper functions
def get_date_from_filename(name):
    try:
        base = os.path.splitext(name)[0]
        date_part = base.split("-")[-1]
        return datetime.strptime(date_part, "%Y%m%d").date()
    except:
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

def load_data_from_hf():
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
    xlsx_files = [f for f in files if f.lower().endswith(".xlsx")]

    stock_by_date = {}
    index_series = {}
    filename_by_date = {}

    for file in xlsx_files:
        df = load_excel_from_hf(file)
        if df is not None:
            date = get_date_from_filename(file)
            if date:
                if file.startswith("index-"):
                    if "Kode Indeks" in df.columns and "Penutupan" in df.columns:
                        df_filtered = df[df["Kode Indeks"].str.lower() == "composite"]
                        if not df_filtered.empty:
                            index_series[date] = df_filtered.iloc[0]["Penutupan"]
                else:
                    if "Kode Saham" in df.columns and "Penutupan" in df.columns:
                        df_filtered = df[["Kode Saham", "Penutupan"]].copy()
                        df_filtered["Tanggal"] = date
                        stock_by_date[date] = df_filtered
                        filename_by_date[date] = file

    st.session_state.data_by_date = stock_by_date
    st.session_state.index_series = pd.Series(index_series).sort_index()
    st.session_state.filename_by_date = filename_by_date

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

# Tabs
tab1, tab2 = st.tabs(["📂 Manajemen Data", "📊 Analisis Saham"])

# Tab 1: Manajemen Data
with tab1:
    if "data_by_date" not in st.session_state or "index_series" not in st.session_state:
        with st.spinner("📦 Mengambil data dari Hugging Face..."):
            load_data_from_hf()

    data_by_date = st.session_state.get("data_by_date", {})
    filename_by_date = st.session_state.get("filename_by_date", {})
    index_series = st.session_state.get("index_series", pd.Series(dtype=float))

    uploaded_files = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            try:
                if "indeks" in file.name.lower():
                    upload_file(path_or_fileobj=file, path_in_repo=f"index-{file.name}", repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
                    st.success(f"✅ Uploaded Indeks: {file.name}")
                else:
                    upload_file(path_or_fileobj=file, path_in_repo=file.name, repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
                    st.success(f"✅ Uploaded Saham: {file.name}")
            except Exception as e:
                st.error(f"❌ Failed: {file.name} - {e}")
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

    st.markdown(f"📄 **Jumlah File Saham:** {len(data_by_date)}")
    st.markdown(f"📄 **Jumlah File Indeks:** {len(index_series)}")

    if data_by_date:
        selected_date = st.selectbox("📆 Pilih Tanggal", sorted(data_by_date.keys(), reverse=True))
        df_show = data_by_date[selected_date].copy()
        df_show['Penutupan'] = df_show['Penutupan'].apply(lambda x: f"{x:,.0f}")

        if selected_date in index_series:
            st.markdown("#### 📊 Ringkasan Indeks (Composite)")
            st.dataframe(pd.DataFrame({"Composite": [index_series[selected_date]]}), use_container_width=True)

        st.markdown("#### 📋 Data Saham")
        st.dataframe(df_show, use_container_width=True)

        if st.button("🗑️ Hapus Data Ini"):
            delete_file(filename_by_date[selected_date], REPO_ID, repo_type="dataset", token=HF_TOKEN)
            st.success("✅ Dihapus.")
            st.cache_resource.clear()
            st.rerun()

# Tab 2: Analisis Saham
with tab2:
    st.markdown("### 📊 Optimasi Portofolio Saham")

    if "data_by_date" not in st.session_state or "index_series" not in st.session_state:
        with st.spinner("📦 Mengambil data dari Hugging Face..."):
            load_data_from_hf()

    data_by_date = st.session_state.get("data_by_date", {})
    index_series = st.session_state.get("index_series", pd.Series(dtype=float))

    if not data_by_date:
        st.warning("Belum ada data saham.")
    elif index_series.empty:
        st.warning("Belum ada data indeks (composite).")
    else:
        df_all = pd.concat(data_by_date.values(), ignore_index=True)
        df_all = df_all.sort_values(by="Tanggal", ascending=False)

        stocks = sorted(df_all["Kode Saham"].unique())
        selected_stocks = st.multiselect("Pilih Saham", stocks)
        period = st.selectbox("Pilih Periode", [20, 50, 100, 200, 500])
        risk_free_rate = st.number_input("Risk-Free Rate (per tahun, %)", value=5.0) / 100

        if selected_stocks and st.button("🔍 Analisis"):
            df_filtered = df_all[df_all['Kode Saham'].isin(selected_stocks)]
            df_filtered = df_filtered.sort_values(by="Tanggal", ascending=False)
            df_filtered = df_filtered.groupby("Kode Saham").head(period)

            df_pivot = df_filtered.pivot(index="Tanggal", columns="Kode Saham", values="Penutupan")
            df_pivot = df_pivot.dropna(axis=0)
            df_returns = np.log(df_pivot.sort_index() / df_pivot.sort_index().shift(1)).dropna()

            index_filtered = index_series[index_series.index.isin(df_returns.index)]
            market_returns = np.log(index_filtered / index_filtered.shift(1)).dropna()
            df_returns = df_returns.loc[market_returns.index]

            historical_returns = df_pivot.sort_index().iloc[-1] / df_pivot.sort_index().iloc[0] - 1

            mean_returns = df_returns.mean()
            cov_matrix = df_returns.cov()
            volatility = df_returns.std()
            avg_corr = df_returns.corr().mean()
            sharpe_ratio = (mean_returns - risk_free_rate) / volatility

            # CAPM calculations
            periods_per_year = 252
            daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
            excess_market = market_returns - daily_rf
            beta_results = {}

            for stock in df_returns.columns:
                excess_stock = df_returns[stock] - daily_rf
                cov = np.cov(excess_stock, excess_market)[0][1]
                var = np.var(excess_market)
                beta = cov / var if var != 0 else np.nan
                expected_return = daily_rf + beta * excess_market.mean()
                beta_results[stock] = {
                    "Beta": beta,
                    "CAPM Expected Return": expected_return
                }

            beta_df = pd.DataFrame(beta_results).T

            st.markdown(f"#### 📈 Statistik Saham (Periode: {period} hari)")
            stats_df = pd.DataFrame({
                "Historical Return": historical_returns,
                "Expected Return": mean_returns,
                "Volatility (Risk)": volatility,
                "Sharpe Ratio": sharpe_ratio,
                "Beta": beta_df["Beta"],
                "CAPM Expected Return": beta_df["CAPM Expected Return"]
            })
            stats_df["Avg Correlation"] = avg_corr[stats_df.index]
            st.dataframe(stats_df.style.format({
                "Historical Return": "{:.2%}",
                "Expected Return": "{:.2%}",
                "Volatility (Risk)": "{:.2%}",
                "Sharpe Ratio": "{:.2f}",
                "Beta": "{:.2f}",
                "CAPM Expected Return": "{:.2%}",
                "Avg Correlation": "{:.2f}"
            }), use_container_width=True)

            # ✅ Use CAPM expected return directly
            adj_return_series = beta_df["CAPM Expected Return"]

            # Optimize portfolio
            w_max, w_min, w_opt = optimize_portfolio(adj_return_series.values, cov_matrix.values, risk_free_rate)

            # ✅ Normalize weights to sum exactly 100%
            def normalize_weights(weights):
                weights = np.maximum(weights, 0)  # remove small negatives
                normalized = weights / weights.sum()
                return np.round(normalized, 6)  # precision to 6 decimals

            w_max = normalize_weights(w_max)
            w_min = normalize_weights(w_min)
            w_opt = normalize_weights(w_opt)

            alloc_df = pd.DataFrame({
                "Saham": stats_df.index,
                "📈 Maksimum Return": w_max,
                "🛡️ Minimum Risk": w_min,
                "⚖️ Optimum Return": w_opt
            }).set_index("Saham")

            sum_row = pd.DataFrame(alloc_df.sum()).T
            sum_row.index = ["TOTAL"]
            alloc_df = pd.concat([alloc_df, sum_row])

            st.markdown("#### 🧮 Alokasi Optimal Portofolio (Metode CAPM)")
            st.dataframe(alloc_df.applymap(lambda x: f"{x:.2%}"), use_container_width=True)

            # Portfolio stats for optimum return portfolio
            port_ret = np.dot(w_opt, adj_return_series.values)
            port_vol = np.sqrt(np.dot(w_opt.T, np.dot(cov_matrix.values, w_opt)))
            port_sharpe = (port_ret - risk_free_rate) / port_vol if port_vol != 0 else 0

            st.markdown("#### 📌 Statistik Portofolio Optimal")
            st.metric("Expected Return", f"{port_ret:.2%}")
            st.metric("Volatility (Risk)", f"{port_vol:.2%}")
            st.metric("Sharpe Ratio", f"{port_sharpe:.2f}")
