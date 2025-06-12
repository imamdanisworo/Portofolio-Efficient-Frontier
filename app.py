import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download, upload_file, delete_file

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

# 💾 Tab: Manajemen Data
st.markdown("### 📂 Manajemen Data")

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
