import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download, upload_file, delete_file

# CONFIG
st.set_page_config(page_title="📈 Ringkasan Saham", layout="wide")
REPO_ID = "imamdanisworo/dbf-storage"
HF_TOKEN = st.secrets["HF_TOKEN"]
api = HfApi()

# Header
st.markdown("<h2 style='text-align:center;'>📈 Ringkasan Saham</h2>", unsafe_allow_html=True)

# Helpers
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
        st.warning(f"⚠️ Gagal memuat {filename}: {e}")
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

# Load data initially
if "data_by_date" not in st.session_state or "index_series" not in st.session_state:
    with st.spinner("📦 Memuat data dari Hugging Face..."):
        load_data_from_hf()

data_by_date = st.session_state.get("data_by_date", {})
index_series = st.session_state.get("index_series", pd.Series(dtype=float))
filename_by_date = st.session_state.get("filename_by_date", {})

# Upload Section
st.markdown("### 🔼 Upload Data")

col1, col2 = st.columns(2)
with col1:
    index_files = st.file_uploader("Upload File Indeks (.xlsx)", type=["xlsx"], accept_multiple_files=True, key="index")
    if index_files:
        for file in index_files:
            try:
                upload_file(path_or_fileobj=file, path_in_repo=f"index-{file.name}", repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
                st.success(f"✅ Index uploaded: {file.name}")
            except Exception as e:
                st.error(f"❌ Gagal upload: {file.name} - {e}")
        st.cache_resource.clear()
        st.rerun()

with col2:
    stock_files = st.file_uploader("Upload File Saham (.xlsx)", type=["xlsx"], accept_multiple_files=True, key="saham")
    if stock_files:
        for file in stock_files:
            try:
                upload_file(path_or_fileobj=file, path_in_repo=file.name, repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
                st.success(f"✅ Saham uploaded: {file.name}")
            except Exception as e:
                st.error(f"❌ Gagal upload: {file.name} - {e}")
        st.cache_resource.clear()
        st.rerun()

# Delete All Button
st.divider()
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

# Summary & Viewer
st.divider()
st.markdown(f"**📄 Jumlah File Saham:** {len(data_by_date)}  &nbsp;&nbsp;|&nbsp;&nbsp; 📄 **Jumlah File Indeks:** {len(index_series)}")

if data_by_date:
    selected_date = st.selectbox("📆 Pilih Tanggal Data", sorted(data_by_date.keys(), reverse=True))
    df_show = data_by_date[selected_date].copy()
    df_show['Penutupan'] = df_show['Penutupan'].apply(lambda x: f"{x:,.0f}")

    if selected_date in index_series:
        st.markdown("#### 📊 Indeks Composite")
        st.dataframe(pd.DataFrame({"Composite": [index_series[selected_date]]}), use_container_width=True)

    st.markdown("#### 📋 Data Saham")
    st.dataframe(df_show, use_container_width=True)

    if st.button("🗑️ Hapus Data Ini"):
        try:
            delete_file(filename_by_date[selected_date], REPO_ID, repo_type="dataset", token=HF_TOKEN)
            st.success("✅ Data berhasil dihapus.")
            st.cache_resource.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Gagal menghapus: {e}")
