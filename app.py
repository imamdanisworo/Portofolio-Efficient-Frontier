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

@st.cache_resource
def get_hf_api():
    return HfApi()

api = get_hf_api()

# Header
st.markdown("<h2 style='text-align:center;'>📈 Ringkasan Saham</h2>", unsafe_allow_html=True)

# Helper
def get_date_from_filename(name):
    try:
        date_part = os.path.splitext(name)[0].split("-")[-1]
        return datetime.strptime(date_part, "%Y%m%d").date()
    except:
        return None

@st.cache_data(show_spinner=False)
def load_excel_from_hf(filename):
    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            token=HF_TOKEN,
            cache_dir="/tmp/huggingface"
        )
        return pd.read_excel(path)
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat {filename}: {e}")
        return None

@st.cache_data(show_spinner=True)
def load_all_data():
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
    xlsx_files = [f for f in files if f.lower().endswith(".xlsx")]

    stock_by_date = {}
    index_series = {}
    filename_by_date = {}

    for file in xlsx_files:
        df = load_excel_from_hf(file)
        if df is None:
            continue
        date = get_date_from_filename(file)
        if not date:
            continue
        if file.startswith("index-"):
            if "Kode Indeks" in df.columns and "Penutupan" in df.columns:
                filtered = df[df["Kode Indeks"].str.lower() == "composite"]
                if not filtered.empty:
                    index_series[date] = filtered.iloc[0]["Penutupan"]
        elif "Kode Saham" in df.columns and "Penutupan" in df.columns:
            filtered = df[["Kode Saham", "Penutupan"]].copy()
            filtered["Tanggal"] = date
            stock_by_date[date] = filtered
            filename_by_date[date] = file

    return stock_by_date, pd.Series(index_series).sort_index(), filename_by_date

# Load on first run
if "data_loaded" not in st.session_state:
    data_by_date, index_series, filename_by_date = load_all_data()
    st.session_state.update({
        "data_by_date": data_by_date,
        "index_series": index_series,
        "filename_by_date": filename_by_date,
        "data_loaded": True
    })

data_by_date = st.session_state["data_by_date"]
index_series = st.session_state["index_series"]
filename_by_date = st.session_state["filename_by_date"]

# Upload Section
st.markdown("### 🔼 Upload Data")

def process_file(file, is_index=False):
    try:
        name_in_repo = f"index-{file.name}" if is_index else file.name
        upload_file(path_or_fileobj=file, path_in_repo=name_in_repo, repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)

        df = pd.read_excel(file)
        date = get_date_from_filename(file.name)
        if not date:
            return False, "Tanggal tidak dikenali"

        if is_index:
            if "Kode Indeks" in df.columns and "Penutupan" in df.columns:
                filtered = df[df["Kode Indeks"].str.lower() == "composite"]
                if not filtered.empty:
                    st.session_state["index_series"][date] = filtered.iloc[0]["Penutupan"]
        else:
            if "Kode Saham" in df.columns and "Penutupan" in df.columns:
                filtered = df[["Kode Saham", "Penutupan"]].copy()
                filtered["Tanggal"] = date
                st.session_state["data_by_date"][date] = filtered
                st.session_state["filename_by_date"][date] = name_in_repo
        return True, f"✅ {file.name} berhasil diunggah"
    except Exception as e:
        return False, f"❌ Gagal unggah {file.name}: {e}"

col1, col2 = st.columns(2)
uploaded_any = False

with col1:
    index_files = st.file_uploader("Upload File Indeks (.xlsx)", type="xlsx", accept_multiple_files=True, key="upload_index")
    if index_files:
        for file in index_files:
            success, msg = process_file(file, is_index=True)
            st.success(msg) if success else st.error(msg)
            uploaded_any = uploaded_any or success

with col2:
    stock_files = st.file_uploader("Upload File Saham (.xlsx)", type="xlsx", accept_multiple_files=True, key="upload_saham")
    if stock_files:
        for file in stock_files:
            success, msg = process_file(file, is_index=False)
            st.success(msg) if success else st.error(msg)
            uploaded_any = uploaded_any or success

if uploaded_any:
    st.cache_data.clear()
    st.rerun()

# Delete All
st.divider()
if st.button("🧹 Hapus Semua Data"):
    try:
        all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
        for file in all_files:
            if file.lower().endswith(".xlsx"):
                delete_file(file, REPO_ID, repo_type="dataset", token=HF_TOKEN)
        st.success("✅ Semua file berhasil dihapus.")
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()
    except Exception as e:
        st.error(str(e))

# Viewer
st.divider()
st.markdown(f"**📄 Jumlah File Saham:** {len(data_by_date)}  &nbsp;&nbsp;|&nbsp;&nbsp; 📄 **Jumlah File Indeks:** {len(index_series)}")

if data_by_date:
    selected_date = st.selectbox("📆 Pilih Tanggal Data", sorted(data_by_date.keys(), reverse=True))
    df_show = data_by_date[selected_date].copy()
    df_show["Penutupan"] = df_show["Penutupan"].apply(lambda x: f"{x:,.0f}")

    if selected_date in index_series:
        st.markdown("#### 📊 Indeks Composite")
        st.dataframe(pd.DataFrame({"Composite": [index_series[selected_date]]}), use_container_width=True)

    st.markdown("#### 📋 Data Saham")
    st.dataframe(df_show, use_container_width=True)

    if st.button("🗑️ Hapus Data Ini"):
        try:
            delete_file(filename_by_date[selected_date], REPO_ID, repo_type="dataset", token=HF_TOKEN)
            st.success("✅ Data berhasil dihapus.")
            st.cache_data.clear()
            st.session_state.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Gagal menghapus: {e}")
