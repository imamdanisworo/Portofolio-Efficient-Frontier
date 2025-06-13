import streamlit as st
import pandas as pd
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

# Utility Functions
def get_date_from_filename(name):
    try:
        date_part = os.path.splitext(name)[0].split("-")[-1]
        return datetime.strptime(date_part, "%Y%m%d").date()
    except:
        return None

@st.cache_data(show_spinner=False)
def load_excel_from_hf(filename):
    try:
        path = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset", token=HF_TOKEN, cache_dir="/tmp/huggingface")
        return pd.read_excel(path)
    except:
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
                filtered = df[df["Kode Indeks"].astype(str).str.lower() == "composite"]
                if not filtered.empty:
                    index_series[date] = filtered.iloc[0]["Penutupan"]
        elif "Kode Saham" in df.columns and "Penutupan" in df.columns:
            filtered = df[["Kode Saham", "Penutupan"]].copy()
            filtered["Tanggal"] = date
            stock_by_date[date] = filtered
            filename_by_date[date] = file

    return stock_by_date, pd.Series(index_series).sort_index(), filename_by_date

# Session State Init
if "initialized" not in st.session_state:
    data_by_date, index_series, filename_by_date = load_all_data()
    st.session_state.update({
        "data_by_date": data_by_date,
        "index_series": index_series,
        "filename_by_date": filename_by_date,
        "initialized": True
    })

data_by_date = st.session_state["data_by_date"]
index_series = st.session_state["index_series"]
filename_by_date = st.session_state["filename_by_date"]

# UI Tabs
tabs = st.tabs(["📤 Upload Data", "📊 Lihat Data", "🧹 Manajemen Data"])

with tabs[0]:
    st.markdown("### Upload File Excel Saham dan Indeks")
    col1, col2 = st.columns(2)
    uploaded_any = False

    def validate_file(file, is_index=False):
        try:
            df = pd.read_excel(file)
            date = get_date_from_filename(file.name)
            if not date:
                return False, "❌ Format nama file salah", None, None

            if is_index:
                if "Kode Indeks" in df.columns and "Penutupan" in df.columns:
                    filtered = df[df["Kode Indeks"].astype(str).str.lower() == "composite"]
                    if filtered.empty:
                        return False, "❌ Tidak ada baris 'composite'", None, None
                    return True, "", date, filtered.iloc[0]["Penutupan"]
                else:
                    return False, "❌ Kolom tidak lengkap", None, None
            else:
                if not all(col in df.columns for col in ["Kode Saham", "Penutupan"]):
                    return False, "❌ Kolom tidak lengkap", None, None
                df_filtered = df[["Kode Saham", "Penutupan"]].copy()
                df_filtered["Tanggal"] = date
                return True, "", date, df_filtered

        except Exception as e:
            return False, f"❌ Gagal baca: {e}", None, None

    def process_files(files, is_index=False):
        messages = []
        new_index_data = {}
        new_stock_data = {}
        new_filenames = {}
        total = len(files)

        for i, file in enumerate(files):
            valid, msg, date, result = validate_file(file, is_index)
            if not valid:
                messages.append(f"{file.name}: {msg}")
                continue
            try:
                name_in_repo = f"index-{file.name}" if is_index else file.name
                upload_file(file, name_in_repo, REPO_ID, repo_type="dataset", token=HF_TOKEN)
                if is_index:
                    new_index_data[date] = result
                else:
                    new_stock_data[date] = result
                    new_filenames[date] = name_in_repo
                messages.append(f"✅ {file.name} diunggah")
            except Exception as e:
                messages.append(f"{file.name}: ❌ Gagal unggah: {e}")

        if is_index:
            st.session_state["index_series"].update(new_index_data)
        else:
            st.session_state["data_by_date"].update(new_stock_data)
            st.session_state["filename_by_date"].update(new_filenames)

        return messages, bool(new_index_data or new_stock_data)

    with col1:
        stock_files = st.file_uploader("Upload Saham (.xlsx)", type="xlsx", accept_multiple_files=True, key="stock_upload")
        if stock_files:
            msgs, changed = process_files(stock_files, is_index=False)
            for msg in msgs:
                st.toast(msg)
            uploaded_any = uploaded_any or changed

    with col2:
        index_files = st.file_uploader("Upload Indeks (.xlsx)", type="xlsx", accept_multiple_files=True, key="index_upload")
        if index_files:
            msgs, changed = process_files(index_files, is_index=True)
            for msg in msgs:
                st.toast(msg)
            uploaded_any = uploaded_any or changed

    if uploaded_any:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.pop("initialized", None)
        st.rerun()

with tabs[1]:
    st.markdown("### Data Saham dan Indeks")
    if data_by_date:
        sorted_dates = sorted(data_by_date.keys(), reverse=True)
        labels = [d.strftime("%d-%b-%Y") for d in sorted_dates]
        selected = st.selectbox("Pilih Tanggal", labels)
        date = sorted_dates[labels.index(selected)]

        st.subheader("📋 Data Saham")
        df = data_by_date[date].copy()
        df["Penutupan"] = df["Penutupan"].apply(lambda x: f"{x:,.0f}")
        st.dataframe(df, use_container_width=True)

        if date in index_series:
            st.subheader("📊 Indeks Composite")
            st.metric("Nilai Composite", f"{index_series[date]:,.0f}")

with tabs[2]:
    st.markdown("### Manajemen Data")
    if st.button("🧹 Hapus Semua Data"):
        with st.spinner("Menghapus semua file..."):
            try:
                files = api.list_repo_files(REPO_ID, repo_type="dataset", token=HF_TOKEN)
                for file in files:
                    if file.endswith(".xlsx"):
                        delete_file(file, REPO_ID, repo_type="dataset", token=HF_TOKEN)
                st.success("✅ Semua file dihapus")
                st.cache_data.clear()
                st.cache_resource.clear()
                st.session_state.clear()
                st.rerun()
            except Exception as e:
                st.error(str(e))
