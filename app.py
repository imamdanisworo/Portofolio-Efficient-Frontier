import streamlit as st
import pandas as pd
import os
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download, upload_file, delete_file

# === CONFIG ===
REPO_ID = "imamdanisworo/dbf-storage"
HF_TOKEN = st.secrets["HF_TOKEN"]
api = HfApi()

st.set_page_config(page_title="📈 Ringkasan Saham", layout="wide")
st.title("📈 Ringkasan Saham - Kode & Penutupan")

# === Helper: Extract date from filename
def get_date_from_filename(name):
    try:
        base = os.path.splitext(name)[0]
        date_part = base.split("-")[-1]
        return datetime.strptime(date_part, "%Y%m%d").date()
    except Exception:
        return None

# === Load Data from Hugging Face
def load_existing_files():
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
    xlsx_files = [f for f in files if f.lower().endswith(".xlsx")]

    data_by_date = {}
    filename_by_date = {}

    for file in xlsx_files:
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=file,
                repo_type="dataset",
                token=HF_TOKEN,
                local_dir="/tmp",
                local_dir_use_symlinks=False
            )
            df = pd.read_excel(local_path)
            date = get_date_from_filename(file)
            if date and 'Kode Saham' in df.columns and 'Penutupan' in df.columns:
                df_filtered = df[['Kode Saham', 'Penutupan']].copy()
                df_filtered['Tanggal'] = date
                data_by_date[date] = df_filtered
                filename_by_date[date] = file
        except Exception as e:
            st.warning(f"Gagal memuat: {file} - {e}")

    return data_by_date, filename_by_date

# === Upload Section ===
st.header("⬆️ Upload File Excel")

uploaded_files = st.file_uploader("Pilih file Excel (.xlsx)", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        try:
            upload_file(
                path_or_fileobj=file,
                path_in_repo=file.name,
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN
            )
            st.success(f"✅ Uploaded: {file.name}")
        except Exception as e:
            st.error(f"❌ Gagal upload: {file.name} — {e}")
    st.rerun()

# === Refresh Button ===
if st.button("🔄 Refresh Data"):
    st.rerun()

# === Bulk Delete Section ===
st.markdown("### ⚠️ Hapus Semua Data")

if st.button("🧹 Hapus SELURUH File Excel dari Dataset"):
    try:
        all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
        xlsx_files = [f for f in all_files if f.lower().endswith(".xlsx")]

        for file in xlsx_files:
            delete_file(
                path_in_repo=file,
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN
            )
        st.success("✅ Semua file Excel berhasil dihapus.")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Gagal menghapus semua file: {e}")

# === Load and Display Section ===
st.header("📅 Pilih Tanggal dan Lihat Data")

with st.spinner("📦 Mengambil data dari Hugging Face..."):
    data_by_date, filename_by_date = load_existing_files()

if data_by_date:
    all_dates = sorted(data_by_date.keys())
    selected_date = st.selectbox("📆 Pilih Tanggal", options=all_dates)

    if selected_date:
        st.subheader(f"📊 Data Penutupan - {selected_date.strftime('%d %b %Y')}")
        st.dataframe(data_by_date[selected_date], use_container_width=True)

        # Delete Button for selected date
        file_to_delete = filename_by_date[selected_date]
        if st.button(f"🗑️ Hapus Data Tanggal Ini ({file_to_delete})"):
            try:
                delete_file(
                    path_in_repo=file_to_delete,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    token=HF_TOKEN
                )
                st.success(f"✅ Berhasil menghapus: {file_to_delete}")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Gagal menghapus: {e}")
else:
    st.info("ℹ️ Tidak ada data yang tersedia. Upload file Excel untuk mulai.")
