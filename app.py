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

# === Helper ===
def get_date_from_filename(name):
    try:
        base = os.path.splitext(name)[0]
        date_part = base.split("-")[-1]
        return datetime.strptime(date_part, "%Y%m%d").date()
    except Exception:
        return None

def load_data_from_hf():
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

    # Cache in session
    st.session_state.data_by_date = data_by_date
    st.session_state.filename_by_date = filename_by_date

# === Upload Section ===
st.header("⬆️ Upload File Excel")

uploaded_files = st.file_uploader("Pilih file Excel (.xlsx)", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    st.markdown("### 🚀 Proses Upload Dimulai...")
    success_count = 0
    fail_count = 0
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)

    for i, file in enumerate(uploaded_files):
        try:
            upload_file(
                path_or_fileobj=file,
                path_in_repo=file.name,
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN
            )
            st.success(f"✅ {file.name} berhasil diunggah.")
            success_count += 1
        except Exception as e:
            st.error(f"❌ {file.name} gagal: {e}")
            fail_count += 1
        progress_bar.progress((i + 1) / total_files)

    st.success(f"✅ Selesai! {success_count} berhasil, {fail_count} gagal.")

    if st.button("🔄 Selesai & Muat Ulang Data"):
        st.session_state.pop("data_by_date", None)
        st.rerun()

# === Refresh Button ===
if st.button("🔁 Muat Ulang Manual"):
    st.session_state.pop("data_by_date", None)
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
        st.session_state.pop("data_by_date", None)
        st.rerun()
    except Exception as e:
        st.error(f"❌ Gagal menghapus semua file: {e}")

# === Load Data ===
if "data_by_date" not in st.session_state:
    with st.spinner("📦 Mengambil data dari Hugging Face..."):
        load_data_from_hf()

data_by_date = st.session_state.get("data_by_date", {})
filename_by_date = st.session_state.get("filename_by_date", {})

# === Show total file count
total_files = len(filename_by_date)
st.markdown(f"### 📦 Jumlah File Excel Dimuat: **{total_files}**")

# === Display Section ===
st.header("📅 Pilih Tanggal dan Lihat Data")

if data_by_date:
    all_dates = sorted(data_by_date.keys(), reverse=True)
    selected_date = st.selectbox("📆 Pilih Tanggal", options=all_dates)

    if selected_date:
        st.subheader(f"📊 Data Penutupan - {selected_date.strftime('%d %b %Y')}")
        df_display = data_by_date[selected_date].copy()
        df_display['Penutupan'] = df_display['Penutupan'].apply(lambda x: f"{x:,.0f}")
        st.dataframe(df_display, use_container_width=True)

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
                st.session_state.pop("data_by_date", None)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Gagal menghapus: {e}")
else:
    st.info("ℹ️ Tidak ada data yang tersedia. Upload file Excel untuk mulai.")
