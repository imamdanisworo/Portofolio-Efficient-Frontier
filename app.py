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

st.markdown("<h1 style='text-align:center;'>📈 Ringkasan Saham - Kode & Penutupan</h1>", unsafe_allow_html=True)

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

def load_data_from_hf():
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
    xlsx_files = [f for f in files if f.lower().endswith(".xlsx")]

    data_by_date = {}
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
                data_by_date[date] = df_filtered
                filename_by_date[date] = file
        progress_bar.progress((i + 1) / len(xlsx_files))

    status.success("✅ Semua file berhasil dimuat.")
    st.session_state.data_by_date = data_by_date
    st.session_state.filename_by_date = filename_by_date

# === Upload Section ===
st.markdown("### 📤 Upload File Excel")

uploaded_files = st.file_uploader("Pilih file Excel (.xlsx)", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    with st.expander("🔄 Status Upload", expanded=True):
        success_count = 0
        fail_count = 0
        progress_bar = st.progress(0)
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
            progress_bar.progress((i + 1) / len(uploaded_files))

    st.success(f"📦 Selesai! {success_count} berhasil, {fail_count} gagal.")
    if st.button("🔃 Muat Ulang untuk Menampilkan Data Terbaru"):
        st.cache_resource.clear()
        st.session_state.pop("data_by_date", None)
        st.rerun()

# === Delete All Files Section ===
st.markdown("### 🗑️ Hapus Semua Data Excel")

if st.button("❌ Hapus SEMUA File Excel dari Dataset"):
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
        st.success("🧹 Semua file berhasil dihapus.")
        st.cache_resource.clear()
        st.session_state.pop("data_by_date", None)
        st.rerun()
    except Exception as e:
        st.error(f"⚠️ Gagal menghapus semua file: {e}")

# === Load Existing Data
if "data_by_date" not in st.session_state:
    with st.spinner("📦 Mengambil data dari Hugging Face..."):
        load_data_from_hf()

data_by_date = st.session_state.get("data_by_date", {})
filename_by_date = st.session_state.get("filename_by_date", {})

# === File Summary
st.markdown(f"### 📂 Jumlah File Tersimpan: **{len(filename_by_date)}**")

# === Display Table by Date
st.markdown("### 📅 Pilih Tanggal untuk Melihat Data")

if data_by_date:
    sorted_dates = sorted(data_by_date.keys(), reverse=True)
    selected_date = st.selectbox("📆 Tanggal:", sorted_dates)

    df_display = data_by_date[selected_date].copy()
    df_display['Penutupan'] = df_display['Penutupan'].apply(lambda x: f"{x:,.0f}")
    st.dataframe(df_display, use_container_width=True)

    file_to_delete = filename_by_date[selected_date]
    if st.button(f"🗑️ Hapus File Tanggal Ini ({file_to_delete})"):
        try:
            delete_file(
                path_in_repo=file_to_delete,
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN
            )
            st.success(f"✅ Berhasil menghapus: {file_to_delete}")
            st.cache_resource.clear()
            st.session_state.pop("data_by_date", None)
            st.rerun()
        except Exception as e:
            st.error(f"❌ Gagal menghapus: {e}")
else:
    st.info("📭 Belum ada data untuk ditampilkan.")
