import streamlit as st
import pandas as pd
import os
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download, upload_file

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

@st.cache_data(show_spinner=False)
def load_existing_files():
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
    xlsx_files = [f for f in files if f.lower().endswith(".xlsx")]
    
    data_by_date = {}
    for file in xlsx_files:
        try:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=file, repo_type="dataset", token=HF_TOKEN)
            df = pd.read_excel(local_path)
            date = get_date_from_filename(file)
            if date and 'Kode Saham' in df.columns and 'Penutupan' in df.columns:
                df_filtered = df[['Kode Saham', 'Penutupan']].copy()
                df_filtered['Tanggal'] = date
                data_by_date[date] = df_filtered
        except Exception as e:
            st.warning(f"Gagal memuat: {file} - {e}")
    return data_by_date

# === Upload Section ===
st.header("⬆️ Upload Data Ringkasan Saham")

uploaded_files = st.file_uploader("Pilih file Excel (.xlsx)", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        try:
            # Upload to Hugging Face dataset
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
    # Clear cache and rerun
    st.cache_data.clear()
    st.rerun()

# === Load and Display Section ===
data_by_date = load_existing_files()

if data_by_date:
    all_dates = sorted(data_by_date.keys())
    selected_date = st.selectbox("📅 Pilih Tanggal", options=all_dates)

    if selected_date:
        st.subheader(f"📊 Data Penutupan - {selected_date.strftime('%d %b %Y')}")
        st.dataframe(data_by_date[selected_date], use_container_width=True)
else:
    st.info("💡 Belum ada data di Hugging Face Dataset.")
