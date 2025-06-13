import streamlit as st
import pandas as pd
import os
import io
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

# Helpers
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
st.info("📌 File yang memiliki nama sama akan otomatis **mengganti** versi lama di database.")

def process_file(file, is_index=False):
    try:
        name_in_repo = f"index-{file.name}" if is_index else file.name

        # Read file into memory buffer
        file_bytes = file.read()
        buffer = io.BytesIO(file_bytes)

        # Delete old file if it exists
        try:
            delete_file(path_in_repo=name_in_repo, repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
        except:
            pass

        # Upload from memory
        upload_file(
            path_or_fileobj=io.BytesIO(file_bytes),
            path_in_repo=name_in_repo,
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )

        # Load DataFrame from buffer
        df = pd.read_excel(io.BytesIO(file_bytes))
        date = get_date_from_filename(file.name)
        if not date:
            return False, "Tanggal tidak dikenali dari nama file"

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

        return True, f"✅ {file.name} berhasil diunggah dan menggantikan file lama (jika ada)"
    except Exception as e:
        return False, f"❌ Gagal unggah {file.name}: {e}"

col1, col2 = st.columns(2)
uploaded_any = False

with col1:
    index_files = st.file_uploader("Upload File Indeks (.xlsx)", type="xlsx", accept_multiple_files=True, key="upload_index")
    if index_files:
        progress = st.progress(0, "📤 Mengunggah file indeks...")
        total = len(index_files)
        for i, file in enumerate(index_files):
            success, msg = process_file(file, is_index=True)
            st.success(msg) if success else st.error(msg)
            progress.progress((i + 1) / total, text=f"📤 Mengunggah {file.name} ({i+1}/{total})")
            uploaded_any = uploaded_any or success
        progress.empty()

with col2:
    stock_files = st.file_uploader("Upload File Saham (.xlsx)", type="xlsx", accept_multiple_files=True, key="upload_saham")
    if stock_files:
        progress = st.progress(0, "📤 Mengunggah file saham...")
        total = len(stock_files)
        for i, file in enumerate(stock_files):
            success, msg = process_file(file, is_index=False)
            st.success(msg) if success else st.error(msg)
            progress.progress((i + 1) / total, text=f"📤 Mengunggah {file.name} ({i+1}/{total})")
            uploaded_any = uploaded_any or success
        progress.empty()

if uploaded_any:
    st.cache_data.clear()
    st.rerun()

# Delete All
st.divider()
if st.button("🧹 Hapus Semua Data"):
    with st.spinner("🚮 Menghapus semua file dari Hugging Face..."):
        try:
            all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
            for file in all_files:
                if file.lower().endswith(".xlsx"):
                    delete_file(file, REPO_ID, repo_type="dataset", token=HF_TOKEN)
            st.success("✅ Semua file berhasil dihapus.")
            st.cache_data.clear()
            for key in ["data_by_date", "index_series", "filename_by_date", "data_loaded"]:
                st.session_state.pop(key, None)
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
        st.metric(label="Indeks Composite", value=f"{index_series[selected_date]:,.0f}")

    st.markdown("#### 📋 Data Saham")
    st.dataframe(df_show, use_container_width=True)

    if st.button("🗑️ Hapus Data Ini"):
        try:
            delete_file(filename_by_date[selected_date], REPO_ID, repo_type="dataset", token=HF_TOKEN)
            st.success("✅ Data berhasil dihapus.")
            st.cache_data.clear()
            for key in ["data_by_date", "index_series", "filename_by_date", "data_loaded"]:
                st.session_state.pop(key, None)
            st.rerun()
        except Exception as e:
            st.error(f"Gagal menghapus: {e}")
