import streamlit as st
import pandas as pd
import os
import io
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download, upload_file, delete_file

# CONFIG
st.set_page_config(page_title="📊 Upload & Tinjau Data Saham", layout="wide")
REPO_ID = "imamdanisworo/dbf-storage"
HF_TOKEN = st.secrets["HF_TOKEN"]

@st.cache_resource
def get_hf_api():
    return HfApi()

api = get_hf_api()

# Header
st.markdown("<h2 style='text-align:center;'>📊 Upload & Tinjau Data</h2>", unsafe_allow_html=True)

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
st.info("📌 File harus dalam format `.xlsx`. Nama file menentukan tipe data:\n\n- Gunakan awalan `index-` untuk data indeks\n- File lain akan dianggap sebagai data saham.\n\n📌 File dengan nama sama akan otomatis **mengganti** versi lama di database.")

uploaded_files = st.file_uploader("Upload File Data (.xlsx)", type="xlsx", accept_multiple_files=True, key="upload_all")

def validate_excel(file_bytes, is_index):
    try:
        df = pd.read_excel(io.BytesIO(file_bytes))
    except Exception as e:
        return False, None, f"Gagal membaca file Excel: {e}"

    if is_index:
        if not all(col in df.columns for col in ["Kode Indeks", "Penutupan"]):
            return False, None, "❌ Kolom wajib 'Kode Indeks' dan 'Penutupan' tidak ditemukan"
    else:
        if not all(col in df.columns for col in ["Kode Saham", "Penutupan"]):
            return False, None, "❌ Kolom wajib 'Kode Saham' dan 'Penutupan' tidak ditemukan"

    return True, df, None

def process_file(file, is_index=False):
    try:
        file_bytes = file.read()
        valid, df, error_msg = validate_excel(file_bytes, is_index)
        if not valid:
            return False, error_msg

        name_in_repo = file.name
        date = get_date_from_filename(file.name)
        if not date:
            return False, "❌ Nama file tidak mengandung tanggal valid (format: YYYYMMDD)"

        # Check if file exists
        existing_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
        will_overwrite = name_in_repo in existing_files

        if will_overwrite:
            st.info(f"⚠️ File dengan nama **{name_in_repo}** sudah ada dan akan diganti.")

        # Delete old version
        try:
            delete_file(path_in_repo=name_in_repo, repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
        except:
            pass

        # Upload
        upload_file(
            path_or_fileobj=io.BytesIO(file_bytes),
            path_in_repo=name_in_repo,
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )

        # Save to session
        if is_index:
            filtered = df[df["Kode Indeks"].str.lower() == "composite"]
            if not filtered.empty:
                st.session_state["index_series"][date] = filtered.iloc[0]["Penutupan"]
        else:
            filtered = df[["Kode Saham", "Penutupan"]].copy()
            filtered["Tanggal"] = date
            st.session_state["data_by_date"][date] = filtered
            st.session_state["filename_by_date"][date] = name_in_repo

        action = "diperbarui" if will_overwrite else "diunggah"
        return True, f"✅ {file.name} berhasil {action}."

    except Exception as e:
        return False, f"❌ Gagal unggah {file.name}: {e}"

def handle_all_upload(files):
    if files:
        st.markdown("#### 📥 Status Upload")
        results = []
        rerun_needed = False

        for file in files:
            is_index = file.name.lower().startswith("index-")
            status_placeholder = st.empty()
            status_placeholder.info(f"⏳ Memproses {file.name}...")

            success, message = process_file(file, is_index=is_index)
            results.append((file.name, success, message))
            status_placeholder.empty()

            rerun_needed = rerun_needed or success

        for fname, success, msg in results:
            icon = "✅" if success else "❌"
            st.markdown(f"{icon} **{fname}**: {msg}")

handle_all_upload(uploaded_files)

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
            for k in ["data_by_date", "index_series", "filename_by_date", "data_loaded"]:
                st.session_state.pop(k, None)
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
            for k in ["data_by_date", "index_series", "filename_by_date", "data_loaded"]:
                st.session_state.pop(k, None)
            st.rerun()
        except Exception as e:
            st.error(f"Gagal menghapus: {e}")
