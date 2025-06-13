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

st.markdown("<h2 style='text-align:center;'>📈 Ringkasan Saham</h2>", unsafe_allow_html=True)

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

def validate_file(file, is_index=False):
    try:
        df = pd.read_excel(file)
        date = get_date_from_filename(file.name)
        if not date:
            return False, "❌ Tanggal tidak dikenali", None, None

        if is_index:
            if "Kode Indeks" in df.columns and "Penutupan" in df.columns:
                filtered = df[df["Kode Indeks"].str.lower() == "composite"]
                if filtered.empty:
                    return False, "❌ Data indeks tidak berisi 'composite'", None, None
                return True, "", date, filtered.iloc[0]["Penutupan"]
            else:
                return False, "❌ Kolom wajib 'Kode Indeks' dan 'Penutupan' tidak ditemukan", None, None
        else:
            if not all(col in df.columns for col in ["Kode Saham", "Penutupan"]):
                return False, "❌ Kolom wajib 'Kode Saham' dan 'Penutupan' tidak ditemukan", None, None
            df_filtered = df[["Kode Saham", "Penutupan"]].copy()
            df_filtered["Tanggal"] = date
            return True, "", date, df_filtered

    except Exception as e:
        return False, f"❌ Gagal membaca file: {e}", None, None

def process_files(files, is_index=False):
    messages = []
    new_index_data = {}
    new_stock_data = {}
    new_filenames = {}

    progress = st.progress(0, "📤 Memproses file...")
    total = len(files)

    for i, file in enumerate(files):
        valid, msg, date, result = validate_file(file, is_index)
        if not valid:
            messages.append(f"{file.name}: {msg}")
        else:
            if not is_index and date in st.session_state["data_by_date"]:
                messages.append(f"⚠️ {file.name}: data untuk tanggal ini sudah ada, akan ditimpa.")

            try:
                name_in_repo = f"index-{file.name}" if is_index else file.name
                upload_file(path_or_fileobj=file, path_in_repo=name_in_repo,
                            repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)

                if is_index:
                    new_index_data[date] = result
                else:
                    new_stock_data[date] = result
                    new_filenames[date] = name_in_repo

                messages.append(f"✅ {file.name} berhasil diunggah")
            except Exception as e:
                messages.append(f"{file.name}: ❌ Gagal unggah ke HF: {e}")
        progress.progress((i + 1) / total)

    progress.empty()

    if is_index:
        st.session_state["index_series"].update(new_index_data)
    else:
        st.session_state["data_by_date"].update(new_stock_data)
        st.session_state["filename_by_date"].update(new_filenames)

    return messages, bool(new_index_data or new_stock_data)

col1, col2 = st.columns(2)
uploaded_any = False

with col1:
    index_files = st.file_uploader("Upload File Indeks (.xlsx)", type="xlsx", accept_multiple_files=True, key="upload_index")
    if index_files:
        msgs, changed = process_files(index_files, is_index=True)
        for m in msgs:
            st.toast(m)
        uploaded_any = uploaded_any or changed

with col2:
    stock_files = st.file_uploader("Upload File Saham (.xlsx)", type="xlsx", accept_multiple_files=True, key="upload_saham")
    if stock_files:
        msgs, changed = process_files(stock_files, is_index=False)
        for m in msgs:
            st.toast(m)
        uploaded_any = uploaded_any or changed

# ✅ BUG FIX: reload all after upload
if uploaded_any:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.pop("data_loaded", None)  # 🧠 Force reloading data
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
            st.cache_resource.clear()
            st.session_state.clear()
            st.rerun()
        except Exception as e:
            st.error(str(e))

# Viewer
st.divider()
st.markdown(f"**📄 Jumlah File Saham:** {len(data_by_date)}  &nbsp;&nbsp;|&nbsp;&nbsp; 📄 **Jumlah File Indeks:** {len(index_series)}")

if data_by_date:
    sorted_dates = sorted(data_by_date.keys(), reverse=True)
    date_labels = [d.strftime("%d-%b-%Y") for d in sorted_dates]
    selected_label = st.selectbox("📆 Pilih Tanggal Data", date_labels)
    selected_date = sorted_dates[date_labels.index(selected_label)]

    df_show = data_by_date[selected_date].copy()
    df_show["Penutupan"] = df_show["Penutupan"].apply(lambda x: f"{x:,.0f}")

    if selected_date in index_series:
        st.markdown("#### 📊 Indeks Composite")
        index_df = pd.DataFrame({"Composite": [f"{index_series[selected_date]:,.0f}"]})
        st.dataframe(index_df, use_container_width=True)

    st.markdown("#### 📋 Data Saham")
    st.dataframe(df_show, use_container_width=True)

    try:
        file_path = hf_hub_download(REPO_ID, filename_by_date[selected_date], repo_type="dataset", token=HF_TOKEN)
        with open(file_path, "rb") as f:
            st.download_button("⬇️ Unduh File Saham", f, file_name=filename_by_date[selected_date])
    except:
        pass

    if st.button("🗑️ Hapus Data Ini"):
        try:
            delete_file(filename_by_date[selected_date], REPO_ID, repo_type="dataset", token=HF_TOKEN)
            st.success("✅ Data berhasil dihapus.")
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Gagal menghapus: {e}")
