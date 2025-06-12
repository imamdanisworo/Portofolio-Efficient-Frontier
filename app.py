import streamlit as st
import pandas as pd
from huggingface_hub import HfApi, upload_file, delete_file, hf_hub_download
import os
from datetime import datetime
import time

# === CONFIG ===
REPO_ID = "imamdanisworo/dbf-storage"
HF_TOKEN = st.secrets["HF_TOKEN"]
api = HfApi()

st.set_page_config(page_title="📁 DBF Stock Manager", layout="wide")
st.title("📁 DBF Stock Manager (Hugging Face)")

# === Helpers ===
def extract_date_from_filename(name):
    try:
        base = os.path.splitext(name)[0]
        date_part = base.split("-")[-1]
        return datetime.strptime(date_part, "%Y%m%d").date()
    except:
        return None

@st.cache_data(show_spinner=False)
def get_valid_files():
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
    excel_files = [f for f in files if f.lower().endswith(".xlsx")]
    valid_files = [(f, extract_date_from_filename(f)) for f in excel_files]
    return [(f, d) for f, d in valid_files if d]

# === Tabs ===
tab1, tab2 = st.tabs(["📂 Upload & View", "📈 Analyze Price Movement"])

# === TAB 1: Upload & View ===
with tab1:
    st.header("⬆️ Upload Excel Files")
    uploaded_files = st.file_uploader("Upload Excel files (Ringkasan Saham-yyyyMMdd.xlsx)", type="xlsx", accept_multiple_files=True)

    if uploaded_files:
        existing_files = [f for f, _ in get_valid_files()]
        for file in uploaded_files:
            filename = file.name
            if filename in existing_files:
                delete_file(path_in_repo=filename, repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
                st.warning(f"⚠️ Overwriting existing file: {filename}")
            with st.spinner(f"Uploading {filename}..."):
                upload_file(path_or_fileobj=file, path_in_repo=filename, repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
                st.success(f"✅ Uploaded: {filename}")
        time.sleep(2)
        st.cache_data.clear()
        st.rerun()

    files = get_valid_files()
    if not files:
        st.info("No valid Excel files uploaded yet.")
        st.stop()

    st.header("📅 Select Date to View")
    date_options = sorted({d for _, d in files}, reverse=True)
    selected_date = st.selectbox("Choose a date", options=date_options, format_func=lambda d: d.strftime('%d %b %Y'))

    for filename, file_date in files:
        if file_date != selected_date:
            continue
        try:
            local_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename, token=HF_TOKEN)
            df = pd.read_excel(local_path)
            df_display = df.copy()
            for col in df_display.select_dtypes(include=['number']).columns:
                df_display[col] = df_display[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
            st.subheader(f"📄 {filename} — {file_date.strftime('%d %b %Y')}")
            st.dataframe(df_display)
        except Exception as e:
            st.error(f"❌ Error reading {filename}: {e}")

# === TAB 2: Analyze Price Movement ===
with tab2:
    st.header("📈 Analyze Price Movement from 'penutupan'")

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    @st.cache_data(show_spinner=False)
    def load_all_data():
        data = []
        for filename, _ in get_valid_files():
            try:
                local_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename, token=HF_TOKEN)
                df = pd.read_excel(local_path)
                df.columns = df.columns.str.strip().str.lower()
                if all(col in df.columns for col in ['tanggal perdagangan terakhir', 'kode saham', 'penutupan']):
                    df = df[['tanggal perdagangan terakhir', 'kode saham', 'penutupan']].dropna()
                    df['tanggal perdagangan terakhir'] = pd.to_datetime(df['tanggal perdagangan terakhir'], dayfirst=True, errors='coerce', format='%d %B %Y')
                    df = df.dropna(subset=['tanggal perdagangan terakhir'])
                    data.append(df)
            except Exception as e:
                st.warning(f"Skipping {filename}: {e}")
        return pd.concat(data) if data else pd.DataFrame()

    combined = load_all_data()
    if combined.empty:
        st.info("No valid data to analyze.")
        st.stop()

    combined = combined.sort_values("tanggal perdagangan terakhir", ascending=False)
    stock_list = sorted(combined['kode saham'].unique())

    selected_stocks = st.multiselect("Select stock codes", stock_list)
    selected_period = st.selectbox("Select analysis period (days)", [20, 50, 100, 200, 500])

    if selected_stocks:
        df_filtered = combined[combined['kode saham'].isin(selected_stocks)].copy()
        df_filtered = df_filtered.sort_values(['kode saham', 'tanggal perdagangan terakhir'], ascending=[True, False])
        df_filtered = df_filtered.groupby('kode saham').head(selected_period)

        pivot = df_filtered.pivot(index='tanggal perdagangan terakhir', columns='kode saham', values='penutupan').sort_index()
        returns = pivot.pct_change().dropna()

        mean_returns = returns.mean()
        risk = returns.std()
        correlation = returns.corr()

        st.subheader("📈 Expected Return (Mean Daily %)")
        st.dataframe((mean_returns * 100).round(3).rename("Return (%)"))

        st.subheader("📉 Risk (Daily Std Deviation %)")
        st.dataframe((risk * 100).round(3).rename("Risk (%)"))

        st.subheader("🔗 Correlation Matrix")
        st.dataframe(correlation.round(3))
    else:
        st.info("Select one or more stock codes to begin analysis.")
