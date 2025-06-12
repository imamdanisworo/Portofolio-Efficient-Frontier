import streamlit as st
import pandas as pd
from huggingface_hub import upload_file, HfApi, hf_hub_download, delete_file
import os
from datetime import datetime
import time

# === CONFIG ===
REPO_ID = "imamdanisworo/dbf-storage"
HF_TOKEN = st.secrets["HF_TOKEN"]
api = HfApi()

st.set_page_config(page_title="📁 DBF Stock Manager", layout="wide")
st.title("📁 DBF Stock Manager (Hugging Face)")

# === Helper ===
def extract_date_from_filename(name):
    try:
        base = os.path.splitext(name)[0]
        date_part = base.split("-")[-1]
        return datetime.strptime(date_part, "%Y%m%d").date()
    except:
        return None

# === File list ===
@st.cache_data(show_spinner=False)
def get_valid_files():
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
    excel_files = sorted([f for f in files if f.lower().endswith(".xlsx")])
    valid_files = [(f, extract_date_from_filename(f)) for f in excel_files]
    valid_files = [(f, d) for f, d in valid_files if d]
    return valid_files

valid_files = get_valid_files()
unique_dates = sorted({d for _, d in valid_files}, reverse=True)

# === Tabs ===
tab1, tab2 = st.tabs(["📂 Upload & View", "📈 Analyze Price Movement"])

if 'just_uploaded' not in st.session_state:
    st.session_state.just_uploaded = False

# === TAB 1 ===
with tab1:
    st.header("⬆️ Upload Excel Files")
    uploaded_files = st.file_uploader("Upload Excel files (Ringkasan Saham-yyyyMMdd.xlsx)", type="xlsx", accept_multiple_files=True)

    if uploaded_files and not st.session_state.just_uploaded:
        for file in uploaded_files:
            try:
                excel_filename = file.name

                if excel_filename in [f for f, _ in valid_files]:
                    delete_file(
                        path_in_repo=excel_filename,
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        token=HF_TOKEN
                    )
                    st.warning(f"⚠️ Overwriting: {excel_filename}")

                with st.spinner(f"Uploading {excel_filename} to Hugging Face..."):
                    upload_file(
                        path_or_fileobj=file,
                        path_in_repo=excel_filename,
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        token=HF_TOKEN
                    )
                    st.success(f"✅ Uploaded: {excel_filename}")

            except Exception as e:
                st.error(f"❌ Failed to upload {file.name}: {e}")

        time.sleep(5)
        st.cache_data.clear()
        st.session_state.just_uploaded = True
        st.rerun()

    if not unique_dates:
        st.info("No valid Excel files uploaded yet.")
        st.stop()

    st.header("📅 Select Date to View")
    selected_date = st.selectbox(
        "Choose a date (from uploaded files)", 
        options=sorted(unique_dates, reverse=True),
        format_func=lambda d: d.strftime('%d %b %Y')
    )

    for filename, file_date in valid_files:
        if file_date != selected_date:
            continue

        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=filename,
                token=HF_TOKEN
            )
            df = pd.read_excel(local_path)

            # Format numeric columns with thousand separators
            formatted_df = df.copy()
            for col in formatted_df.select_dtypes(include=['number']).columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")

            st.subheader(f"📄 {filename} — {file_date.strftime('%d %b %Y')}")
            st.dataframe(formatted_df)

        except Exception as e:
            st.error(f"❌ Error reading {filename}: {e}")

    st.session_state.just_uploaded = False

# === TAB 2 ===
with tab2:
    st.header("📈 Analyze Price Movement from 'penutupan'")

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    @st.cache_data(show_spinner=False)
    def load_all_data():
        valid_files = get_valid_files()
        data = []
        for filename, file_date in valid_files:
            try:
                local_path = hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=filename,
                    token=HF_TOKEN
                )
                df = pd.read_excel(local_path)
                df.columns = df.columns.str.strip().str.lower()

                required_cols = ['tanggal perdagangan terakhir', 'kode saham', 'penutupan']
                if all(col in df.columns for col in required_cols):
                    df = df[required_cols].copy()
                    df.dropna(inplace=True)
                    df['tanggal perdagangan terakhir'] = pd.to_datetime(
                        df['tanggal perdagangan terakhir'], dayfirst=True, errors='coerce', format='%d %B %Y'
                    )
                    df.dropna(subset=['tanggal perdagangan terakhir'], inplace=True)
                    if not df.empty:
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
        filtered = combined[combined['kode saham'].isin(selected_stocks)].copy()
        filtered = filtered.sort_values(['kode saham', 'tanggal perdagangan terakhir'], ascending=[True, False])
        filtered = filtered.groupby('kode saham').head(selected_period)

        pivoted = filtered.pivot(index='tanggal perdagangan terakhir', columns='kode saham', values='penutupan').sort_index()
        returns = pivoted.pct_change().dropna()

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
