import streamlit as st
import pandas as pd
from huggingface_hub import upload_file, HfApi, hf_hub_download
import os
from datetime import datetime
import time

# === CONFIG ===
REPO_ID = "imamdanisworo/dbf-storage"
HF_TOKEN = st.secrets["HF_TOKEN"]
api = HfApi()

st.set_page_config(page_title="📁 Ringkasan Saham", layout="wide")
st.title("📁 Ringkasan Saham (Hugging Face)")

# === Helper ===
def extract_date_from_filename(name):
    try:
        base = os.path.splitext(name)[0]
        date_str = base.split("-")[-1]  # Expecting Ringkasan Saham-YYYYMMDD.xlsx
        return datetime.strptime(date_str, "%Y%m%d").date()
    except:
        return None

# === File list ===
files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
excel_files = sorted([f for f in files if f.lower().endswith(".xlsx")])
valid_files = [(f, extract_date_from_filename(f)) for f in excel_files if extract_date_from_filename(f)]
unique_dates = sorted({d for _, d in valid_files})

# === Tabs ===
tab1, tab2 = st.tabs(["📂 Upload & View", "📈 Analyze Stocks"])

if 'just_uploaded' not in st.session_state:
    st.session_state.just_uploaded = False

# === TAB 1 ===
with tab1:
    st.header("⬆️ Upload Ringkasan Saham (.xlsx)")
    uploaded_files = st.file_uploader("Upload Excel files", type="xlsx", accept_multiple_files=True)

    if uploaded_files and not st.session_state.just_uploaded:
        for file in uploaded_files:
            filename = file.name
            if extract_date_from_filename(filename) is None:
                st.warning(f"⚠️ Filename format invalid (should include date): {filename}")
                continue

            try:
                temp_path = f"/tmp/{filename}"
                with open(temp_path, "wb") as f:
                    f.write(file.read())

                if filename in excel_files:
                    st.warning(f"⚠️ Overwriting: {filename}")

                with st.spinner(f"Uploading {filename} to Hugging Face..."):
                    upload_file(
                        path_or_fileobj=temp_path,
                        path_in_repo=filename,
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        token=HF_TOKEN
                    )
                    st.success(f"✅ Uploaded: {filename}")
            except Exception as e:
                st.error(f"❌ Failed to upload {filename}: {e}")

        time.sleep(3)
        st.session_state.just_uploaded = True
        st.rerun()

    if not unique_dates:
        st.info("No uploaded Excel files found.")
        st.stop()

    st.header("📅 Select Date to View")
    selected_date = st.selectbox(
        "Choose a date", options=unique_dates, format_func=lambda d: d.strftime('%d %b %Y')
    )

    for filename, file_date in valid_files:
        if file_date == selected_date:
            try:
                local_path = hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=filename,
                    token=HF_TOKEN
                )
                df = pd.read_excel(local_path)
                st.subheader(f"📄 {filename} — {file_date.strftime('%d %b %Y')}")
                st.dataframe(df)
            except Exception as e:
                st.error(f"❌ Error reading {filename}: {e}")

    st.session_state.just_uploaded = False

# === TAB 2 ===
with tab2:
    st.header("📈 Analyze Stock Risk, Return, and Correlation")

    all_data = []
    for filename, file_date in valid_files:
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=filename,
                token=HF_TOKEN
            )
            df = pd.read_excel(local_path)

            if 'Kode Saham' in df.columns and 'Penutupan' in df.columns:
                df['DATE'] = pd.to_datetime(df['Tanggal Perdagangan Terakhir'], errors='coerce')
                df = df[['DATE', 'Kode Saham', 'Penutupan']].dropna()
                df.columns = ['DATE', 'STK_CODE', 'STK_CLOS']
                all_data.append(df)
        except Exception as e:
            st.warning(f"Skipping {filename}: {e}")

    if not all_data:
        st.info("No valid stock data found.")
        st.stop()

    combined = pd.concat(all_data)
    combined['DATE'] = pd.to_datetime(combined['DATE'])
    combined = combined.sort_values('DATE', ascending=False)

    stock_list = sorted(combined['STK_CODE'].unique())
    selected_stocks = st.multiselect("Select stock codes", stock_list)
    selected_period = st.selectbox("Select period (days)", [20, 50, 100, 200, 500])

    if selected_stocks:
        filtered = combined[combined['STK_CODE'].isin(selected_stocks)].copy()
        filtered = filtered.groupby('STK_CODE').head(selected_period)

        pivoted = filtered.pivot(index='DATE', columns='STK_CODE', values='STK_CLOS').sort_index()
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
