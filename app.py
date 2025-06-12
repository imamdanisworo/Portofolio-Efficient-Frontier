import streamlit as st
import pandas as pd
from dbfread import DBF
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
        if base.startswith("CP") and len(base) >= 8:
            return datetime.strptime(base[2:], "%y%m%d").date()
    except:
        pass
    return None

def clean_stk_clos(value):
    try:
        if isinstance(value, bytes):
            value = value.decode('utf-8', errors='ignore')
        elif isinstance(value, str):
            value = value

        value = value.replace('\x00', '').strip()
        value = ''.join(c for c in value if c.isdigit() or c in ['.', '-'])
        return float(value)
    except:
        return None

# === File list ===
files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
dbf_files = sorted([f for f in files if f.lower().endswith(".dbf")])
valid_files = [(f, extract_date_from_filename(f)) for f in dbf_files]
valid_files = [(f, d) for f, d in valid_files if d]
unique_dates = sorted({d for _, d in valid_files})

# === Tabs ===
tab1, tab2 = st.tabs(["📂 Upload & View", "📈 Analyze Stocks"])

if 'just_uploaded' not in st.session_state:
    st.session_state.just_uploaded = False

# === TAB 1 ===
with tab1:
    st.header("⬆️ Upload DBF Files")
    uploaded_files = st.file_uploader("Upload DBF files", type="dbf", accept_multiple_files=True)

    if uploaded_files and not st.session_state.just_uploaded:
        for file in uploaded_files:
            try:
                temp_path = f"/tmp/{file.name}"
                with open(temp_path, "wb") as f:
                    f.write(file.read())

                if file.name in dbf_files:
                    delete_file(
                        path_in_repo=file.name,
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        token=HF_TOKEN
                    )
                    st.warning(f"⚠️ Overwriting: {file.name}")

                with st.spinner(f"Uploading {file.name} to Hugging Face..."):
                    upload_file(
                        path_or_fileobj=temp_path,
                        path_in_repo=file.name,
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        token=HF_TOKEN
                    )
                    st.success(f"✅ Uploaded: {file.name}")
            except Exception as e:
                st.error(f"❌ Failed to upload {file.name}: {e}")

        time.sleep(5)
        st.session_state.just_uploaded = True
        st.rerun()

    if not unique_dates:
        st.info("No valid DBF files uploaded yet.")
        st.stop()

    st.header("📅 Select Date to View")
    selected_date = st.selectbox(
        "Choose a date (from uploaded files)", 
        options=unique_dates,
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
            table = DBF(local_path, load=True)
            df = pd.DataFrame(iter(table))
            df.columns = df.columns.str.upper().str.strip()
            if 'STK_CLOS' in df.columns:
                df['STK_CLOS'] = df['STK_CLOS'].apply(clean_stk_clos)

            st.subheader(f"📄 {filename} — {file_date.strftime('%d %b %Y')}")
            st.dataframe(df)

            if st.button(f"🗑️ Delete {filename}", key=filename):
                delete_file(
                    path_in_repo=filename,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    token=HF_TOKEN
                )
                st.success(f"🗑️ Deleted: {filename}")
                time.sleep(2)
                st.rerun()

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
            table = DBF(local_path, load=True)
            df = pd.DataFrame(iter(table))
            df.columns = df.columns.str.upper().str.strip()

            if 'STK_CODE' in df.columns and 'STK_CLOS' in df.columns:
                df['STK_CLOS'] = df['STK_CLOS'].apply(clean_stk_clos)
                df['DATE'] = pd.to_datetime(file_date)
                df = df[['DATE', 'STK_CODE', 'STK_CLOS']].dropna()
                all_data.append(df)
        except Exception as e:
            st.warning(f"Skipping {filename}: {e}")

    if not all_data:
        st.info("No valid data to analyze.")
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
