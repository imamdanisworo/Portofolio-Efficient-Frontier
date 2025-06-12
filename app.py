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

st.set_page_config(page_title="DBF Viewer & Manager", layout="wide")
st.title("📁 View, Upload & Manage DBF Files (Hugging Face)")

# === Helper: Extract date from filename like CPyymmdd.dbf ===
def extract_date_from_filename(name):
    try:
        base = os.path.splitext(name)[0]
        if base.startswith("CP") and len(base) >= 8:
            return datetime.strptime(base[2:], "%y%m%d").date()
    except:
        pass
    return None

# === Session flag to prevent infinite rerun
if 'just_uploaded' not in st.session_state:
    st.session_state.just_uploaded = False

# === File Upload ===
uploaded_files = st.file_uploader("⬆️ Upload new DBF files", type="dbf", accept_multiple_files=True)

if uploaded_files and not st.session_state.just_uploaded:
    for file in uploaded_files:
        try:
            temp_path = f"/tmp/{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.read())

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

    # Wait for Hugging Face to sync
    with st.spinner("⏳ Waiting for Hugging Face to sync..."):
        time.sleep(5)

    st.session_state.just_uploaded = True
    st.rerun()

# === Load file list from Hugging Face ===
st.header("📂 Stored DBF Files from Hugging Face")
files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
dbf_files = sorted([f for f in files if f.lower().endswith(".dbf")])

# === Extract valid CPyymmdd.dbf files and dates
valid_files = [(f, extract_date_from_filename(f)) for f in dbf_files]
valid_files = [(f, d) for f, d in valid_files if d]
unique_dates = sorted({d for _, d in valid_files})

# === If no uploaded data, show info
if not unique_dates:
    st.info("No valid CPyymmdd.dbf files found in Hugging Face.")
    st.stop()

# === Dropdown date selector (only valid dates)
selected_date = st.selectbox(
    "📅 Select a date to display",
    options=unique_dates,
    format_func=lambda d: d.strftime('%d %b %Y')
)

# === Display DBF tables only for selected date
displayed = 0
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
            st.session_state.just_uploaded = False
            st.rerun()

        displayed += 1
        if displayed % 2 == 0:
            st.markdown("### --- 📎 ---")

    except Exception as e:
        st.error(f"❌ Error reading or displaying {filename}: {e}")

# === Reset upload flag
st.session_state.just_uploaded = False
