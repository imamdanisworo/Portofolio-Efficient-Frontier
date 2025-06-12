import streamlit as st
import pandas as pd
from dbfread import DBF
from huggingface_hub import upload_file, HfApi, hf_hub_download, delete_file
import os
from datetime import datetime

# === CONFIG ===
REPO_ID = "imamdanisworo/dbf-storage"
HF_TOKEN = st.secrets["HF_TOKEN"]
api = HfApi()

st.set_page_config(page_title="DBF Viewer & Manager", layout="wide")
st.title("📁 View, Upload & Manage DBF Files (Hugging Face)")

# === File Upload ===
uploaded_files = st.file_uploader("⬆️ Upload new DBF files", type="dbf", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        try:
            temp_path = f"/tmp/{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.read())

            # Upload to HF
            with st.spinner(f"Uploading {file.name}..."):
                upload_file(
                    path_or_fileobj=temp_path,
                    path_in_repo=file.name,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    token=HF_TOKEN
                )
                st.success(f"✅ Uploaded: {file.name}")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Upload failed: {e}")

# === Load and Display All Existing Files from HF ===
st.header("📂 Stored DBF Files from Hugging Face")
files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
dbf_files = sorted([f for f in files if f.lower().endswith(".dbf")])

# === Date Filter ===
st.subheader("📅 Filter by File Date (from filename like CP250515.dbf)")
selected_date = st.date_input("Select a date to display (optional)", value=None)

def extract_date_from_filename(name):
    try:
        base = os.path.splitext(name)[0]
        if base.startswith("CP") and len(base) >= 8:
            return datetime.strptime(base[2:], "%y%m%d").date()
    except:
        pass
    return None

displayed = 0
if not dbf_files:
    st.info("No .dbf files found in Hugging Face Dataset.")
else:
    for filename in dbf_files:
        file_date = extract_date_from_filename(filename)

        # Filter by selected date
        if selected_date and file_date != selected_date:
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

            # Display table
            st.subheader(f"📄 {filename} — {file_date.strftime('%d %b %Y') if file_date else 'Unknown date'}")
            st.dataframe(df)

            # Delete button
            if st.button(f"🗑️ Delete {filename}", key=filename):
                delete_file(path_in_repo=filename, repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
                st.success(f"🗑️ Deleted: {filename}")
                st.rerun()

            displayed += 1
            if displayed % 2 == 0:  # separator every 2 tables
                st.markdown("### --- 📎 ---")  # bold styled line

        except Exception as e:
            st.error(f"Error reading {filename}: {e}")
