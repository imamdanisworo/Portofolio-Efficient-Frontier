import streamlit as st
import pandas as pd
from dbfread import DBF
from huggingface_hub import upload_file, HfApi, hf_hub_download, delete_file
import os

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

            # Upload to Hugging Face
            with st.spinner(f"Uploading {file.name}..."):
                upload_file(
                    path_or_fileobj=temp_path,
                    path_in_repo=file.name,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    token=HF_TOKEN
                )
                st.success(f"✅ Uploaded to Hugging Face: {file.name}")
            st.rerun()  # Refresh after upload
        except Exception as e:
            st.error(f"❌ Upload failed: {e}")

# === Load and Display All Existing Files from HF ===
st.header("📂 Stored DBF Files from Hugging Face")
files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)

dbf_files = [f for f in files if f.lower().endswith(".dbf")]
if not dbf_files:
    st.info("No .dbf files found in your Hugging Face Dataset.")
else:
    for filename in dbf_files:
        try:
            local_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename, token=HF_TOKEN)

            # Read and display
            table = DBF(local_path, load=True)
            df = pd.DataFrame(iter(table))
            df.columns = df.columns.str.upper().str.strip()

            st.subheader(f"📄 {filename}")
            st.dataframe(df)
            delete_button = st.button(f"🗑️ Delete {filename}", key=filename)

            if delete_button:
                delete_file(path_in_repo=filename, repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
                st.success(f"🗑️ Deleted: {filename}")
                st.rerun()

            st.markdown("---")
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
