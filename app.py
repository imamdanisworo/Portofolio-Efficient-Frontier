import streamlit as st
import pandas as pd
from dbfread import DBF
from huggingface_hub import upload_file
import os

# === CONFIG ===
REPO_ID = "imamdanisworo/dbf-storage"
HF_TOKEN = st.secrets["HF_TOKEN"]  # Set this in .streamlit/secrets.toml

st.set_page_config(page_title="DBF Viewer & Uploader", layout="wide")
st.title("📁 Upload & View DBF Files (with Hugging Face Upload)")

# Upload DBF files
uploaded_files = st.file_uploader("Upload DBF files", type="dbf", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        try:
            # Save to temporary file
            temp_path = f"/tmp/{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.read())

            # Read DBF
            table = DBF(temp_path, load=True)
            df = pd.DataFrame(iter(table))
            df.columns = df.columns.str.upper().str.strip()

            # Display contents
            st.subheader(f"📄 {file.name}")
            st.dataframe(df)

            # Separator
            st.markdown("---")

            # Upload to HF
            with st.spinner(f"Uploading {file.name} to Hugging Face..."):
                upload_file(
                    path_or_fileobj=temp_path,
                    path_in_repo=file.name,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    token=HF_TOKEN
                )
                st.success(f"✅ Uploaded to Hugging Face: {file.name}")

        except Exception as e:
            st.error(f"❌ Error processing {file.name}: {e}")
