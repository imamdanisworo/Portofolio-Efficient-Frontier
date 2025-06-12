import streamlit as st
from huggingface_hub import HfApi, delete_file

# === CONFIG ===
REPO_ID = "imamdanisworo/dbf-storage"
HF_TOKEN = st.secrets["HF_TOKEN"]
api = HfApi()

st.set_page_config(page_title="🗑️ Delete All Excel Files", layout="wide")
st.title("🗑️ Reset Hugging Face Dataset")

# === List all Excel files ===
def get_excel_files():
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
    return [f for f in files if f.lower().endswith(".xlsx")]

excel_files = get_excel_files()

if not excel_files:
    st.success("✅ No Excel files found — dataset is already clean.")
else:
    st.warning("⚠️ This will permanently delete all Excel (.xlsx) files from the dataset.")
    if st.button("🔥 Delete All Excel Files"):
        for f in excel_files:
            try:
                delete_file(path_in_repo=f, repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
                st.success(f"🗑️ Deleted: {f}")
            except Exception as e:
                st.error(f"❌ Failed to delete {f}: {e}")
        st.cache_data.clear()
        st.rerun()
