import streamlit as st
import pandas as pd
from dbfread import DBF
from datetime import datetime

st.set_page_config(page_title="Simple DBF Viewer", layout="wide")
st.title("📁 Upload & View DBF Stock Data")

# Upload DBF files
uploaded_files = st.file_uploader("Upload DBF files", type="dbf", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        try:
            # Temporarily save file
            temp_path = f"/tmp/{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.read())

            # Read DBF content
            table = DBF(temp_path, load=True)
            df = pd.DataFrame(iter(table))
            df.columns = df.columns.str.upper().str.strip()

            st.subheader(f"📄 {file.name}")
            st.dataframe(df)

        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")
