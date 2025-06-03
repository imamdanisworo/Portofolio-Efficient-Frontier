import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import hashlib
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from dbfread import DBF

# --- File Persistence ---
STORAGE_DIR = "uploaded_dbf_files"
META_FILE = "file_metadata.pkl"
os.makedirs(STORAGE_DIR, exist_ok=True)

def clean_closing_price(value):
    try:
        if isinstance(value, bytes):
            return float(value.decode('utf-8').strip().replace('\x00', ''))
        elif isinstance(value, str):
            return float(value.strip())
        return float(value)
    except:
        return np.nan

def get_file_hash(file):
    content = file.read()
    file.seek(0)
    return hashlib.md5(content).hexdigest(), content

def save_uploaded_file(filename, df, file_hash):
    df.to_pickle(os.path.join(STORAGE_DIR, filename + ".pkl"))
    meta = load_metadata()
    meta[filename] = file_hash
    with open(os.path.join(STORAGE_DIR, META_FILE), "wb") as f:
        pickle.dump(meta, f)

def load_metadata():
    try:
        with open(os.path.join(STORAGE_DIR, META_FILE), "rb") as f:
            return pickle.load(f)
    except:
        return {}

def load_all_files():
    meta = load_metadata()
    data = {}
    for filename in meta:
        try:
            df = pd.read_pickle(os.path.join(STORAGE_DIR, filename + ".pkl"))
            data[filename] = df
        except:
            continue
    return data

def delete_file(filename):
    try:
        os.remove(os.path.join(STORAGE_DIR, filename + ".pkl"))
        meta = load_metadata()
        if filename in meta:
            del meta[filename]
            with open(os.path.join(STORAGE_DIR, META_FILE), "wb") as f:
                pickle.dump(meta, f)
    except:
        pass

# --- App Start ---
st.set_page_config(page_title="Stock DBF Analyzer", layout="wide")
st.title("📈 Multi-tab Stock DBF Analyzer")

# Initialize persistent file data
if 'stored_data' not in st.session_state:
    st.session_state.stored_data = load_all_files()

# --- Tabs ---
tabs = st.tabs(["📁 Upload Data", "📦 Stored Files", "📊 Analyze Data"])

# --- Tab 1: Upload ---
with tabs[0]:
    st.header("Upload DBF Stock Files")
    uploaded_files = st.file_uploader("Upload DBF files", type="dbf", accept_multiple_files=True)

    if uploaded_files:
        existing_meta = load_metadata()
        for file in uploaded_files:
            file_hash, file_bytes = get_file_hash(file)
            filename = file.name

            if filename in existing_meta:
                st.warning(f"File with name '{filename}' already exists and will not be overwritten.")
                continue

            if file_hash in existing_meta.values():
                st.warning(f"Duplicate file content detected for '{filename}', upload skipped.")
                continue

            try:
                temp_path = f"/tmp/{filename}"
                with open(temp_path, "wb") as f:
                    f.write(file_bytes)

                table = DBF(temp_path, load=True)
                df = pd.DataFrame(iter(table))
                df.columns = df.columns.str.upper().str.strip()

                if 'STK_CODE' in df.columns and 'STK_CLOS' in df.columns:
                    date_part = filename.replace("CP", "").split('.')[0]
                    file_date = datetime.strptime(date_part, "%d%m%y").date()
                    df['DATE'] = file_date
                    df['STK_CLOS'] = df['STK_CLOS'].apply(clean_closing_price)

                    save_uploaded_file(filename, df[['STK_CODE', 'STK_CLOS', 'DATE']], file_hash)
                    st.session_state.stored_data[filename] = df[['STK_CODE', 'STK_CLOS', 'DATE']]
                    st.success(f"Uploaded: {filename}")
                else:
                    st.error(f"Missing STK_CODE or STK_CLOS in {filename}")
            except Exception as e:
                st.error(f"Error processing {filename}: {e}")

        st.experimental_rerun()

# --- Tab 2: Stored Files ---
with tabs[1]:
    st.header("Stored Files")
    if st.session_state.stored_data:
        for name in list(st.session_state.stored_data.keys()):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📄 {name}")
            with col2:
                if st.button(f"❌ Delete", key=f"del_{name}"):
                    delete_file(name)
                    del st.session_state.stored_data[name]
                    st.experimental_rerun()
    else:
        st.info("No files stored.")

# --- Tab 3: Analyze ---
with tabs[2]:
    st.header("Analyze Stored Data")

    if st.session_state.stored_data:
        all_df = pd.concat(st.session_state.stored_data.values(), ignore_index=True)
        prices = all_df.pivot(index='DATE', columns='STK_CODE', values='STK_CLOS').sort_index()

        st.subheader("Select Filters")
        selected_stocks = st.multiselect("Choose stock codes", list(prices.columns))
        selected_dates = st.multiselect("Choose dates", list(prices.index))

        if selected_stocks and selected_dates:
            filtered_prices = prices.loc[selected_dates, selected_stocks]
            st.subheader("Filtered Price Data")
            st.dataframe(filtered_prices)

            if st.button("🔍 Analyze"):
                returns = filtered_prices.pct_change().dropna()
                mean_returns = returns.mean() * 252
                risk = returns.std() * np.sqrt(252)

                result_df = pd.DataFrame({
                    "Expected Return": mean_returns,
                    "Risk (Volatility)": risk
                })

                st.subheader("📈 Risk vs Return")
                st.dataframe(result_df)

                st.subheader("📌 Correlation Matrix")
                st.dataframe(returns.corr())

                fig, ax = plt.subplots()
                ax.scatter(result_df["Risk (Volatility)"], result_df["Expected Return"])
                for i, txt in enumerate(result_df.index):
                    ax.annotate(txt, (result_df["Risk (Volatility)"][i], result_df["Expected Return"][i]))
                ax.set_xlabel("Risk (Volatility)")
                ax.set_ylabel("Expected Return")
                ax.set_title("Risk vs Return Scatter Plot")
                st.pyplot(fig)

                st.subheader("💡 Suggested Allocation (Equal Weight)")
                equal_weights = np.repeat(1 / len(mean_returns), len(mean_returns))
                suggestion_df = pd.DataFrame({
                    "Stock": mean_returns.index,
                    "Weight %": equal_weights * 100
                })
                st.dataframe(suggestion_df)
        else:
            st.info("Please select stock codes and dates to run analysis.")
    else:
        st.warning("Upload data in Tab 1 first.")
