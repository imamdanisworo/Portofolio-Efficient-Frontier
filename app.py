import streamlit as st
import pandas as pd
import numpy as np
import os
import hashlib
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from dbfread import DBF

# --- File Storage Setup ---
STORAGE_DIR = "uploaded_dbf_files"
META_FILE = "file_metadata.pkl"
os.makedirs(STORAGE_DIR, exist_ok=True)

# --- Rerun triggers ---
if 'trigger_rerun' in st.session_state:
    del st.session_state['trigger_rerun']
    st.experimental_rerun()

if 'clear_data_trigger' in st.session_state:
    del st.session_state['clear_data_trigger']
    st.experimental_rerun()

# --- Helpers ---
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

def load_metadata():
    try:
        with open(os.path.join(STORAGE_DIR, META_FILE), "rb") as f:
            return pickle.load(f)
    except:
        return {}

def save_metadata(meta):
    with open(os.path.join(STORAGE_DIR, META_FILE), "wb") as f:
        pickle.dump(meta, f)

def save_combined_prices(df, file_hash):
    prices = df.pivot(index='DATE', columns='STK_CODE', values='STK_CLOS')
    prices.sort_index(inplace=True)
    filename = f"pivot_{file_hash}.pkl"
    prices.to_pickle(os.path.join(STORAGE_DIR, filename))
    meta = load_metadata()
    meta[filename] = file_hash
    save_metadata(meta)
    return prices

def load_all_pivoted():
    meta = load_metadata()
    all_frames = []
    for filename in meta:
        if filename.startswith("pivot_"):
            try:
                df = pd.read_pickle(os.path.join(STORAGE_DIR, filename))
                all_frames.append(df)
            except:
                continue
    if all_frames:
        combined = pd.concat(all_frames).groupby(level=0).first()
        return combined.sort_index()
    else:
        return pd.DataFrame()

def delete_by_hash(hash_value):
    meta = load_metadata()
    for fname, h in list(meta.items()):
        if h == hash_value:
            try:
                os.remove(os.path.join(STORAGE_DIR, fname))
            except:
                pass
            del meta[fname]
    save_metadata(meta)

# --- App Layout ---
st.set_page_config(page_title="Stock DBF Analyzer", layout="wide")
st.title("📈 Combined Stock DBF Analyzer")

if 'stored_hashes' not in st.session_state:
    st.session_state.stored_hashes = load_metadata()

# --- Tabs ---
tabs = st.tabs(["📁 Upload Data", "📊 Stored Preview", "📈 Analyze Data"])

# --- Tab 1: Upload ---
with tabs[0]:
    st.header("Upload DBF Stock Files")
    uploaded_files = st.file_uploader("Upload DBF files", type="dbf", accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            file_hash, file_bytes = get_file_hash(file)

            if file_hash in st.session_state.stored_hashes.values():
                st.warning(f"Duplicate content detected, skipped: {file.name}")
                continue

            try:
                temp_path = f"/tmp/{file.name}"
                with open(temp_path, "wb") as f:
                    f.write(file_bytes)

                table = DBF(temp_path, load=True)
                df = pd.DataFrame(iter(table))
                df.columns = df.columns.str.upper().str.strip()

                if 'STK_CODE' in df.columns and 'STK_CLOS' in df.columns:
                    date_part = file.name.replace("CP", "").split('.')[0]
                    file_date = datetime.strptime(date_part, "%d%m%y").date()
                    df['DATE'] = file_date
                    df['STK_CLOS'] = df['STK_CLOS'].apply(clean_closing_price)

                    _ = save_combined_prices(df[['STK_CODE', 'STK_CLOS', 'DATE']], file_hash)
                    st.session_state.stored_hashes[file.name] = file_hash
                    st.success(f"Uploaded: {file.name}")
                else:
                    st.error(f"Missing STK_CODE or STK_CLOS in {file.name}")
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")

        st.session_state['trigger_rerun'] = True
        st.stop()

# --- Tab 2: Preview Stored Data ---
with tabs[1]:
    st.header("Combined Price Table Preview")
    all_prices = load_all_pivoted()

    if not all_prices.empty:
        st.dataframe(all_prices.T)
        if st.button("Clear All Data"):
            for h in st.session_state.stored_hashes.values():
                delete_by_hash(h)
            st.session_state.stored_hashes = {}
            st.session_state['clear_data_trigger'] = True
            st.stop()
    else:
        st.info("No price data available yet.")

# --- Tab 3: Analysis ---
with tabs[2]:
    st.header("Analyze Stock Data")
    prices = load_all_pivoted()

    if not prices.empty:
        st.subheader("Select Stocks and Dates")
        selected_stocks = st.multiselect("Select stock codes", list(prices.columns))
        selected_dates = st.multiselect("Select dates", list(prices.index))

        if selected_stocks and selected_dates:
            filtered = prices.loc[selected_dates, selected_stocks]
            st.subheader("Filtered Prices")
            st.dataframe(filtered)

            if st.button("🔍 Analyze"):
                returns = filtered.pct_change().dropna()
                mean_returns = returns.mean() * 252
                risk = returns.std() * np.sqrt(252)
                result_df = pd.DataFrame({"Expected Return": mean_returns, "Risk (Volatility)": risk})

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
                ax.set_title("Risk vs Return")
                st.pyplot(fig)

                st.subheader("💡 Equal Weight Portfolio")
                weights = np.repeat(1 / len(mean_returns), len(mean_returns))
                st.dataframe(pd.DataFrame({"Stock": mean_returns.index, "Weight %": weights * 100}))
        else:
            st.info("Select both stock codes and dates to proceed.")
    else:
        st.warning("Upload data in Tab 1 first.")
