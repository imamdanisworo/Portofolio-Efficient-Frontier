# Streamlit multi-tab DBF analyzer with persistent session file store
import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import hashlib
from datetime import datetime
import matplotlib.pyplot as plt
from dbfread import DBF

# Helper: clean padded byte strings from DBF closing price fields
def clean_closing_price(value):
    try:
        if isinstance(value, bytes):
            return float(value.decode('utf-8').strip().replace('\x00', ''))
        elif isinstance(value, str):
            return float(value.strip())
        return float(value)
    except:
        return np.nan

# Hashing to identify duplicate files by content
def get_file_hash(file):
    content = file.read()
    file.seek(0)
    return hashlib.md5(content).hexdigest(), content

# Initialize session state to store uploaded files
def init_session():
    if 'stored_data' not in st.session_state:
        st.session_state.stored_data = {}
    if 'stored_hashes' not in st.session_state:
        st.session_state.stored_hashes = {}

init_session()

st.set_page_config(page_title="Stock DBF Analyzer", layout="wide")
st.title("📈 Multi-tab Stock DBF Analyzer")

tabs = st.tabs(["📁 Upload Data", "📦 Stored Files", "📊 Analyze Data"])

# Tab 1: Upload DBF Files
with tabs[0]:
    st.header("Upload DBF Stock Files")
    uploaded_files = st.file_uploader("Upload DBF files", type="dbf", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            file_hash, file_bytes = get_file_hash(file)
            filename = file.name

            if file_hash in st.session_state.stored_hashes.values():
                st.warning(f"Duplicate file skipped: {filename}")
                continue

            if filename in st.session_state.stored_data:
                st.warning(f"Filename already exists and will not be replaced: {filename}")
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
                    st.session_state.stored_data[filename] = df[['STK_CODE', 'STK_CLOS', 'DATE']]
                    st.session_state.stored_hashes[filename] = file_hash
                    st.success(f"Uploaded and stored: {filename}")
                else:
                    st.error(f"Missing STK_CODE or STK_CLOS in {filename}")
            except Exception as e:
                st.error(f"Error processing {filename}: {e}")

# Tab 2: Stored Files & Delete Option
with tabs[1]:
    st.header("Stored Data Files")
    if st.session_state.stored_data:
        for name in list(st.session_state.stored_data.keys()):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📄 {name}")
            with col2:
                if st.button(f"❌ Delete", key=f"del_{name}"):
                    del st.session_state.stored_data[name]
                    del st.session_state.stored_hashes[name]
                    st.success(f"Deleted: {name}")
    else:
        st.info("No data files stored.")

# Tab 3: Analysis
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
