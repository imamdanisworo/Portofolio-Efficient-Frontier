import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime
import matplotlib.pyplot as plt
from dbfread import DBF

# Add this helper function to clean the STK_CLOS values
def clean_closing_price(value):
    try:
        if isinstance(value, bytes):
            return float(value.decode('utf-8').strip().replace('\x00', ''))
        elif isinstance(value, str):
            return float(value.strip())
        return float(value)
    except:
        return np.nan

st.set_page_config(page_title="Stock Analyzer (DBF)", layout="wide")
st.title("📈 DBF Stock Price Analyzer")

uploaded_files = st.file_uploader(
    "Upload daily stock price files (.dbf)", 
    type="dbf", 
    accept_multiple_files=True
)

if uploaded_files:
    all_data = pd.DataFrame()

    for file in uploaded_files:
        filename = file.name.split('.')[0]
        try:
            date_part = filename.replace("CP", "")  # Remove 'CP' prefix
            file_date = datetime.strptime(date_part, "%d%m%y").date()
        except:
            st.error(f"❌ Invalid filename format: {filename}. Expected format: CPddmmyy.dbf")
            continue

        try:
            # Save uploaded file to a temporary path and read with DBF
            temp_path = f"/tmp/{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.read())

            table = DBF(temp_path, load=True)
            df = pd.DataFrame(iter(table))
            df.columns = df.columns.str.upper().str.strip()
        except Exception as e:
            st.error(f"❌ Error reading {file.name}: {e}")
            continue

        required_cols = ['STK_CODE', 'STK_CLOS']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ {file.name} must include columns: {required_cols}")
            continue

        df['STK_CLOS'] = df['STK_CLOS'].apply(clean_closing_price)
        df['DATE'] = file_date
        all_data = pd.concat([all_data, df[['STK_CODE', 'STK_CLOS', 'DATE']]], ignore_index=True)

    if all_data.empty:
        st.warning("⚠️ No valid data collected from uploaded files.")
    else:
        # Pivot with STK_CODE as index and DATE as columns
        prices = all_data.pivot(index='STK_CODE', columns='DATE', values='STK_CLOS').sort_index()
        st.subheader("📊 Combined Price Table")
        st.dataframe(prices)

        available_stocks = list(prices.index)
        available_dates = list(prices.columns)

        selected_stocks = st.multiselect("Select stock codes to analyze", available_stocks)
        selected_dates = st.multiselect("Select dates to include", available_dates)

        if selected_stocks and selected_dates:
            filtered_prices = prices.loc[selected_stocks, selected_dates].T.sort_index()

            st.subheader("📈 Filtered Prices")
            st.dataframe(filtered_prices)

            if st.button("🔍 Analyze Data"):
                returns = filtered_prices.pct_change().dropna()
                st.subheader("📉 Daily Returns")
                st.dataframe(returns)

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
            st.info("Select stock codes and dates to proceed with analysis.")
else:
    st.info("📅 Please upload .dbf stock price files named like CP250515.dbf")
