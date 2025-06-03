import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
import matplotlib.pyplot as plt
from dbfread import DBF

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
            # FIX: Read DBF from in-memory buffer
            table = DBF(io.BytesIO(file.read()), load=True)
            df = pd.DataFrame(iter(table))
            df.columns = df.columns.str.lower().str.strip()
        except Exception as e:
            st.error(f"❌ Error reading {file.name}: {e}")
            continue

        required_cols = ['code', 'closing price']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ {file.name} must include columns: {required_cols}")
            continue

        df['date'] = file_date
        all_data = pd.concat([all_data, df[['code', 'closing price', 'date']]], ignore_index=True)

    if all_data.empty:
        st.warning("⚠️ No valid data collected from uploaded files.")
    else:
        # Pivot data for time series
        prices = all_data.pivot(index='date', columns='code', values='closing price').sort_index()
        st.subheader("📊 Combined Price Table")
        st.dataframe(prices)

        if st.button("🔍 Analyze Data"):
            # Calculate daily returns
            returns = prices.pct_change().dropna()
            st.subheader("📉 Daily Returns")
            st.dataframe(returns)

            mean_returns = returns.mean() * 252  # Annualized return
            risk = returns.std() * np.sqrt(252)  # Annualized volatility

            result_df = pd.DataFrame({
                "Expected Return": mean_returns,
                "Risk (Volatility)": risk
            })

            st.subheader("📈 Risk vs Return")
            st.dataframe(result_df)

            st.subheader("📌 Correlation Matrix")
            st.dataframe(returns.corr())

            # Scatter Plot: Risk vs Return
            fig, ax = plt.subplots()
            ax.scatter(result_df["Risk (Volatility)"], result_df["Expected Return"])
            for i, txt in enumerate(result_df.index):
                ax.annotate(txt, (result_df["Risk (Volatility)"][i], result_df["Expected Return"][i]))
            ax.set_xlabel("Risk (Volatility)")
            ax.set_ylabel("Expected Return")
            ax.set_title("Risk vs Return Scatter Plot")
            st.pyplot(fig)

            # Equal-weighted portfolio suggestion
            st.subheader("💡 Suggested Allocation (Equal Weight)")
            equal_weights = np.repeat(1 / len(mean_returns), len(mean_returns))
            suggestion_df = pd.DataFrame({
                "Stock": mean_returns.index,
                "Weight %": equal_weights * 100
            })
            st.dataframe(suggestion_df)
else:
    st.info("📥 Please upload .dbf stock price files named like CP250515.dbf")
