import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Risk & Return Analyzer", layout="wide")

st.title("?? Stock Price Risk & Return Analyzer")

uploaded_files = st.file_uploader("Upload daily stock price files", type=["csv", "txt", "xlsx"], accept_multiple_files=True)

if uploaded_files:
    all_data = pd.DataFrame()

    for file in uploaded_files:
        filename = file.name.split('.')[0]
        try:
            file_date = datetime.strptime(filename, "%d%m%y").date()
        except:
            st.error(f"Invalid filename format: {filename}")
            continue

        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        df.columns = df.columns.str.lower()
        if 'code' not in df.columns or 'closing price' not in df.columns:
            st.error(f"Missing required columns in {file.name}")
            continue

        df['date'] = file_date
        all_data = pd.concat([all_data, df[['code', 'closing price', 'date']]], ignore_index=True)

    # Pivot to get time series format
    prices = all_data.pivot_table(index='date', columns='code', values='closing price').sort_index()
    st.subheader("?? Combined Price Table")
    st.dataframe(prices)

    if st.button("?? Analyze Data"):
        # Calculate daily returns
        returns = prices.pct_change().dropna()

        st.subheader("?? Daily Returns")
        st.dataframe(returns)

        # Expected return and risk
        mean_returns = returns.mean() * 252  # Annualized
        risk = returns.std() * np.sqrt(252)  # Annualized

        result_df = pd.DataFrame({
            "Expected Return": mean_returns,
            "Risk (Volatility)": risk
        })

        st.subheader("?? Risk vs Return")
        st.dataframe(result_df)

        # Correlation Matrix
        st.subheader("?? Correlation Matrix")
        corr = returns.corr()
        st.dataframe(corr)

        # Plot Risk-Return Scatter
        fig, ax = plt.subplots()
        ax.scatter(result_df["Risk (Volatility)"], result_df["Expected Return"])

        for i, txt in enumerate(result_df.index):
            ax.annotate(txt, (result_df["Risk (Volatility)"][i], result_df["Expected Return"][i]))

        ax.set_xlabel("Risk (Volatility)")
        ax.set_ylabel("Expected Return")
        ax.set_title("Risk vs Return Scatter Plot")
        st.pyplot(fig)

        # Simple Equal-Weighted Portfolio Suggestion
        st.subheader("?? Suggested Allocation (Equal Weight)")
        equal_weights = np.repeat(1 / len(mean_returns), len(mean_returns))
        suggestion_df = pd.DataFrame({
            "Stock": mean_returns.index,
            "Weight %": equal_weights * 100
        })
        st.dataframe(suggestion_df)

else:
    st.info("Please upload at least one stock price file.")

