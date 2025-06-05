import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Page Setup ---
st.set_page_config(page_title="📈 Stock Up/Down Forecast", layout="centered")
st.title("📈 Stock Up/Down Probability with Real-Time Price")

# --- Ticker Input ---
ticker = st.text_input("Enter ticker (e.g., AAPL, 005930.KS)", value="SOXL")

@st.cache_data
def load_price_history(ticker):
    return yf.Ticker(ticker).history(period="5y")

if ticker:
    try:
        # Load historical data
        hist = load_price_history(ticker)
        stock = yf.Ticker(ticker)
        current_price = stock.history(period="1d")["Close"].iloc[-1]

        # --- Suggested Buy/Sell Prices ---
        suggested_buy = current_price * 0.84
        suggested_sell = current_price * 2.2

        # --- Current Price Info with colored buy/sell prices ---
        st.subheader(f"💰 Current Price: ${current_price:.2f}")
        st.markdown(
            f"📌 Suggested Buy Price: <span style='color:red; font-weight:bold'>${suggested_buy:.2f}</span>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"📌 Suggested Sell Price: <span style='color:blue; font-weight:bold'>${suggested_sell:.2f}</span>",
            unsafe_allow_html=True
        )

        # --- Long-Term Moving Averages ---
        hist["MA_6M"] = hist["Close"].rolling(window=126).mean()
        hist["MA_1Y"] = hist["Close"].rolling(window=252).mean()
        hist["MA_2Y"] = hist["Close"].rolling(window=504).mean()

        # --- 📊 Graph (below price info) ---
        st.subheader("📊 5-Year Price Chart with Long-Term Moving Averages")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hist.index, hist["Close"], label="Close Price", color="black", linewidth=1)
        ax.plot(hist.index, hist["MA_6M"], label="6-Month MA", linestyle='--', color="orange")
        ax.plot(hist.index, hist["MA_1Y"], label="1-Year MA", linestyle='--', color="green")
        ax.plot(hist.index, hist["MA_2Y"], label="2-Year MA", linestyle='--', color="red")

        # Buy/Sell guide lines
        ax.axhline(suggested_buy, color="red", linestyle=":", label=f"Buy @ ${suggested_buy:.2f}")
        ax.axhline(suggested_sell, color="blue", linestyle=":", label=f"Sell @ ${suggested_sell:.2f}")

        ax.set_title(f"{ticker.upper()} - Historical Price & Long-Term Moving Averages")
        ax.set_ylabel("Price")
        ax.set_xlabel("Date")
        ax.legend(loc='upper left')
        ax.grid(True)
        st.pyplot(fig)

        # --- Probabilities ---
        st.subheader(f"{ticker.upper()} Up/Down Probabilities")

        def up_down_probability(period_days):
            future_returns = hist["Close"].pct_change(periods=period_days).dropna()
            up_prob = (future_returns > 0).mean() * 100
            down_prob = 100 - up_prob
            return up_prob, down_prob

        periods = {
            "1 Day": 1,
            "1 Week": 5,
            "1 Month": 21,
            "1 Year": 252,
        }

        for label, days in periods.items():
            up, down = up_down_probability(days)
            st.markdown(
                f"**{label} Later →** Up Probability: "
                f"<span style='color:red'>{up:.2f}%</span>, "
                f"Down Probability: <span style='color:blue'>{down:.2f}%</span>",
                unsafe_allow_html=True
            )

        # --- Forecast Prices ---
        st.subheader("📈 Predicted Future Prices (Using Moving Average Return)")

        daily_returns = hist["Close"].pct_change().dropna()
        avg_daily_return = daily_returns.mean()

        for label, days in periods.items():
            predicted_price = current_price * ((1 + avg_daily_return) ** days)
            st.markdown(f"💡 **{label} Later:** `${predicted_price:.2f}`")

    except Exception as e:
        st.error(f"⚠️ Failed to fetch data for ticker `{ticker}`.\n\nDetails: {e}")

