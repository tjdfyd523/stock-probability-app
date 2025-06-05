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

def find_cross_price(series1, series2):
    """Find most recent upward crossover point and return close price at that time."""
    cross = (series1 > series2) & (series1.shift(1) <= series2.shift(1))
    if cross.any():
        return hist.loc[cross, "Close"].iloc[-1]
    return None

if ticker:
    try:
        # Load historical data
        hist = load_price_history(ticker)
        stock = yf.Ticker(ticker)
        current_price = stock.history(period="1d")["Close"].iloc[-1]

        # --- Moving Averages ---
        hist["MA_6M"] = hist["Close"].rolling(window=126).mean()
        hist["MA_1Y"] = hist["Close"].rolling(window=252).mean()
        hist["MA_2Y"] = hist["Close"].rolling(window=504).mean()

        # --- 매수 권장가: 골든크로스 기반 ---
        # 원래 2차 → 1차로 표시
        buy_price_1 = find_cross_price(hist["MA_1Y"], hist["MA_2Y"])  # 골든크로스 1Y > 2Y

        # --- 매도 권장가: 웨이브 최고점 기반 하락 %
        wave_high = hist["Close"].rolling(window=252).max().iloc[-1]  # 최근 1년 최고가
        sell_price_1 = wave_high * 0.90
        sell_price_2 = wave_high * 0.85

        # --- Price Display ---
        st.subheader(f"💰 Current Price: ${current_price:.2f}")
        if buy_price_1:
            st.markdown(f"📌 **1차 매수 권장가 (1Y > 2Y 골든크로스):** <span style='color:green; font-weight:bold'>${buy_price_1:.2f}</span>", unsafe_allow_html=True)
        st.markdown(f"📌 **1차 매도 권장가 (고점 대비 -10%):** <span style='color:red; font-weight:bold'>${sell_price_1:.2f}</span>", unsafe_allow_html=True)
        st.markdown(f"📌 **2차 매도 권장가 (고점 대비 -15%):** <span style='color:darkred; font-weight:bold'>${sell_price_2:.2f}</span>", unsafe_allow_html=True)

        # --- 📊 Graph ---
        st.subheader("📊 5-Year Price Chart with Moving Averages and Entry/Exit Zones")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hist.index, hist["Close"], label="Close Price", color="black", linewidth=1)
        ax.plot(hist.index, hist["MA_6M"], label="6-Month MA", linestyle='--', color="orange")
        ax.plot(hist.index, hist["MA_1Y"], label="1-Year MA", linestyle='--', color="green")
        ax.plot(hist.index, hist["MA_2Y"], label="2-Year MA", linestyle='--', color="red")

        if buy_price_1:
            ax.axhline(buy_price_1, color="green", linestyle=":", label=f"1차 매수 @ ${buy_price_1:.2f}")
        ax.axhline(sell_price_1, color="red", linestyle="--", label=f"1차 매도 @ ${sell_price_1:.2f}")
        ax.axhline(sell_price_2, color="darkred", linestyle="--", label=f"2차 매도 @ ${sell_price_2:.2f}")

        ax.set_title(f"{ticker.upper()} - Price & Strategy-Based Entry/Exit Zones")
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


