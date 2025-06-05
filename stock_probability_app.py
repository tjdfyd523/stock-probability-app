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
        # Load historical data (5 years fixed for analysis)
        hist = load_price_history(ticker)
        stock = yf.Ticker(ticker)
        current_price = stock.history(period="1d")["Close"].iloc[-1]

        # --- Suggested Buy/Sell Prices 개선 ---
        max_price_5y = hist['Close'].max()

        if max_price_5y > current_price * 1.2:
            suggested_sell = max_price_5y * 0.9
        else:
            suggested_sell = current_price * 1.2

        suggested_buy = current_price * 0.84

        # --- Current Price Info ---
        st.subheader(f"💰 Current Price: ${current_price:.2f}")
        st.markdown(
            f"📌 Suggested Buy Price: <span style='color:red; font-weight:bold'>${suggested_buy:.2f}</span>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"📌 Suggested Sell Price: <span style='color:blue; font-weight:bold'>${suggested_sell:.2f}</span>",
            unsafe_allow_html=True
        )

        # --- Calculate Moving Averages ---
        hist["MA_6M"] = hist["Close"].rolling(window=126).mean()
        hist["MA_1Y"] = hist["Close"].rolling(window=252).mean()
        hist["MA_2Y"] = hist["Close"].rolling(window=504).mean()

        # --- 1) 5-Year Price Chart with Moving Averages ---
        st.subheader("📊 5-Year Price Chart with Long-Term Moving Averages")

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(hist.index, hist["Close"], label="Close Price", color="black", linewidth=1)
        ax1.plot(hist.index, hist["MA_6M"], label="6-Month MA", linestyle='--', color="orange")
        ax1.plot(hist.index, hist["MA_1Y"], label="1-Year MA", linestyle='--', color="green")
        ax1.plot(hist.index, hist["MA_2Y"], label="2-Year MA", linestyle='--', color="red")

        ax1.axhline(suggested_buy, color="red", linestyle=":", label=f"Buy @ ${suggested_buy:.2f}")
        ax1.axhline(suggested_sell, color="blue", linestyle=":", label=f"Sell @ ${suggested_sell:.2f}")

        ax1.set_title(f"{ticker.upper()} - 5-Year Historical Price & MAs")
        ax1.set_ylabel("Price")
        ax1.set_xlabel("Date")
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        ax1.grid(True)
        st.pyplot(fig1)

        # --- 2) 기간 선택 슬라이더 ---
        st.subheader("📅 Select Period for Custom Chart (1 Month ~ 5 Years)")

        # 1개월 ~ 5년(252 trading days * 5 = 1260)
        period_days = st.slider("Select number of trading days to display:", min_value=21, max_value=1260, value=252, step=21)

        # 기간 필터링 (최근 period_days만)
        hist_period = hist.tail(period_days).copy()

        # 이동평균 다시 계산 (선택 기간이 적으면 rolling window 주의)
        hist_period["MA_6M"] = hist_period["Close"].rolling(window=126, min_periods=1).mean()
        hist_period["MA_1Y"] = hist_period["Close"].rolling(window=252, min_periods=1).mean()
        hist_period["MA_2Y"] = hist_period["Close"].rolling(window=504, min_periods=1).mean()

        # --- Custom Period Chart ---
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(hist_period.index, hist_period["Close"], label="Close Price", color="black", linewidth=1)
        ax2.plot(hist_period.index, hist_period["MA_6M"], label="6-Month MA", linestyle='--', color="orange")
        ax2.plot(hist_period.index, hist_period["MA_1Y"], label="1-Year MA", linestyle='--', color="green")
        ax2.plot(hist_period.index, hist_period["MA_2Y"], label="2-Year MA", linestyle='--', color="red")

        ax2.axhline(suggested_buy, color="red", linestyle=":", label=f"Buy @ ${suggested_buy:.2f}")
        ax2.axhline(suggested_sell, color="blue", linestyle=":", label=f"Sell @ ${suggested_sell:.2f}")

        ax2.set_title(f"{ticker.upper()} - Last {period_days} Trading Days Price & MAs")
        ax2.set_ylabel("Price")
        ax2.set_xlabel("Date")
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        ax2.grid(True)
        st.pyplot(fig2)

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






