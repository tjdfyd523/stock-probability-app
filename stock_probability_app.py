import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.dates import date2num

st.set_page_config(page_title="📈 Stock Forecast with Rice Bowl Method", layout="centered")
st.title("📈 Stock Price & Buy/Sell Signals with Rice Bowl Method")

ticker = st.text_input("Enter ticker (e.g., AAPL, 005930.KS)", value="SOXL")

@st.cache_data
def load_price_history(ticker):
    return yf.Ticker(ticker).history(period="5y")

def find_ma_buy_signals(df):
    buy_signals = []
    for i in range(1, len(df)):
        if df["MA_6M"].iloc[i-1] < df["MA_1Y"].iloc[i-1] and df["MA_6M"].iloc[i] >= df["MA_1Y"].iloc[i]:
            buy_signals.append(df.index[i])
    return buy_signals

def find_rice_bowl_zone(df, window=30):
    recent = df.tail(window)
    bowl_low = recent["Close"].min()
    bowl_high = recent["Close"].max()
    bowl_start = recent.index[0]
    bowl_end = recent.index[-1]
    return bowl_start, bowl_end, bowl_low, bowl_high

if ticker:
    try:
        hist = load_price_history(ticker)
        stock = yf.Ticker(ticker)
        current_price = stock.history(period="1d")["Close"].iloc[-1]

        month_options = [1, 2, 3, 4, 5, 6, 9, 12, 24, 36, 48, 60]
        selected_months = st.selectbox("Select period to analyze:", options=month_options, format_func=lambda x: f"{x} Month{'s' if x>1 else ''}")

        period_days = selected_months * 21
        hist_period = hist.tail(period_days).copy()

        hist_period["MA_6M"] = hist_period["Close"].rolling(window=126, min_periods=1).mean()
        hist_period["MA_1Y"] = hist_period["Close"].rolling(window=252, min_periods=1).mean()
        hist_period["MA_2Y"] = hist_period["Close"].rolling(window=504, min_periods=1).mean()

        ma_buy_signals = find_ma_buy_signals(hist_period)

        st.subheader(f"💰 Current Price: ${current_price:.2f}")

        st.subheader(f"📊 Price Chart & Moving Averages - Last {selected_months} Month{'s' if selected_months > 1 else ''}")

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(hist_period.index, hist_period["Close"], label="Close Price", color="black", linewidth=1)
        ax.plot(hist_period.index, hist_period["MA_6M"], label="6-Month MA", linestyle='--', color="orange")
        ax.plot(hist_period.index, hist_period["MA_1Y"], label="1-Year MA", linestyle='--', color="green")
        ax.plot(hist_period.index, hist_period["MA_2Y"], label="2-Year MA", linestyle='--', color="red")

        # Find rice bowl zone
        bowl_start, bowl_end, bowl_low, bowl_high = find_rice_bowl_zone(hist_period, window=30)

        # Convert dates to matplotlib float format for rectangle width
        bowl_start_num = date2num(bowl_start)
        bowl_end_num = date2num(bowl_end)
        width = bowl_end_num - bowl_start_num

        # Draw rectangle for rice bowl zone
        rect = patches.Rectangle(
            (bowl_start_num, bowl_low),
            width,
            bowl_high - bowl_low,
            linewidth=1,
            edgecolor='purple',
            facecolor='purple',
            alpha=0.1,
            label='Rice Bowl Zone'
        )
        ax.add_patch(rect)

        # Detect buy signals near bowl bottom: price crossing above bowl_low
        df_bowl = hist_period.loc[bowl_start:bowl_end]
        buy_points = []
        for i in range(1, len(df_bowl)):
            prev_close = df_bowl["Close"].iloc[i-1]
            curr_close = df_bowl["Close"].iloc[i]
            if prev_close < bowl_low and curr_close >= bowl_low:
                buy_points.append(df_bowl.index[i])

        # Detect sell signals: price crossing above bowl_high
        sell_points = []
        for i in range(1, len(df_bowl)):
            prev_close = df_bowl["Close"].iloc[i-1]
            curr_close = df_bowl["Close"].iloc[i]
            if prev_close <= bowl_high and curr_close > bowl_high:
                sell_points.append(df_bowl.index[i])

        # Plot buy/sell signals
        if buy_points:
            ax.scatter(buy_points, hist_period.loc[buy_points]["Close"], color="green", marker="^", s=120, label="Rice Bowl Buy Signal")
        if sell_points:
            ax.scatter(sell_points, hist_period.loc[sell_points]["Close"], color="red", marker="v", s=120, label="Rice Bowl Sell Signal")

        # Plot MA cross buy signals
        if ma_buy_signals:
            ax.scatter(ma_buy_signals, hist_period.loc[ma_buy_signals]["Close"], color="orange", marker="^", s=100, label="MA Cross Buy Signal")

        ax.set_title(f"{ticker.upper()} Price & Moving Averages with Rice Bowl Method")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
        ax.grid(True)
        plt.xticks(rotation=30)
        st.pyplot(fig)

        st.subheader(f"{ticker.upper()} Up/Down Probabilities")

        def up_down_probability(days):
            future_returns = hist["Close"].pct_change(periods=days).dropna()
            up_prob = (future_returns > 0).mean() * 100
            down_prob = 100 - up_prob
            return up_prob, down_prob

        periods_prob = {
            "1 Day": 1,
            "1 Week": 5,
            "1 Month": 21,
            "1 Year": 252,
        }

        for label, days in periods_prob.items():
            up, down = up_down_probability(days)
            st.markdown(
                f"**{label} Later →** Up Probability: "
                f"<span style='color:red'>{up:.2f}%</span>, "
                f"Down Probability: <span style='color:blue'>{down:.2f}%</span>",
                unsafe_allow_html=True
            )

        st.subheader("📈 Predicted Future Prices (Based on Average Daily Return)")

        daily_returns = hist["Close"].pct_change().dropna()
        avg_daily_return = daily_returns.mean()

        for label, days in periods_prob.items():
            predicted_price = current_price * ((1 + avg_daily_return) ** days)
            st.markdown(f"💡 **{label} Later:** `${predicted_price:.2f}`")

    except Exception as e:
        st.error(f"⚠️ Failed to fetch data for ticker `{ticker}`.\n\nDetails: {e}")



