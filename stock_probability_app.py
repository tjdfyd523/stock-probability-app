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

# 밥그릇 구간과 시그널을 슬라이딩 윈도우로 찾기
def find_rice_bowl_signals(df, window=30, step=5):
    bowl_rects = []
    buy_signals = []
    sell_signals = []

    for start_idx in range(0, len(df) - window + 1, step):
        window_df = df.iloc[start_idx:start_idx + window]
        start_date = window_df.index[0]
        end_date = window_df.index[-1]
        low = window_df["Close"].min()
        high = window_df["Close"].max()

        bowl_rects.append((start_date, end_date, low, high))

        # buy signal: 이전일 종가가 bowl low 밑, 당일 종가가 bowl low 이상일 때
        for i in range(1, len(window_df)):
            prev_close = window_df["Close"].iloc[i-1]
            curr_close = window_df["Close"].iloc[i]
            curr_date = window_df.index[i]
            if prev_close < low and curr_close >= low:
                buy_signals.append(curr_date)

        # sell signal: 이전일 종가가 bowl high 이하, 당일 종가가 bowl high 초과일 때
        for i in range(1, len(window_df)):
            prev_close = window_df["Close"].iloc[i-1]
            curr_close = window_df["Close"].iloc[i]
            curr_date = window_df.index[i]
            if prev_close <= high and curr_close > high:
                sell_signals.append(curr_date)

    # 중복 제거
    buy_signals = list(pd.Series(buy_signals).drop_duplicates())
    sell_signals = list(pd.Series(sell_signals).drop_duplicates())

    return bowl_rects, buy_signals, sell_signals

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

        # 밥그릇 구간, 매수/매도 시그널 찾기
        bowl_rects, bowl_buy_signals, bowl_sell_signals = find_rice_bowl_signals(hist_period, window=30, step=5)

        # 밥그릇 구간 그리기
        for (start_date, end_date, low, high) in bowl_rects:
            start_num = date2num(start_date)
            end_num = date2num(end_date)
            width = end_num - start_num
            rect = patches.Rectangle(
                (start_num, low),
                width,
                high - low,
                linewidth=1,
                edgecolor='purple',
                facecolor='purple',
                alpha=0.07
            )
            ax.add_patch(rect)

        # 밥그릇 매수 신호 표시
        if bowl_buy_signals:
            ax.scatter(bowl_buy_signals, hist_period.loc[bowl_buy_signals]["Close"], color="green", marker="^", s=120, label="Rice Bowl Buy Signal")

        # 밥그릇 매도 신호 표시
        if bowl_sell_signals:
            ax.scatter(bowl_sell_signals, hist_period.loc[bowl_sell_signals]["Close"], color="red", marker="v", s=120, label="Rice Bowl Sell Signal")

        # MA 교차 매수 시그널 표시
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



