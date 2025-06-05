import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

st.set_page_config(page_title="📈 Stock Up/Down Forecast with Rice Bowl Method", layout="centered")
st.title("📈 Stock Up/Down Probability with Real-Time Price & 밥그릇 기법")

ticker = st.text_input("Enter ticker (e.g., AAPL, 005930.KS)", value="SOXL")

@st.cache_data
def load_price_history(ticker):
    return yf.Ticker(ticker).history(period="5y")

def find_buy_signals(df):
    buy_signals = []
    for i in range(1, len(df)):
        if df["MA_6M"].iloc[i-1] < df["MA_1Y"].iloc[i-1] and df["MA_6M"].iloc[i] >= df["MA_1Y"].iloc[i]:
            buy_signals.append(df.index[i])
    return buy_signals

def find_rice_bowl_zone(df, window=30):
    """
    최근 window일간 종가의 밥그릇 구간(최저가~최고가) 계산
    밥그릇 하단: window 기간 내 최저가
    밥그릇 상단: window 기간 내 최고가
    """
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
        selected_period_months = st.selectbox("Select period to analyze:", options=month_options, format_func=lambda x: f"{x} Month{'s' if x>1 else ''}")

        period_days = selected_period_months * 21
        hist_period = hist.tail(period_days).copy()

        hist_period["MA_6M"] = hist_period["Close"].rolling(window=126, min_periods=1).mean()
        hist_period["MA_1Y"] = hist_period["Close"].rolling(window=252, min_periods=1).mean()
        hist_period["MA_2Y"] = hist_period["Close"].rolling(window=504, min_periods=1).mean()

        buy_signals_ma = find_buy_signals(hist_period)

        recent_peak = hist_period["Close"].max()
        suggested_sell = recent_peak * 0.9
        suggested_buy = hist_period["MA_6M"].iloc[-1]

        st.subheader(f"💰 Current Price: ${current_price:.2f}")
        st.markdown(f"📌 Suggested Buy Price: <span style='color:red; font-weight:bold'>${suggested_buy:.2f}</span>", unsafe_allow_html=True)
        st.markdown(f"📌 Suggested Sell Price: <span style='color:blue; font-weight:bold'>${suggested_sell:.2f}</span>", unsafe_allow_html=True)

        st.subheader(f"📊 Price Chart & Moving Averages - Last {selected_period_months} Month{'s' if selected_period_months > 1 else ''}")

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(hist_period.index, hist_period["Close"], label="Close Price", color="black", linewidth=1)
        ax.plot(hist_period.index, hist_period["MA_6M"], label="6-Month MA", linestyle='--', color="orange")
        ax.plot(hist_period.index, hist_period["MA_1Y"], label="1-Year MA", linestyle='--', color="green")
        ax.plot(hist_period.index, hist_period["MA_2Y"], label="2-Year MA", linestyle='--', color="red")

        ax.axhline(suggested_buy, color="red", linestyle=":", label=f"Suggested Buy (${suggested_buy:.2f})")
        ax.axhline(suggested_sell, color="blue", linestyle=":", label=f"Suggested Sell (${suggested_sell:.2f})")

        # 밥그릇 구간 찾기 (최근 30일 기준)
        bowl_start, bowl_end, bowl_low, bowl_high = find_rice_bowl_zone(hist_period, window=30)
        
        # 밥그릇 구간 사각형 표시
        ax.add_patch(
            patches.Rectangle(
                (bowl_start, bowl_low),
                bowl_end - bowl_start,
                bowl_high - bowl_low,
                linewidth=1,
                edgecolor='purple',
                facecolor='purple',
                alpha=0.1,
                label='밥그릇 구간'
            )
        )

        # 밥그릇 하단 매수 시그널
        buy_points = []
        df_bowl = hist_period.loc[bowl_start:bowl_end]
        for i in range(1, len(df_bowl)):
            prev_close = df_bowl["Close"].iloc[i-1]
            curr_close = df_bowl["Close"].iloc[i]
            if prev_close < bowl_low and curr_close >= bowl_low:
                buy_points.append(df_bowl.index[i])

        # 밥그릇 상단 돌파 매도 시그널
        sell_points = []
        for i in range(1, len(df_bowl)):
            prev_close = df_bowl["Close"].iloc[i-1]
            curr_close = df_bowl["Close"].iloc[i]
            if prev_close <= bowl_high and curr_close > bowl_high:
                sell_points.append(df_bowl.index[i])

        # 매수/매도 시그널 표시
        if buy_points:
            ax.scatter(buy_points, hist_period.loc[buy_points]["Close"], color="green", marker="^", s=120, label="밥그릇 매수 시그널")
        if sell_points:
            ax.scatter(sell_points, hist_period.loc[sell_points]["Close"], color="red", marker="v", s=120, label="밥그릇 매도 시그널")

        # 기존 MA 교차 매수 시그널도 표시
        if buy_signals_ma:
            ax.scatter(buy_signals_ma, hist_period.loc[buy_signals_ma]["Close"], color="orange", label="MA Cross Buy Signal", marker="^", s=100)

        ax.set_title(f"{ticker.upper()} Price & Moving Averages with 밥그릇 기법")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
        ax.grid(True)
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


