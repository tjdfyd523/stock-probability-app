import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="📈 Stock Up/Down Forecast", layout="centered")
st.title("📈 Stock Up/Down Probability with Real-Time Price")

ticker = st.text_input("Enter ticker (e.g., AAPL, 005930.KS)", value="SOXL")

@st.cache_data
def load_price_history(ticker):
    return yf.Ticker(ticker).history(period="5y")

def find_buy_signals(df):
    # 6M MA가 1Y MA를 위로 교차하는 시점
    buy_signals = []
    for i in range(1, len(df)):
        if df["MA_6M"].iloc[i-1] < df["MA_1Y"].iloc[i-1] and df["MA_6M"].iloc[i] >= df["MA_1Y"].iloc[i]:
            buy_signals.append(df.index[i])
    return buy_signals

if ticker:
    try:
        hist = load_price_history(ticker)
        stock = yf.Ticker(ticker)
        current_price = stock.history(period="1d")["Close"].iloc[-1]

        # 기간 선택 (1개월 ~ 5년, 1개월 단위)
        month_options = [1, 2, 3, 4, 5, 6, 9, 12, 24, 36, 48, 60]
        month_label = [f"{m} Month{'s' if m>1 else ''}" if m<12 else f"{m//12} Year{'s' if m//12>1 else ''}" for m in month_options]
        selected_period_months = st.selectbox("Select period to analyze:", options=month_options, format_func=lambda x: f"{x} Month{'s' if x>1 else ''}")

        # 월 -> 거래일 근사 변환 (21 거래일/월)
        period_days = selected_period_months * 21
        hist_period = hist.tail(period_days).copy()

        # 이동평균 계산 (window 크기가 기간보다 크면 min_periods=1)
        hist_period["MA_6M"] = hist_period["Close"].rolling(window=126, min_periods=1).mean()
        hist_period["MA_1Y"] = hist_period["Close"].rolling(window=252, min_periods=1).mean()
        hist_period["MA_2Y"] = hist_period["Close"].rolling(window=504, min_periods=1).mean()

        # 매수 신호 (6M MA가 1Y MA를 골든크로스하는 시점)
        buy_signals = find_buy_signals(hist_period)
        buy_prices = hist_period.loc[buy_signals]["Close"] if buy_signals else pd.Series(dtype=float)

        # 매도 권장가: 최근 상승 파동 최고점의 90%
        # 최고점은 기간 내 최고점
        recent_peak = hist_period["Close"].max()
        suggested_sell = recent_peak * 0.9

        # 매수 권장가: 해당 기간 마지막 6M MA 값 (또는 직전 매수 포인트 평균)
        # 여기선 마지막 6M MA 사용
        suggested_buy = hist_period["MA_6M"].iloc[-1]

        # 현재 가격 표시
        st.subheader(f"💰 Current Price: ${current_price:.2f}")

        st.markdown(f"📌 Suggested Buy Price: <span style='color:red; font-weight:bold'>${suggested_buy:.2f}</span>", unsafe_allow_html=True)
        st.markdown(f"📌 Suggested Sell Price: <span style='color:blue; font-weight:bold'>${suggested_sell:.2f}</span>", unsafe_allow_html=True)

        # 그래프 그리기
        st.subheader(f"📊 Price Chart & Moving Averages - Last {selected_period_months} Month{'s' if selected_period_months > 1 else ''}")

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(hist_period.index, hist_period["Close"], label="Close Price", color="black", linewidth=1)
        ax.plot(hist_period.index, hist_period["MA_6M"], label="6-Month MA", linestyle='--', color="orange")
        ax.plot(hist_period.index, hist_period["MA_1Y"], label="1-Year MA", linestyle='--', color="green")
        ax.plot(hist_period.index, hist_period["MA_2Y"], label="2-Year MA", linestyle='--', color="red")

        ax.axhline(suggested_buy, color="red", linestyle=":", label=f"Suggested Buy (${suggested_buy:.2f})")
        ax.axhline(suggested_sell, color="blue", linestyle=":", label=f"Suggested Sell (${suggested_sell:.2f})")

        # 매수 신호 포인트 (Close price 기준)
        if len(buy_signals) > 0:
            ax.scatter(buy_signals, hist_period.loc[buy_signals]["Close"], color="red", label="Buy Signal", marker="^", s=100)

        ax.set_title(f"{ticker.upper()} Price & Moving Averages")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
        ax.grid(True)
        st.pyplot(fig)

        # --- 확률 예측 ---
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

        # --- 예측 가격 ---
        st.subheader("📈 Predicted Future Prices (Based on Average Daily Return)")

        daily_returns = hist["Close"].pct_change().dropna()
        avg_daily_return = daily_returns.mean()

        for label, days in periods_prob.items():
            predicted_price = current_price * ((1 + avg_daily_return) ** days)
            st.markdown(f"💡 **{label} Later:** `${predicted_price:.2f}`")

    except Exception as e:
        st.error(f"⚠️ Failed to fetch data for ticker `{ticker}`.\n\nDetails: {e}")

