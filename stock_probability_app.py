import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit 기본 설정
st.set_page_config(page_title="📈 MA Crossover Buy Signal App", layout="centered")
st.title("📈 Buy Signals with MA Crossover + Probability Forecast")

# 사용자 입력
ticker = st.text_input("Enter stock ticker (e.g., AAPL, SOXL, 005930.KS)", value="AAPL")

# 데이터 불러오기
@st.cache_data
def load_price_history(ticker):
    return yf.Ticker(ticker).history(period="5y")

# 매수 시그널: 6개월 MA가 1년 MA를 상향 돌파할 때
def find_buy_signals(short_ma, long_ma):
    cross_up = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
    return short_ma[cross_up]

# 확률 계산 함수
def up_down_probability(hist, period_days):
    future_returns = hist["Close"].pct_change(periods=period_days).dropna()
    up_prob = (future_returns > 0).mean() * 100
    down_prob = 100 - up_prob
    return up_prob, down_prob

# 메인 로직
if ticker:
    try:
        hist = load_price_history(ticker)
        hist['MA_6M'] = hist['Close'].rolling(window=126).mean()
        hist['MA_1Y'] = hist['Close'].rolling(window=252).mean()

        buy_signals = find_buy_signals(hist['MA_6M'], hist['MA_1Y'])

        # 그래프 시각화
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(hist.index, hist['Close'], label='Close Price', color='black', linewidth=1)
        ax.plot(hist.index, hist['MA_6M'], label='6-Month MA', color='orange', linestyle='--')
        ax.plot(hist.index, hist['MA_1Y'], label='1-Year MA', color='green', linestyle='--')

        # 매수 신호 표시
        ax.scatter(buy_signals.index, buy_signals.values, marker='^', color='red', s=100, label='Buy Signal (6M > 1Y MA)')

        # 그래프 제목 및 범례
        ax.set_title(f"{ticker.upper()} Buy Signals (5-Year History)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=10)
        st.pyplot(fig)

        # 확률 예측
        st.subheader(f"{ticker.upper()} Up/Down Probability Forecast")
        periods = {
            "1 Day": 1,
            "1 Week": 5,
            "1 Month": 21,
            "1 Year": 252,
        }

        for label, days in periods.items():
            up, down = up_down_probability(hist, days)
            st.markdown(
                f"**{label} Later →** Up Probability: "
                f"<span style='color:red'>{up:.2f}%</span>, "
                f"Down Probability: <span style='color:blue'>{down:.2f}%</span>",
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"⚠️ Failed to load data for {ticker}. Error: {e}")


