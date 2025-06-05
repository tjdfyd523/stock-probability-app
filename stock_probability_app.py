import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 설정
st.set_page_config(page_title="📈 MA Strategy Backtest", layout="centered")
st.title("📈 MA Crossover Buy Signal + Backtest Simulation")

# 입력
ticker = st.text_input("Enter stock ticker (e.g., AAPL, SOXL, 005930.KS)", value="AAPL")

@st.cache_data
def load_price_history(ticker):
    return yf.Ticker(ticker).history(period="5y")

# 매수 신호: short_ma가 long_ma를 상향 돌파
def find_buy_signals(short_ma, long_ma):
    cross_up = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
    return short_ma[cross_up]

# 백테스트 (자동매매 시뮬레이션)
def backtest_strategy(hist, buy_signals):
    balance = 10000  # 초기 자산
    position = 0
    last_buy_price = 0
    trade_log = []

    for date in hist.index:
        price = hist.loc[date, 'Close']

        # 매수 시점
        if date in buy_signals.index and position == 0:
            position = balance / price
            last_buy_price = price
            balance = 0
            trade_log.append((date, "BUY", price))

        # 매도 시점: 6M < 1Y
        elif hist['MA_6M'][date] < hist['MA_1Y'][date] and position > 0:
            balance = position * price
            position = 0
            trade_log.append((date, "SELL", price))

    # 종료 시점에 정리
    if position > 0:
        balance = position * hist['Close'].iloc[-1]
        trade_log.append((hist.index[-1], "SELL (end)", hist['Close'].iloc[-1]))

    return balance, trade_log

# 상승/하락 확률 계산
def up_down_probability(hist, period_days):
    future_returns = hist["Close"].pct_change(periods=period_days).dropna()
    up_prob = (future_returns > 0).mean() * 100
    down_prob = 100 - up_prob
    return up_prob, down_prob

# 메인 실행
if ticker:
    try:
        hist = load_price_history(ticker)
        hist['MA_6M'] = hist['Close'].rolling(window=126).mean()
        hist['MA_1Y'] = hist['Close'].rolling(window=252).mean()
        hist['MA_2Y'] = hist['Close'].rolling(window=504).mean()

        current_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
        st.subheader(f"💰 Current Price: ${current_price:.2f}")

        # 매수 신호 탐색
        buy_signals = find_buy_signals(hist['MA_6M'], hist['MA_1Y'])

        # 📉 그래프
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(hist.index, hist['Close'], label='Close Price', color='black')
        ax.plot(hist.index, hist['MA_6M'], label='6-Month MA', color='orange', linestyle='--')
        ax.plot(hist.index, hist['MA_1Y'], label='1-Year MA', color='green', linestyle='--')
        ax.plot(hist.index, hist['MA_2Y'], label='2-Year MA', color='blue', linestyle='--')
        ax.scatter(buy_signals.index, buy_signals.values, marker='^', color='red', s=100, label='Buy Signal (6M > 1Y MA)')
        ax.set_title(f"{ticker.upper()} - Buy Signal Backtest")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=10)
        st.pyplot(fig)

        # 📊 자동매매 백테스트
        st.subheader("📊 Backtest Result")
        final_balance, trades = backtest_strategy(hist, buy_signals)
        st.markdown(f"📈 Final portfolio value: **${final_balance:,.2f}**")

        # 거래 로그 출력
        with st.expander("💼 Trade Log"):
            for date, action, price in trades:
                st.write(f"{date.date()} - {action} @ ${price:.2f}")

        # 📊 상승/하락 확률
        st.subheader("📈 Up/Down Probability Forecast")
        periods = {
            "1 Day": 1,
            "1 Week": 5,
            "1 Month": 21,
            "1 Year": 252,
        }
        for label, days in periods.items():
            up, down = up_down_probability(hist, days)
            st.markdown(
                f"**{label} Later →** Up: <span style='color:red'>{up:.2f}%</span>, "
                f"Down: <span style='color:blue'>{down:.2f}%</span>",
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"⚠️ Failed to load data for {ticker}. Error: {e}")


