import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 일본풍 스타일 적용
st.markdown("""
    <style>
        .main {
            background-color: #fdfcf7;
            font-family: 'Yu Gothic', 'Meiryo', sans-serif;
        }
        .title {
            color: #e60012;
            font-size: 2.5em;
            text-align: center;
        }
        .subheader {
            color: #e60012;
            font-size: 1.3em;
        }
        .buy-signal {
            color: #d7000f;
            font-weight: bold;
        }
        .sell-signal {
            color: #005bac;
            font-weight: bold;
        }
        .highlight {
            background-color: #fff0f0;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# 제목
st.markdown('<div class="title">📈 Stock Forecast with Buy/Sell Signals</div>', unsafe_allow_html=True)

# 사용자 입력
ticker = st.text_input("Enter stock ticker (e.g., AAPL, 005930.KS)", value="AAPL")

@st.cache_data
def load_price_history(ticker):
    return yf.Ticker(ticker).history(period="5y")

# 매수/매도 판단 함수
def find_trade_signals(df):
    buy_signals = []
    sell_signals = []
    in_position = False

    for i in range(20, len(df)):
        recent_20_low = df["Close"].iloc[i-20:i].min()
        recent_10_high = df["Close"].iloc[i-10:i].max()
        today_close = df["Close"].iloc[i]
        ma_5 = df["Close"].rolling(window=5).mean().iloc[i]

        # 매수 조건
        if not in_position and today_close >= recent_20_low * 1.10:
            buy_signals.append(df.index[i])
            in_position = True

        # 매도 조건
        elif in_position and (today_close <= recent_10_high * 0.93 or today_close < ma_5):
            sell_signals.append(df.index[i])
            in_position = False

    return buy_signals, sell_signals

if ticker:
    try:
        hist = load_price_history(ticker)
        stock = yf.Ticker(ticker)
        current_price = stock.history(period="1d")["Close"].iloc[-1]

        # 이동 평균선 계산
        hist["MA_6M"] = hist["Close"].rolling(window=126, min_periods=1).mean()
        hist["MA_1Y"] = hist["Close"].rolling(window=252, min_periods=1).mean()

        # 매수/매도 시점 찾기
        buy_signals, sell_signals = find_trade_signals(hist)

        # 현재 가격
        st.subheader(f"💰 Current Price: ${current_price:.2f}")

        # 차트 시각화
        st.subheader("📊 Price Chart with Buy/Sell Signals")

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(hist.index, hist["Close"], label="Close Price", color="black")
        ax.plot(hist.index, hist["MA_6M"], label="6M MA", linestyle='--', color="orange")
        ax.plot(hist.index, hist["MA_1Y"], label="1Y MA", linestyle='--', color="green")

        if buy_signals:
            ax.scatter(buy_signals, hist.loc[buy_signals]["Close"], color="red", marker="^", s=100, label="Buy Signal")
        if sell_signals:
            ax.scatter(sell_signals, hist.loc[sell_signals]["Close"], color="blue", marker="v", s=100, label="Sell Signal")

        ax.set_title(f"{ticker.upper()} Price & Trade Signals")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left")
        ax.grid(True)
        st.pyplot(fig)

        # 미래 예측
        st.subheader("🔮 Predicted Future Price (Based on Avg Daily Return)")

        daily_returns = hist["Close"].pct_change().dropna()
        avg_daily_return = daily_returns.mean()
        days = st.slider("Forecast days ahead:", min_value=1, max_value=90, value=30)

        future_price = current_price * ((1 + avg_daily_return) ** days)
        st.markdown(f"💡 Estimated price in {days} days: <span class='highlight'>${future_price:.2f}</span>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Failed to load data for `{ticker}`.\n\nDetails: {e}")




