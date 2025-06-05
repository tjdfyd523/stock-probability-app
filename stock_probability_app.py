import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 페이지 설정
st.set_page_config(page_title="📈 MA Strategy vs Buy & Hold", layout="centered")
st.title("📊 MA Crossover Strategy vs Buy & Hold Backtest")

# 입력
ticker = st.text_input("Enter stock ticker (e.g., AAPL, SOXL, 005930.KS)", value="AAPL")

@st.cache_data
def load_price_history(ticker):
    return yf.Ticker(ticker).history(period="5y")

def find_buy_signals(short_ma, long_ma):
    cross_up = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
    return short_ma[cross_up]

# 전략 백테스트
def backtest_strategy(hist, buy_signals):
    balance = 10000
    position = 0
    value_list = []
    trade_log = []
    in_trade = False

    for date in hist.index:
        price = hist.loc[date, 'Close']

        # 매수
        if date in buy_signals.index and not in_trade:
            position = balance / price
            balance = 0
            in_trade = True
            trade_log.append((date, "BUY", price))

        # 매도
        elif hist['MA_6M'][date] < hist['MA_1Y'][date] and in_trade:
            balance = position * price
            position = 0
            in_trade = False
            trade_log.append((date, "SELL", price))

        portfolio_value = balance if not in_trade else position * price
        value_list.append((date, portfolio_value))

    # 마지막날 정리
    if in_trade:
        balance = position * hist['Close'].iloc[-1]
        trade_log.append((hist.index[-1], "SELL (end)", hist['Close'].iloc[-1]))
        value_list[-1] = (hist.index[-1], balance)

    return balance, trade_log, pd.DataFrame(value_list, columns=["Date", "Portfolio"]).set_index("Date")

# Buy & Hold 계산
def buy_and_hold(hist, initial=10000):
    start_price = hist['Close'].iloc[0]
    units = initial / start_price
    hist['BuyHold'] = units * hist['Close']
    return hist['BuyHold']

# 확률 예측
def up_down_probability(hist, period_days):
    future_returns = hist["Close"].pct_change(periods=period_days).dropna()
    up_prob = (future_returns > 0).mean() * 100
    return up_prob, 100 - up_prob

# 실행
if ticker:
    try:
        hist = load_price_history(ticker)
        hist['MA_6M'] = hist['Close'].rolling(window=126).mean()
        hist['MA_1Y'] = hist['Close'].rolling(window=252).mean()
        hist['MA_2Y'] = hist['Close'].rolling(window=504).mean()

        current_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
        st.subheader(f"💰 Current Price: ${current_price:.2f}")

        # 매수 신호
        buy_signals = find_buy_signals(hist['MA_6M'], hist['MA_1Y'])

        # 백테스트 실행
        final_val, trades, strategy_df = backtest_strategy(hist, buy_signals)
        buy_hold = buy_and_hold(hist)

        # 그래프
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(hist.index, hist['Close'], label='Close Price', color='black')
        ax.plot(hist.index, hist['MA_6M'], label='6M MA', linestyle='--', color='orange')
        ax.plot(hist.index, hist['MA_1Y'], label='1Y MA', linestyle='--', color='green')
        ax.plot(hist.index, hist['MA_2Y'], label='2Y MA', linestyle='--', color='blue')
        ax.scatter(buy_signals.index, buy_signals.values, marker='^', color='red', s=100, label='Buy Signal')
        ax.set_title(f"{ticker.upper()} - Price & Buy Signals")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3)
        ax.grid(True)
        st.pyplot(fig)

        # 수익률 그래프
        st.subheader("📈 Strategy vs Buy & Hold Performance")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(strategy_df.index, strategy_df['Portfolio'], label='MA Strategy', color='red')
        ax2.plot(buy_hold.index, buy_hold.values, label='Buy & Hold', color='blue')
        ax2.set_ylabel("Portfolio Value ($)")
        ax2.set_title("Backtest: Strategy vs Buy & Hold")
        ax2.legend(loc="upper left")
        ax2.grid(True)
        st.pyplot(fig2)

        st.markdown(f"📊 Final MA Strategy Value: **${final_val:,.2f}**")
        st.markdown(f"📊 Final Buy & Hold Value: **${buy_hold.iloc[-1]:,.2f}**")

        with st.expander("📋 Trade Log"):
            for date, action, price in trades:
                st.write(f"{date.date()} - {action} @ ${price:.2f}")

        st.subheader("📊 Up/Down Probability")
        periods = {"1 Day": 1, "1 Week": 5, "1 Month": 21, "1 Year": 252}
        for label, days in periods.items():
            up, down = up_down_probability(hist, days)
            st.markdown(
                f"**{label} Later →** Up: <span style='color:red'>{up:.2f}%</span>, "
                f"Down: <span style='color:blue'>{down:.2f}%</span>",
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"⚠️ Error loading data: {e}")



