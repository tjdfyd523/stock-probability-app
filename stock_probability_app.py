import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="📈 MA Crossover + Peak-Based Sell Signals", layout="centered")
st.title("📈 Buy/Sell Signals with MA Crossover and Peak-Based Strategy")

ticker = st.text_input("Enter stock ticker (e.g., AAPL, SOXL, 005930.KS)", value="AAPL")

@st.cache_data
def load_price_history(ticker):
    return yf.Ticker(ticker).history(period="5y")

def find_cross_signals(short_ma, long_ma):
    cross_up = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
    cross_down = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
    buy_points = short_ma[cross_up]
    sell_points = short_ma[cross_down]
    return buy_points, sell_points

def up_down_probability(hist, period_days):
    future_returns = hist["Close"].pct_change(periods=period_days).dropna()
    up_prob = (future_returns > 0).mean() * 100
    down_prob = 100 - up_prob
    return up_prob, down_prob

def detect_peak_sell_signals(prices, threshold=0.10):
    peaks = []
    sells = []

    temp_min = prices[0]
    peak = prices[0]

    for i in range(1, len(prices)):
        price = prices[i]

        # New peak
        if price > peak:
            peak = price

        # Drop more than threshold from peak
        if price <= peak * (1 - threshold):
            peaks.append(peak)
            sells.append((prices.index[i], price))
            temp_min = price
            peak = price  # reset

    return sells

if ticker:
    try:
        hist = load_price_history(ticker)
        hist['MA_6M'] = hist['Close'].rolling(window=126).mean()
        hist['MA_1Y'] = hist['Close'].rolling(window=252).mean()

        buy_points, _ = find_cross_signals(hist['MA_6M'], hist['MA_1Y'])
        sell_signals = detect_peak_sell_signals(hist['Close'])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(hist.index, hist['Close'], label='Close Price', color='black', linewidth=1)
        ax.plot(hist.index, hist['MA_6M'], label='6-Month MA', color='orange', linestyle='--')
        ax.plot(hist.index, hist['MA_1Y'], label='1-Year MA', color='green', linestyle='--')

        # Buy signals (red ↑)
        ax.scatter(buy_points.index, buy_points.values, marker='^', color='red', s=100, label='Buy Signal (6M > 1Y MA)')

        # Sell signals (blue ↓)
        for sell_date, sell_price in sell_signals:
            ax.scatter(sell_date, sell_price, marker='v', color='blue', s=100)

        ax.set_title(f"{ticker.upper()} Buy/Sell Signals (5Y)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
        st.pyplot(fig)

        # --- Probabilities ---
        st.subheader(f"{ticker.upper()} Up/Down Probabilities")
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
        st.error(f"⚠️ Error loading data for {ticker}: {e}")




