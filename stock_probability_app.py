import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit config
st.set_page_config(page_title="📈 Stock Signal Chart", layout="centered")
st.title("📈 Buy/Sell Signal Chart (High Probability Zones Only)")

# Ticker input
ticker = st.text_input("Enter stock ticker (e.g., AAPL, 005930.KS)", value="AAPL")

@st.cache_data
def load_price_history(ticker):
    return yf.Ticker(ticker).history(period="5y")

# Find signals with high confidence
def find_trade_signals_with_confidence(df, window=20, threshold=0.75):
    buy_signals = []
    sell_signals = []

    for i in range(window, len(df)):
        sub_df = df.iloc[i-window:i]
        up_prob = (sub_df["Close"].pct_change() > 0).mean()
        down_prob = 1 - up_prob

        current_price = df["Close"].iloc[i]
        recent_20_low = df["Close"].iloc[i-window:i].min()
        recent_10_high = df["Close"].iloc[i-10:i].max()
        ma_5 = df["Close"].rolling(window=5).mean().iloc[i]

        if current_price >= recent_20_low * 1.10 and up_prob >= threshold:
            buy_signals.append((df.index[i], current_price))

        elif (current_price <= recent_10_high * 0.93 or current_price < ma_5) and down_prob >= threshold:
            sell_signals.append((df.index[i], current_price))

    return buy_signals, sell_signals

if ticker:
    try:
        hist = load_price_history(ticker)
        stock = yf.Ticker(ticker)
        current_price = stock.history(period="1d")["Close"].iloc[-1]

        # Add moving averages for context
        hist["MA_6M"] = hist["Close"].rolling(window=126).mean()
        hist["MA_1Y"] = hist["Close"].rolling(window=252).mean()

        # Find signals
        buy_signals, sell_signals = find_trade_signals_with_confidence(hist)

        st.subheader(f"💰 Current Price: ${current_price:.2f}")

        # Plotting
        st.subheader("📊 Price Chart with High-Confidence Buy/Sell Signals")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hist.index, hist["Close"], label="Close Price", color="black", linewidth=1)
        ax.plot(hist.index, hist["MA_6M"], label="6M MA", linestyle='--', color="orange")
        ax.plot(hist.index, hist["MA_1Y"], label="1Y MA", linestyle='--', color="green")

        # Buy signals
        for date, price in buy_signals:
            ax.scatter(date, price, color="green", marker="^", s=120, label="Buy Signal" if date == buy_signals[0][0] else "")
            ax.text(date, price * 1.01, f"${price:.2f}", color="green", fontsize=8, ha='center')

        # Sell signals
        for date, price in sell_signals:
            ax.scatter(date, price, color="red", marker="v", s=120, label="Sell Signal" if date == sell_signals[0][0] else "")
            ax.text(date, price * 0.99, f"${price:.2f}", color="red", fontsize=8, ha='center')

        ax.set_title(f"{ticker.upper()} Price Chart (High-Confidence Signals)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left")
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Failed to fetch data for ticker `{ticker}`.\n\nDetails: {e}")




