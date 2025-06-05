import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="📈 Stock Up/Down & MA Crossover Signals", layout="centered")
st.title("📈 Stock Up/Down Probabilities & Moving Average Crossover Signals")

ticker = st.text_input("Enter stock ticker (e.g., AAPL, 005930.KS)", value="SOXL")

@st.cache_data
def load_price_history(ticker):
    return yf.Ticker(ticker).history(period="5y")

def detect_cross(ma_short, ma_long):
    cross_up = (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
    cross_down = (ma_short < ma_long) & (ma_short.shift(1) >= ma_long.shift(1))
    result = pd.Series(index=ma_short.index, data=None)
    result[cross_up] = 'up'
    result[cross_down] = 'down'
    return result

def get_cross_points(ma_short, ma_long):
    crosses = detect_cross(ma_short, ma_long)
    cross_dates = crosses.dropna().index
    cross_vals = []
    for date in cross_dates:
        val = (ma_short.loc[date] + ma_long.loc[date]) / 2
        cross_vals.append(val)
    return cross_dates, cross_vals, crosses.dropna().values

def up_down_probability(hist, period_days):
    future_returns = hist["Close"].pct_change(periods=period_days).dropna()
    up_prob = (future_returns > 0).mean() * 100
    down_prob = 100 - up_prob
    return up_prob, down_prob

if ticker:
    try:
        hist = load_price_history(ticker)
        current_price = hist['Close'].iloc[-1]

        hist["MA_6M"] = hist["Close"].rolling(window=126).mean()
        hist["MA_1Y"] = hist["Close"].rolling(window=252).mean()
        hist["MA_2Y"] = hist["Close"].rolling(window=504).mean()

        # Moving average cross points
        buy1_dates, buy1_vals, buy1_types = get_cross_points(hist["MA_6M"], hist["MA_1Y"])
        buy2_dates, buy2_vals, buy2_types = get_cross_points(hist["MA_1Y"], hist["MA_2Y"])

        buy1_price = buy1_vals[-1] if len(buy1_vals) > 0 else None
        buy2_price = buy2_vals[-1] if len(buy2_vals) > 0 else None

        # Sell signals based on price drops after buys
        sell1_price, sell1_date = None, None
        if buy1_dates.size > 0:
            buy1_date = buy1_dates[-1]
            max_after_buy1 = hist.loc[buy1_date:]["Close"].cummax()
            current_close = hist.loc[buy1_date:]["Close"]
            drop_10 = max_after_buy1 * 0.9
            condition_10 = current_close < drop_10
            if condition_10.any():
                sell1_price = current_close[condition_10].iloc[0]
                sell1_date = current_close[condition_10].index[0]

        sell2_price, sell2_date = None, None
        if buy2_dates.size > 0:
            buy2_date = buy2_dates[-1]
            max_after_buy2 = hist.loc[buy2_date:]["Close"].cummax()
            current_close2 = hist.loc[buy2_date:]["Close"]
            drop_15 = max_after_buy2 * 0.85
            condition_15 = current_close2 < drop_15
            if condition_15.any():
                sell2_price = current_close2[condition_15].iloc[0]
                sell2_date = current_close2[condition_15].index[0]

        # Show current price and buy/sell recommendations
        st.subheader(f"💰 Current Price: ${current_price:.2f}")
        if buy1_price is not None:
            st.markdown(f"📌 1st Buy Recommendation (6M MA crosses above 1Y MA): ${buy1_price:.2f}")
        else:
            st.markdown("📌 No 1st buy signal found.")
        if buy2_price is not None:
            st.markdown(f"📌 2nd Buy Recommendation (1Y MA crosses above 2Y MA): ${buy2_price:.2f}")
        else:
            st.markdown("📌 No 2nd buy signal found.")
        if sell1_price is not None:
            st.markdown(f"📌 1st Sell Recommendation (10% drop from highest price after 1st buy): ${sell1_price:.2f}")
        else:
            st.markdown("📌 No 1st sell signal found.")
        if sell2_price is not None:
            st.markdown(f"📌 2nd Sell Recommendation (15% drop from highest price after 2nd buy): ${sell2_price:.2f}")
        else:
            st.markdown("📌 No 2nd sell signal found.")

        # Plot chart with MAs and buy/sell signals
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(hist.index, hist["Close"], label="Close Price", color="black")
        ax.plot(hist.index, hist["MA_6M"], label="6-Month MA", color="orange", linestyle="--")
        ax.plot(hist.index, hist["MA_1Y"], label="1-Year MA", color="green", linestyle="--")
        ax.plot(hist.index, hist["MA_2Y"], label="2-Year MA", color="red", linestyle="--")

        for date, val, typ in zip(buy1_dates, buy1_vals, buy1_types):
            if typ == 'up':
                ax.scatter(date, val, marker="^", color="darkred", s=120, label="1st Buy Signal" if date == buy1_dates[0] else "")
        for date, val, typ in zip(buy2_dates, buy2_vals, buy2_types):
            if typ == 'up':
                ax.scatter(date, val, marker="^", color="pink", s=120, label="2nd Buy Signal" if date == buy2_dates[0] else "")
        if sell1_price and sell1_date:
            ax.scatter(sell1_date, sell1_price, marker="v", color="blue", s=120, label="1st Sell Signal")
        if sell2_price and sell2_date:
            ax.scatter(sell2_date, sell2_price, marker="v", color="deepskyblue", s=120, label="2nd Sell Signal")

        ax.set_title(f"{ticker.upper()} Price and MA Crossover Signals")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=10)
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

        # --- Predicted Prices ---
        st.subheader("📈 Predicted Future Prices (Using Average Daily Return)")
        daily_returns = hist["Close"].pct_change().dropna()
        avg_daily_return = daily_returns.mean()
        for label, days in periods.items():
            predicted_price = current_price * ((1 + avg_daily_return) ** days)
            st.markdown(f"💡 **{label} Later:** `${predicted_price:.2f}`")

    except Exception as e:
        st.error(f"⚠️ Error processing data: {e}")



