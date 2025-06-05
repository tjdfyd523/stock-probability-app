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

def calculate_up_down_probability(df, window=20):
    """
    Calculate the up and down probabilities based on the recent 'window' period
    """
    returns = df["Close"].pct_change()
    up_probs = returns.rolling(window).apply(lambda x: (x > 0).mean())
    down_probs = 1 - up_probs
    return up_probs, down_probs

def find_confident_signals(df, up_probs, down_probs, threshold=0.75):
    buy_signals = []
    sell_signals = []

    # Filter only signals within the last 2 years
    cutoff_date = df.index[-1] - pd.DateOffset(years=2)

    for i in range(len(df)):
        if df.index[i] < cutoff_date:
            continue

        price = df["Close"].iloc[i]

        # Sell signal condition (75% probability of down movement)
        recent_20_low = df["Close"].iloc[max(0,i-20):i+1].min()
        if down_probs.iloc[i] is not np.nan and down_probs.iloc[i] >= threshold and price <= recent_20_low * 0.93:
            sell_signals.append((df.index[i], price))

        # Buy signal condition (75% probability of up movement)
        recent_10_high = df["Close"].iloc[max(0,i-10):i+1].max()
        ma_5 = df["Close"].rolling(window=5).mean().iloc[i]
        if up_probs.iloc[i] is not np.nan and up_probs.iloc[i] >= threshold and price >= recent_10_high * 1.10:
            buy_signals.append((df.index[i], price))

    return buy_signals, sell_signals

if ticker:
    try:
        hist = load_price_history(ticker)

        current_price = hist["Close"].iloc[-1]

        month_options = [1, 2, 3, 4, 5, 6, 9, 12, 24, 36, 48, 60]
        selected_period_months = st.selectbox("Select period to analyze:", options=month_options,
                                              format_func=lambda x: f"{x} Month{'s' if x>1 else ''}")

        period_days = selected_period_months * 21
        hist_period = hist.tail(period_days).copy()

        # Calculate moving averages
        hist_period["MA_6M"] = hist_period["Close"].rolling(window=126, min_periods=1).mean()
        hist_period["MA_1Y"] = hist_period["Close"].rolling(window=252, min_periods=1).mean()
        hist_period["MA_2Y"] = hist_period["Close"].rolling(window=504, min_periods=1).mean()

        # Calculate up and down probabilities (20-day window)
        up_probs, down_probs = calculate_up_down_probability(hist_period, window=20)

        # Find buy and sell signals based on 75% confidence
        buy_signals, sell_signals = find_confident_signals(hist_period, up_probs, down_probs, threshold=0.75)

        st.subheader(f"💰 Current Price: ${current_price:.2f}")

        st.subheader(f"📊 Price Chart & Moving Averages - Last {selected_period_months} Month{'s' if selected_period_months > 1 else ''}")

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(hist_period.index, hist_period["Close"], label="Close Price", color="black", linewidth=1)
        ax.plot(hist_period.index, hist_period["MA_6M"], label="6-Month MA", linestyle='--', color="orange")
        ax.plot(hist_period.index, hist_period["MA_1Y"], label="1-Year MA", linestyle='--', color="green")
        ax.plot(hist_period.index, hist_period["MA_2Y"], label="2-Year MA", linestyle='--', color="red")

        # Display buy signals (now sell signals in original code)
        if sell_signals:
            sell_dates, sell_prices = zip(*sell_signals)
            ax.scatter(sell_dates, sell_prices, color="blue", label="Sell Signal", marker="v", s=100)
            for d, p in sell_signals:
                ax.text(d, p * 1.01, f"${p:.2f}", color="blue", fontsize=9, fontweight="bold", ha='center')

        # Display sell signals (now buy signals in original code)
        if buy_signals:
            buy_dates, buy_prices = zip(*buy_signals)
            ax.scatter(buy_dates, buy_prices, color="red", label="Buy Signal", marker="^", s=100)
            for d, p in buy_signals:
                ax.text(d, p * 0.99, f"${p:.2f}", color="red", fontsize=9, fontweight="bold", ha='center')

        ax.set_title(f"{ticker.upper()} Price & Moving Averages")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
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

