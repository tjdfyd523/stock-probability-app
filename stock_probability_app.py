import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

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

def calculate_predicted_price(current_price, avg_daily_return, days):
    """
    Predict future price based on average daily return
    """
    predicted_price = current_price * ((1 + avg_daily_return) ** days)
    return predicted_price

def calculate_price_probability(predicted_price, current_price, returns, days):
    """
    Calculate probability that the stock will reach the predicted price
    based on historical daily returns and their distribution.
    """
    # Calculate the daily returns' mean and std deviation
    daily_mean = returns.mean()
    daily_std = returns.std()

    # Calculate the expected future price based on the historical data
    future_return = (predicted_price / current_price) - 1

    # Calculate z-score (how many std deviations away from the mean)
    z_score = (future_return - daily_mean) / (daily_std * np.sqrt(days))

    # Calculate the cumulative probability of the predicted price being reached
    probability = norm.cdf(z_score)

    return probability

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

        st.subheader(f"💰 Current Price: ${current_price:.2f}")

        st.subheader(f"📊 Price Chart & Moving Averages - Last {selected_period_months} Month{'s' if selected_period_months > 1 else ''}")

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(hist_period.index, hist_period["Close"], label="Close Price", color="black", linewidth=1)
        ax.plot(hist_period.index, hist_period["MA_6M"], label="6-Month MA", linestyle='--', color="orange")
        ax.plot(hist_period.index, hist_period["MA_1Y"], label="1-Year MA", linestyle='--', color="green")
        ax.plot(hist_period.index, hist_period["MA_2Y"], label="2-Year MA", linestyle='--', color="red")

        ax.set_title(f"{ticker.upper()} Price & Moving Averages")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
        ax.grid(True)
        st.pyplot(fig)

        st.subheader("📈 Predicted Future Prices (Based on Average Daily Return)")

        daily_returns = hist["Close"].pct_change().dropna()
        avg_daily_return = daily_returns.mean()

        periods_prob = {
            "1 Day": 1,
            "1 Week": 5,
            "1 Month": 21,
            "1 Year": 252,
        }

        # Calculate predicted prices and probabilities for each period
        for label, days in periods_prob.items():
            predicted_price = calculate_predicted_price(current_price, avg_daily_return, days)
            probability = calculate_price_probability(predicted_price, current_price, daily_returns, days)

            # Display predicted price and the probability of reaching it
            st.markdown(f"💡 **{label} Later:** `${predicted_price:.2f}`")
            st.markdown(f"📊 **Probability of Reaching This Price:** <span style='color:red'>{probability * 100:.2f}%</span>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Failed to fetch data for ticker `{ticker}`.\n\nDetails: {e}")
