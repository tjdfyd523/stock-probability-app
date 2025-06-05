import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="📈 Stock Up/Down Forecast (MA Cross)", layout="centered")
st.title("📈 이동평균선 교차 기반 매수/매도 권장가")

ticker = st.text_input("Enter ticker (e.g., AAPL, 005930.KS)", value="SOXL")

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

if ticker:
    try:
        hist = load_price_history(ticker)
        current_price = hist['Close'].iloc[-1]

        hist["MA_6M"] = hist["Close"].rolling(window=126).mean()
        hist["MA_1Y"] = hist["Close"].rolling(window=252).mean()
        hist["MA_2Y"] = hist["Close"].rolling(window=504).mean()

        cross_6m_1y = detect_cross(hist["MA_6M"], hist["MA_1Y"])
        cross_1y_2y = detect_cross(hist["MA_1Y"], hist["MA_2Y"])

        buy1_dates = cross_6m_1y[cross_6m_1y == 'up'].index
        buy1_prices = hist.loc[buy1_dates, "Close"]

        buy2_dates = cross_1y_2y[cross_1y_2y == 'up'].index
        buy2_prices = hist.loc[buy2_dates, "Close"]

        sell1_price = None
        sell2_price = None

        if not buy1_prices.empty:
            buy1_date = buy1_prices.index[-1]
            max_after_buy1 = hist.loc[buy1_date:]["Close"].cummax()
            current_close = hist.loc[buy1_date:]["Close"]
            drop_10 = max_after_buy1 * 0.9
            condition_10 = current_close < drop_10
            if condition_10.any():
                sell1_price = current_close[condition_10].iloc[0]
                sell1_date = current_close[condition_10].index[0]
            else:
                sell1_date = None
        else:
            sell1_date = None

        if not buy2_prices.empty:
            buy2_date = buy2_prices.index[-1]
            max_after_buy2 = hist.loc[buy2_date:]["Close"].cummax()
            current_close2 = hist.loc[buy2_date:]["Close"]
            drop_15 = max_after_buy2 * 0.85
            condition_15 = current_close2 < drop_15
            if condition_15.any():
                sell2_price = current_close2[condition_15].iloc[0]
                sell2_date = current_close2[condition_15].index[0]
            else:
                sell2_date = None
        else:
            sell2_date = None

        st.subheader(f"💰 현재가: ${current_price:.2f}")

        if not buy1_prices.empty:
            st.markdown(f"📌 1차 매수 권장가 (6M MA가 1Y MA 위로 교차): ${buy1_prices.iloc[-1]:.2f}")
        else:
            st.markdown("📌 1차 매수 권장가 신호가 없습니다.")

        if not buy2_prices.empty:
            st.markdown(f"📌 2차 매수 권장가 (1Y MA가 2Y MA 위로 교차): ${buy2_prices.iloc[-1]:.2f}")
        else:
            st.markdown("📌 2차 매수 권장가 신호가 없습니다.")

        if sell1_price is not None:
            st.markdown(f"📌 1차 매도 권장가 (1차 매수 후 최고가 대비 10% 하락): ${sell1_price:.2f}")
        else:
            st.markdown("📌 1차 매도 권장가 신호가 없습니다.")

        if sell2_price is not None:
            st.markdown(f"📌 2차 매도 권장가 (2차 매수 후 최고가 대비 15% 하락): ${sell2_price:.2f}")
        else:
            st.markdown("📌 2차 매도 권장가 신호가 없습니다.")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(hist.index, hist["Close"], label="종가", color="black")
        ax.plot(hist.index, hist["MA_6M"], label="6개월 MA", color="orange", linestyle="--")
        ax.plot(hist.index, hist["MA_1Y"], label="1년 MA", color="green", linestyle="--")
        ax.plot(hist.index, hist["MA_2Y"], label="2년 MA", color="red", linestyle="--")

        # 매수 권장가: 위쪽 화살표, 빨간색 계열
        ax.scatter(buy1_prices.index, buy1_prices.values, marker="^", color="darkred", s=120, label="1차 매수 권장가")
        ax.scatter(buy2_prices.index, buy2_prices.values, marker="^", color="pink", s=120, label="2차 매수 권장가")

        # 매도 권장가: 아래쪽 화살표, 파란색 계열
        if sell1_price is not None and sell1_date is not None:
            ax.scatter([sell1_date], [sell1_price], marker="v", color="blue", s=120, label="1차 매도 권장가")
        if sell2_price is not None and sell2_date is not None:
            ax.scatter([sell2_date], [sell2_price], marker="v", color="deepskyblue", s=120, label="2차 매도 권장가")

        ax.set_title(f"{ticker.upper()} 가격 및 이동평균선 교차 매매 신호")
        ax.set_xlabel("날짜")
        ax.set_ylabel("가격")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ 데이터 처리 중 오류가 발생했습니다: {e}")

