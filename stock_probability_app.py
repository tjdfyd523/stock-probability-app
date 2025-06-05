import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="📈 Stock Up/Down Forecast (MA Cross)", layout="centered")
st.title("📈 이동평균선 교차 기반 매수/매도 권장가")

ticker = st.text_input("Enter ticker (예: AAPL, 005930.KS)", value="SOXL")

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

if ticker:
    try:
        hist = load_price_history(ticker)
        current_price = hist['Close'].iloc[-1]

        hist["MA_6M"] = hist["Close"].rolling(window=126).mean()
        hist["MA_1Y"] = hist["Close"].rolling(window=252).mean()
        hist["MA_2Y"] = hist["Close"].rolling(window=504).mean()

        buy1_dates, buy1_vals, buy1_types = get_cross_points(hist["MA_6M"], hist["MA_1Y"])
        buy2_dates, buy2_vals, buy2_types = get_cross_points(hist["MA_1Y"], hist["MA_2Y"])

        buy1_price = buy1_vals[-1] if len(buy1_vals) > 0 else None
        buy2_price = buy2_vals[-1] if len(buy2_vals) > 0 else None

        sell1_price = None
        sell1_date = None
        if buy1_dates.size > 0:
            buy1_date = buy1_dates[-1]
            max_after_buy1 = hist.loc[buy1_date:]["Close"].cummax()
            current_close = hist.loc[buy1_date:]["Close"]
            drop_10 = max_after_buy1 * 0.9
            condition_10 = current_close < drop_10
            if condition_10.any():
                sell1_price = current_close[condition_10].iloc[0]
                sell1_date = current_close[condition_10].index[0]

        sell2_price = None
        sell2_date = None
        if buy2_dates.size > 0:
            buy2_date = buy2_dates[-1]
            max_after_buy2 = hist.loc[buy2_date:]["Close"].cummax()
            current_close2 = hist.loc[buy2_date:]["Close"]
            drop_15 = max_after_buy2 * 0.85
            condition_15 = current_close2 < drop_15
            if condition_15.any():
                sell2_price = current_close2[condition_15].iloc[0]
                sell2_date = current_close2[condition_15].index[0]

        st.subheader(f"💰 현재가: ${current_price:.2f}")

        if buy1_price is not None:
            st.markdown(f"📌 1차 매수 권장가 (6M MA가 1Y MA 위로 교차): ${buy1_price:.2f}")
        else:
            st.markdown("📌 1차 매수 권장가 신호가 없습니다.")

        if buy2_price is not None:
            st.markdown(f"📌 2차 매수 권장가 (1Y MA가 2Y MA 위로 교차): ${buy2_price:.2f}")
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

        for date, val, typ in zip(buy1_dates, buy1_vals, buy1_types):
            if typ == 'up':
                ax.scatter(date, val, marker="^", color="darkred", s=120, label="1차 매수 권장가" if date == buy1_dates[0] else "")
        for date, val, typ in zip(buy2_dates, buy2_vals, buy2_types):
            if typ == 'up':
                ax.scatter(date, val, marker="^", color="pink", s=120, label="2차 매수 권장가" if date == buy2_dates[0] else "")

        if sell1_price is not None and sell1_date is not None:
            ax.scatter(sell1_date, sell1_price, marker="v", color="blue", s=120, label="1차 매도 권장가")
        if sell2_price is not None and sell2_date is not None:
            ax.scatter(sell2_date, sell2_price, marker="v", color="deepskyblue", s=120, label="2차 매도 권장가")

        ax.set_title(f"{ticker.upper()} 가격 및 이동평균선 교차 매매 신호")
        ax.set_xlabel("날짜")
        ax.set_ylabel("가격")
        ax.grid(True)

        # 범례를 그래프 오른쪽 밖으로 이동
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ 데이터 처리 중 오류가 발생했습니다: {e}")


