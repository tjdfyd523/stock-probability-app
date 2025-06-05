import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import talib

# --- 페이지 설정 ---
st.set_page_config(page_title="📈 주식 매매 예측", layout="centered")
st.title("📈 주식 매수/매도 예측 (ZigZag + 기술 지표)")

# --- 티커 입력 ---
ticker = st.text_input("티커를 입력하세요 (예: AAPL, 005930.KS)", value="SOXL")

@st.cache_data
def load_price_history(ticker):
    return yf.Ticker(ticker).history(period="5y")

if ticker:
    try:
        # 히스토리 데이터 로드
        hist = load_price_history(ticker)
        hist['Close'] = hist['Adj Close']
        hist['Volume'] = hist['Volume']

        # --- ZigZag 알고리즘을 통한 웨이브 분석 ---
        def calculate_zigzag(data, threshold):
            zigzag = pd.Series(index=data.index)
            trend = None
            last_pivot = None
            for i in range(1, len(data)):
                if trend is None:
                    trend = 'up' if data[i] > data[i-1] else 'down'
                    last_pivot = data[i]
                elif trend == 'up' and data[i] < last_pivot * (1 - threshold):
                    zigzag.iloc[i] = last_pivot
                    trend = 'down'
                    last_pivot = data[i]
                elif trend == 'down' and data[i] > last_pivot * (1 + threshold):
                    zigzag.iloc[i] = last_pivot
                    trend = 'up'
                    last_pivot = data[i]
            return zigzag

        # ZigZag 계산 (5% 기준)
        threshold = 0.05
        hist['ZigZag'] = calculate_zigzag(hist['Close'], threshold)

        # --- 기술 지표 계산 ---
        hist['RSI'] = talib.RSI(hist['Close'], timeperiod=14)
        hist['MACD'], hist['MACD_signal'], _ = talib.MACD(hist['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        hist['UpperBand'], hist['MiddleBand'], hist['LowerBand'] = talib.BBANDS(hist['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

        # --- 매수/매도 시점 계산 ---
        buy_signals = []
        sell_signals = []

        for i in range(1, len(hist)):
            # 매수 조건
            if hist['ZigZag'].iloc[i] > hist['ZigZag'].iloc[i-1] and hist['RSI'].iloc[i] < 30 and hist['MACD'].iloc[i] > hist['MACD_signal'].iloc[i]:
                buy_signals.append(hist['Close'].iloc[i])
                sell_signals.append(np.nan)
            # 매도 조건
            elif hist['ZigZag'].iloc[i] < hist['ZigZag'].iloc[i-1] and hist['RSI'].iloc[i] > 70 and hist['MACD'].iloc[i] < hist['MACD_signal'].iloc[i]:
                buy_signals.append(np.nan)
                sell_signals.append(hist['Close'].iloc[i])
            else:
                buy_signals.append(np.nan)
                sell_signals.append(np.nan)

        hist['Buy_Signal_price'] = buy_signals
        hist['Sell_Signal_price'] = sell_signals

        # --- 차트 그리기 ---
        st.subheader(f"{ticker.upper()}의 5년간 주가 및 매매 신호")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hist.index, hist['Close'], label="종가", color='black', linewidth=1)
        ax.plot(hist.index, hist['UpperBand'], label="상한선 (Bollinger)", linestyle='--', color='red')
        ax.plot(hist.index, hist['MiddleBand'], label="중앙선 (Bollinger)", linestyle='--', color='green')
        ax.plot(hist.index, hist['LowerBand'], label="하한선 (Bollinger)", linestyle='--', color='blue')

        ax.scatter(hist.index, hist['Buy_Signal_price'], label="매수 신호", marker='^', color='green', alpha=1)
        ax.scatter(hist.index, hist['Sell_Signal_price'], label="매도 신호", marker='v', color='red', alpha=1)

        ax.set_title(f"{ticker.upper()} - 5년간 주가 및 매매 신호")
        ax.set_xlabel("날짜")
        ax.set_ylabel("가격")
        ax.legend(loc='upper left')
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ 티커 `{ticker}`의 데이터를 가져오는 데 실패했습니다.\n\n자세한 내용: {e}")


