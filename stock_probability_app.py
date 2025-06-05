import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 페이지 설정 ---
st.set_page_config(page_title="📈 주식 매매 예측 (Pandas)", layout="centered")
st.title("📈 주식 매수/매도 예측 (ZigZag + Pandas 기술 지표)")

# --- 티커 입력 ---
ticker = st.text_input("티커를 입력하세요 (예: AAPL, 005930.KS)", value="SOXL")

@st.cache_data
def load_price_history(ticker):
    return yf.Ticker(ticker).history(period="5y")

# --- 기술 지표 계산 함수들 ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_bollinger_bands(series, window=20):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    return upper, sma, lower

def zigzag_pandas(price_series, pct=5):
    peak = price_series[0]
    valley = price_series[0]
    trend = None
    zz = pd.Series(np.nan, index=price_series.index)

    for i in range(1, len(price_series)):
        change = price_series[i] / peak - 1 if trend == 'down' else price_series[i] / valley - 1
        if trend is None:
            if price_series[i] > peak * (1 + pct/100):
                trend = 'up'
                peak = price_series[i]
                zz.iloc[i] = peak
            elif price_series[i] < valley * (1 - pct/100):
                trend = 'down'
                valley = price_series[i]
                zz.iloc[i] = valley
        elif trend == 'up':
            if price_series[i] > peak:
                peak = price_series[i]
                zz.iloc[i] = peak
            elif price_series[i] < peak * (1 - pct/100):
                trend = 'down'
                valley = price_series[i]
                zz.iloc[i] = valley
        elif trend == 'down':
            if price_series[i] < valley:
                valley = price_series[i]
                zz.iloc[i] = valley
            elif price_series[i] > valley * (1 + pct/100):
                trend = 'up'
                peak = price_series[i]
                zz.iloc[i] = peak
    return zz

# --- 메인 로직 ---
if ticker:
    try:
        hist = load_price_history(ticker)

        # 종가 기준 설정 (Adj Close가 없으면 Close 사용)
        if 'Adj Close' in hist.columns:
            hist['Close'] = hist['Adj Close']
        else:
            hist['Close'] = hist['Close']

        # 기술 지표 계산
        hist['RSI'] = compute_rsi(hist['Close'])
        hist['MACD'], hist['MACD_signal'] = compute_macd(hist['Close'])
        hist['UpperBand'], hist['MiddleBand'], hist['LowerBand'] = compute_bollinger_bands(hist['Close'])
        hist['ZigZag'] = zigzag_pandas(hist['Close'], pct=5)

        # 매수/매도 신호 계산
        buy_signals = []
        sell_signals = []

        for i in range(1, len(hist)):
            # 매수 조건
            if (hist['ZigZag'].iloc[i] > hist['ZigZag'].iloc[i - 1] and
                hist['RSI'].iloc[i] < 30 and
                hist['MACD'].iloc[i] > hist['MACD_signal'].iloc[i]):
                buy_signals.append(hist['Close'].iloc[i])
                sell_signals.append(np.nan)
            # 매도 조건
            elif (hist['ZigZag'].iloc[i] < hist['ZigZag'].iloc[i - 1] and
                  hist['RSI'].iloc[i] > 70 and
                  hist['MACD'].iloc[i] < hist['MACD_signal'].iloc[i]):
                sell_signals.append(hist['Close'].iloc[i])
                buy_signals.append(np.nan)
            else:
                buy_signals.append(np.nan)
                sell_signals.append(np.nan)

        hist['Buy_Signal'] = buy_signals
        hist['Sell_Signal'] = sell_signals

        # --- 차트 표시 ---
        st.subheader(f"{ticker.upper()} 주가 및 매매 시그널")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(hist.index, hist['Close'], label='종가', color='black')
        ax.plot(hist.index, hist['UpperBand'], label='Bollinger 상단', linestyle='--', color='red')
        ax.plot(hist.index, hist['MiddleBand'], label='Bollinger 중앙', linestyle='--', color='green')
        ax.plot(hist.index, hist['LowerBand'], label='Bollinger 하단', linestyle='--', color='blue')

        ax.scatter(hist.index, hist['Buy_Signal'], label='매수 신호', marker='^', color='green', s=100)
        ax.scatter(hist.index, hist['Sell_Signal'], label='매도 신호', marker='v', color='red', s=100)

        ax.set_title(f"{ticker.upper()} - 기술적 매매 신호")
        ax.set_xlabel("날짜")
        ax.set_ylabel("가격")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"데이터 로딩 실패: {e}")


