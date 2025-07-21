#!/usr/bin/env python3
"""
US Major Sector ETF Monitor + MACD/Stoch/BB(5:4:1) Signal Bot
────────────────────────────────────────────────────────────────────
이 프로그램은 미국 주요 섹터별 상장지수펀드(ETF)를 모니터링하여 기술적 지표 신호 발생 시 Telegram으로 알림을 전송합니다.

ETF(Exchange Traded Fund)는 주식처럼 거래소에서 매매되는 펀드 상품으로, 특정 지수나 섹터를 추종하여 분산투자 및 리스크 관리에 유용합니다.
이 스크립트에서는 S&P 500 섹터 지수를 기반으로 구성된 11개 섹터 ETF를 기본으로 사용합니다.

기능:
1) FinanceDataReader 로 미국 섹터 ETF 시세 조회 (과거 90일)
2) MACD, Stochastic, Bollinger Bands 계산 (가중치 5:4:1)
3) 매수/매도 신호 판단 → Telegram HTTP API 알림
"""
import os
import sys
import logging
import datetime as dt
from typing import List, Optional, Tuple

import pandas as pd
import requests
import FinanceDataReader as fdr
from ta.trend import MACD
from ta.momentum import StochasticOscillator
from ta.volatility import BollingerBands

# ───── 0. 환경 변수 ───── #
TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
# 기본 미국 섹터 ETF 리스트: Materials, Energy, Financials, Industrials,
# Technology, Consumer Staples, Real Estate, Utilities, Health Care, Consumer Discretionary, Communication
DEFAULT_ETFS = [
    "XLB","XLE","XLF","XLI","XLK",
    "XLP","XLRE","XLU","XLV","XLY","XLC"
]
ETF_LIST = os.getenv("ETF_LIST")
ETFS = [s.strip().upper() for s in (ETF_LIST.split(",") if ETF_LIST else DEFAULT_ETFS) if s.strip()]

if not (TOKEN and CHAT_ID and ETFS):
    sys.exit("필수 환경변수 누락: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID 또는 ETF_LIST")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ───── 지표 계산 헬퍼 ───── #
def latest(series: pd.Series) -> Optional[float]:
    val = series.iloc[-1]
    return None if pd.isna(val) else float(val)

def signal(df: pd.DataFrame) -> Tuple[Optional[str], int]:
    macd = MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
    st   = StochasticOscillator(df["Close"], df["High"], df["Low"], window=14, smooth_window=3)
    bb   = BollingerBands(df["Close"], window=20, window_dev=2)

    m = 1 if latest(macd.macd())        > latest(macd.macd_signal())    else -1
    s = 1 if latest(st.stoch())         > latest(st.stoch_signal())     else -1
    c = latest(df["Close"])
    b = 1 if c < latest(bb.bollinger_lband()) else -1 if c > latest(bb.bollinger_hband()) else 0

    score = 5*m + 4*s + b
    if score >= 7:  return "BUY", score
    if score <= -7: return "SELL", score
    return None, score

# ───── 가격 데이터 조회 ───── #
def fetch_daily(ticker: str, days: int = 90) -> Optional[pd.DataFrame]:
    end   = dt.datetime.now()
    start = end - dt.timedelta(days=days)
    try:
        df = fdr.DataReader(ticker, start, end)
        if df.empty or len(df) < days * 0.5:
            logging.warning(f"{ticker}: 데이터 부족 ({len(df)} rows)")
            return None
        df = df.reset_index()
        return df.rename(columns={
            "Date":"Date", "Open":"Open", "High":"High",
            "Low":"Low", "Close":"Close", "Volume":"Volume"
        })
    except Exception as e:
        logging.error(f"{ticker}: 데이터 조회 실패 - {e}")
        return None

# ───── Telegram 알림 ───── #
def tg(message: str):
    api = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    for part in [message[i:i+3500] for i in range(0, len(message), 3500)]:
        requests.post(api, json={"chat_id": CHAT_ID, "text": part}, timeout=10)

# ───── 메인 ───── #
def main():
    alerts: List[str] = []
    for ticker in ETFS:
        df = fetch_daily(ticker)
        if df is None or len(df) < 40:
            alerts.append(f"{ticker}: 데이터 없음 또는 부족")
            continue

        sig, score = signal(df)
        if sig:
            alerts.append(f"{ticker}: **{sig}** (score {score:+d})")

    if not alerts:
        alerts = ["No BUY/SELL signals detected."]

    tg("\n".join(alerts))

if __name__ == "__main__":
    main()
