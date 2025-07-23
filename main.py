#!/usr/bin/env python3
"""
US Sector ETF Monitor – NH MTS-style MACD+Stochastic
────────────────────────────────────────────────────
📌 규칙(추정 구현)
- MACD_raw = EMA(12) - EMA(26)
- MACD_norm = 14기간 스토캐스틱(0~100)화 후 3기간 스무딩
- Slow%K   = 가격기반 Stoch(14,3)
- Composite K = (MACD_norm + Slow%K) / 2
- Composite D = SMA(Composite K, 3)
- Golden Cross  : CompK ↑ CompD → BUY
- Dead   Cross  : CompK ↓ CompD → SELL

모든 종목 차트를 텔레그램으로 전송하고, 마지막에 요약 텍스트 전송.
"""

import os
import sys
import logging
import datetime as dt
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import FinanceDataReader as fdr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.font_manager as fm

# ───── 한글 폰트 설정 ───── #
FONT_PATH = os.getenv("FONT_PATH", "")
if FONT_PATH and os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    _fp = fm.FontProperties(fname=FONT_PATH)
    plt.rcParams["font.family"] = _fp.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    font_prop = _fp
else:
    font_prop = None

# ───── ENV ───── #
TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DEFAULT_ETFS = ["XLB","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY","XLC"]
ETF_KR = {
    "XLB":"SPDR 소재 섹터 ETF",
    "XLE":"SPDR 에너지 섹터 ETF",
    "XLF":"SPDR 금융 섹터 ETF",
    "XLI":"SPDR 산업재 섹터 ETF",
    "XLK":"SPDR 기술 섹터 ETF",
    "XLP":"SPDR 필수소비재 섹터 ETF",
    "XLRE":"SPDR 부동산 섹터 ETF",
    "XLU":"SPDR 유틸리티 섹터 ETF",
    "XLV":"SPDR 헬스케어 섹터 ETF",
    "XLY":"SPDR 임의소비재 섹터 ETF",
    "XLC":"SPDR 커뮤니케이션 섹터 ETF"
}
ETFS     = [s.strip().upper() for s in os.getenv("ETF_LIST", ",".join(DEFAULT_ETFS)).split(",") if s.strip()]
SAVE_CSV = os.getenv("SAVE_CSV", "false").lower() == "true"

if not (TOKEN and CHAT_ID and ETFS):
    sys.exit("필수 환경변수 누락: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ETF_LIST")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ───── 유틸 ───── #
def latest(s: pd.Series, n: int = 1) -> Optional[float]:
    if len(s) < n:
        return None
    return float(s.iloc[-n])

# ───── NH 스타일 지표 계산 ───── #
def add_composites(df: pd.DataFrame,
                   fast=12, slow=26,
                   k_window=14, k_smooth=3,
                   d_smooth=3, use_ema=True, clip=True) -> pd.DataFrame:
    close, high, low = df["Close"], df["High"], df["Low"]

    # MACD raw
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_raw = ema_fast - ema_slow

    # MACD -> stochastic(0~100)
    macd_min = macd_raw.rolling(k_window, min_periods=1).min()
    macd_max = macd_raw.rolling(k_window, min_periods=1).max()
    macd_norm = (macd_raw - macd_min) / (macd_max - macd_min).replace(0, np.nan) * 100
    macd_norm = macd_norm.fillna(50)
    if k_smooth > 1:
        macd_norm = macd_norm.ewm(span=k_smooth, adjust=False).mean() if use_ema \
            else macd_norm.rolling(k_smooth, min_periods=1).mean()

    # Slow%K (가격)
    ll = low.rolling(k_window, min_periods=1).min()
    hh = high.rolling(k_window, min_periods=1).max()
    k_raw = (close - ll) / (hh - ll).replace(0, np.nan) * 100
    k_raw = k_raw.fillna(50)
    slow_k = (k_raw.ewm(span=k_smooth, adjust=False).mean() if (k_smooth > 1 and use_ema)
              else k_raw.rolling(k_smooth, min_periods=1).mean() if k_smooth > 1 else k_raw)

    comp_k = (macd_norm + slow_k) / 2.0
    comp_d = comp_k.rolling(d_smooth, min_periods=1).mean() if d_smooth > 1 else comp_k

    if clip:
        comp_k = comp_k.clip(0, 100)
        comp_d = comp_d.clip(0, 100)

    df["CompK"] = comp_k
    df["CompD"] = comp_d
    df["Diff"]  = comp_k - comp_d
    return df

# ───── 시그널 판정 ───── #
def detect_cross(df: pd.DataFrame, ob=80, os=20) -> Optional[str]:
    if len(df) < 2 or pd.isna(df["Diff"].iloc[-1]) or pd.isna(df["Diff"].iloc[-2]):
        return None
    prev, curr = df["Diff"].iloc[-2], df["Diff"].iloc[-1]
    prev_k = df["CompK"].iloc[-2]
    if prev <= 0 < curr:
        return "BUY" if prev_k < os else "BUY_W"
    if prev >= 0 > curr:
        return "SELL" if prev_k > ob else "SELL_W"
    return None

# ───── 데이터 로드 ───── #
def fetch_daily(tk: str, days: int = 180) -> Optional[pd.DataFrame]:
    end, start = dt.datetime.now(), dt.datetime.now() - dt.timedelta(days=days)
    try:
        df = fdr.DataReader(tk, start, end)
        if df.empty:
            logging.warning(f"{tk}: 데이터 없음")
            return None
        df = df.reset_index()
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        return df
    except Exception as e:
        logging.error(f"{tk}: 조회 실패 - {e}")
        return None

# ───── 차트 ───── #
def make_chart(df: pd.DataFrame, tk: str) -> str:
    fig, (ax1,
