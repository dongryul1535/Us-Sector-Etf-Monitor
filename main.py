#!/usr/bin/env python3
"""
US Sector ETF Monitor – NH MTS-style MACD+Stochastic
────────────────────────────────────────────────────
- MACD_raw = EMA(12) - EMA(26)
- MACD_norm = 14기간 Stoch(0~100) → 3기간 스무딩
- Slow%K   = 가격기반 Stoch(14,3)
- Composite K = (MACD_norm + Slow%K) / 2
- Composite D = SMA(Composite K, 3)
- Golden Cross : CompK ↑ CompD → BUY
- Dead   Cross : CompK ↓ CompD → SELL
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

# ───── Font ─────
FONT_PATH = os.getenv("FONT_PATH", "")
if FONT_PATH and os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    _fp = fm.FontProperties(fname=FONT_PATH)
    plt.rcParams["font.family"] = _fp.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    font_prop = _fp
else:
    font_prop = None

# ───── ENV ─────
TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DEFAULT_ETFS = ["XLB","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY","XLC"]
ETF_KR = {
    "XLB":"SPDR 소재 섹터 ETF","XLE":"SPDR 에너지 섹터 ETF","XLF":"SPDR 금융 섹터 ETF",
    "XLI":"SPDR 산업재 섹터 ETF","XLK":"SPDR 기술 섹터 ETF","XLP":"SPDR 필수소비재 섹터 ETF",
    "XLRE":"SPDR 부동산 섹터 ETF","XLU":"SPDR 유틸리티 섹터 ETF","XLV":"SPDR 헬스케어 섹터 ETF",
    "XLY":"SPDR 임의소비재 섹터 ETF","XLC":"SPDR 커뮤니케이션 섹터 ETF"
}
ETFS     = [s.strip().upper() for s in os.getenv("ETF_LIST", ",".join(DEFAULT_ETFS)).split(",") if s.strip()]
SAVE_CSV = os.getenv("SAVE_CSV", "false").lower() == "true"

if not (TOKEN and CHAT_ID and ETFS):
    sys.exit("필수 환경변수 누락: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ETF_LIST")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ───── Utils ─────
def latest(s: pd.Series, n: int = 1) -> Optional[float]:
    if len(s) < n:
        return None
    return float(s.iloc[-n])

# ───── Indicator (NH style) ─────
def add_composites(df: pd.DataFrame,
                   fast=12, slow=26,
                   k_window=14, k_smooth=3,
                   d_smooth=3, use_ema=True, clip=True) -> pd.DataFrame:
    close, high, low = df["Close"], df["High"], df["Low"]

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_raw = ema_fast - ema_slow

    macd_min = macd_raw.rolling(k_window, min_periods=1).min()
    macd_max = macd_raw.rolling(k_window, min_periods=1).max()
    macd_norm = (macd_raw - macd_min) / (macd_max - macd_min).replace(0, np.nan) * 100
    macd_norm = macd_norm.fillna(50)
    if k_smooth > 1:
        macd_norm = macd_norm.ewm(span=k_smooth, adjust=False).mean() if use_ema \
            else macd_norm.rolling(k_smooth, min_periods=1).mean()

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

    df["CompK"], df["CompD"], df["Diff"] = comp_k, comp_d, comp_k - comp_d
    return df

# ───── Signal ─────
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

# ───── Data load ─────
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

# ───── Chart ─────
def make_chart(df: pd.DataFrame, tk: str) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                   gridspec_kw={"height_ratios":[3,1]})

    ax1.plot(df["Date"], df["Close"], label="Close", linewidth=1.2)
    ax1.plot(df["Date"], df["Close"].rolling(20).mean(), "--", linewidth=0.8, label="MA20")
    ax1.set_title(f"{tk} ({ETF_KR.get(tk, tk)}) Price", fontproperties=font_prop)
    ax1.grid(True, linestyle=":", linewidth=0.4)
    ax1.legend(loc="upper left", prop=font_prop)

    ax2.plot(df["Date"], df["CompK"], color="red",    label="MACD+Slow%K (CompK)", linewidth=1.2)
    ax2.plot(df["Date"], df["CompD"], color="purple", label="MACD+Slow%D (CompD)", linewidth=1.2)
    ax2.axhline(20, color="gray", linestyle="--", linewidth=0.5)
    ax2.axhline(80, color="gray", linestyle="--", linewidth=0.5)
    ax2.set_ylim(0, 100)
    ax2.set_title("MACD+Stochastic (NH Style)", fontproperties=font_prop)
    ax2.grid(True, linestyle=":", linewidth=0.4)
    ax2.legend(loc="upper left", prop=font_prop)
    ax2.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

    fig.autofmt_xdate()
    fig.tight_layout()

    path = f"{tk}_comp_chart.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

# ───── Telegram ─────
def tg_text(msg: str):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    for chunk in [msg[i:i+3500] for i in range(0, len(msg), 3500)]:
        try:
            requests.post(url, json={"chat_id": CHAT_ID, "text": chunk}, timeout=15)
        except Exception as e:
            logging.warning("텍스트 전송 실패: %s", e)

def tg_photo(path: str, caption: str = ""):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    try:
        with open(path, "rb") as img:
            requests.post(url, data={"chat_id": CHAT_ID, "caption": caption},
                          files={"photo": img}, timeout=30)
    except Exception as e:
        logging.warning("사진 전송 실패: %s", e)

# ───── Main ─────
def main():
    alerts: List[str] = []
    for tk in ETFS:
        df = fetch_daily(tk)
        if df is None:
            continue

        df = add_composites(df)
        signal = detect_cross(df)

        caption = f"{tk}: CompK={latest(df['CompK']):.2f}  CompD={latest(df['CompD']):.2f}"
        if signal:
            caption = f"{tk}: **{signal}**\n" + caption
            alerts.append(caption.replace("**", ""))

        img_path = make_chart(df.tail(180), tk)
        tg_photo(img_path, caption=caption)

        if SAVE_CSV:
            df.to_csv(f"{tk}_history.csv", index=False)

    if alerts:
        tg_text("\n".join(alerts))
    else:
        tg_text("크로스 신호 없음 – No crossover detected.")

if __name__ == "__main__":
    main()
