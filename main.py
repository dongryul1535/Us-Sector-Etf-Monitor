#!/usr/bin/env python3
"""
US Sector ETF Monitor – Golden / Dead Cross on Composite Lines
────────────────────────────────────────────────────────────────────
📌 **매수·매도 규칙**
- **Composite K** = MACD(12,26) + Slow %K(14,3)
- **Composite D** = MACD Signal + Slow %D(14,3)
- **Golden Cross** (Composite K ↑ Composite D)  → **BUY**
- **Dead Cross**   (Composite K ↓ Composite D)  → **SELL**

매 실행 시 마지막 두 일자의 교차여부를 판정해 신호가 발생하면 Telegram으로 **텍스트 + 차트 이미지**를 전송합니다.

환경 변수
-----------
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (필수)
- `ETF_LIST="XLF,XLK"` 모니터링 대상 (없으면 11개 섹터 ETF 기본)
- `SCALE_MACD=true` → 0‑100 정규화 후 합산 (Stoch와 스케일 맞춤)
- `SAVE_CSV=true`   → CSV 저장

requirements.txt (추가 패키지 포함)
-----------------------------------
```
pandas>=1.5.3
requests>=2.28.2
finance-datareader>=0.9.59
ta>=0.10.2
matplotlib>=3.8.4
```
"""
import os
import sys
import logging
import datetime as dt
from typing import List, Optional

import pandas as pd
import requests
import FinanceDataReader as fdr
from ta.trend import MACD
from ta.momentum import StochasticOscillator

import matplotlib
matplotlib.use("Agg")  # 서버·CI 환경용
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.font_manager as fm

# 한글 폰트 설정 (환경변수 FONT_PATH로 .ttf 경로 지정)
FONT_PATH = os.getenv("FONT_PATH", "")
if FONT_PATH and os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    font_prop = fm.FontProperties(fname=FONT_PATH)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False
else:
    font_prop = None

# ───── 환경 변수 ───── #
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
ETFS = [s.strip().upper() for s in os.getenv("ETF_LIST", ",".join(DEFAULT_ETFS)).split(",") if s.strip()]
SCALE_MACD = os.getenv("SCALE_MACD", "false").lower() == "true"
SAVE_CSV   = os.getenv("SAVE_CSV",   "false").lower() == "true"

if not (TOKEN and CHAT_ID and ETFS):
    sys.exit("필수 환경변수 누락: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ETF_LIST")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ───── 유틸 함수 ───── #

def latest(s: pd.Series, n: int = 1) -> Optional[float]:
    """n=1 → 마지막 값, n=2 → 마지막‑1 값"""
    if len(s) < n:
        return None
    return float(s.iloc[-n])

# ───── 지표 계산 ───── #

def add_composites(df: pd.DataFrame) -> pd.DataFrame:
    """MACDㆍStoch 계산 후 Composite K, D 컬럼 추가 (Composite D = MACD Signal + Slow %D)"""
    macd = MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
    stoch = StochasticOscillator(df["Close"], df["High"], df["Low"], window=14, smooth_window=3)

    df["MACD"]     = macd.macd()
    df["MACD_SIG"] = macd.macd_signal()
    df["SlowK"]    = stoch.stoch()
    df["SlowD"]    = stoch.stoch_signal()

    macd_line = df["MACD"]
    macd_sig  = df["MACD_SIG"]
    if SCALE_MACD:
        min_m, max_m = macd_line.min(), macd_line.max()
        macd_line = 100 * (macd_line - min_m) / (max_m - min_m)
        macd_sig  = 100 * (macd_sig  - min_m) / (max_m - min_m)

    df["CompK"] = macd_line + df["SlowK"]
    df["CompD"] = macd_sig  + df["SlowD"]
    df["Diff"]  = df["CompK"] - df["CompD"]
    return df

# ───── 시그널 판정 ───── #

def detect_cross(df: pd.DataFrame) -> Optional[str]:
    """골든·데드 크로스 판별"""
    if len(df) < 2 or pd.isna(df["Diff"].iloc[-1]) or pd.isna(df["Diff"].iloc[-2]):
        return None
    prev, curr = df["Diff"].iloc[-2], df["Diff"].iloc[-1]
    if prev <= 0 and curr > 0:
        return "BUY"
    if prev >= 0 and curr < 0:
        return "SELL"
    return None

# ───── 데이터 로드 ───── #

def fetch_daily(tk: str, days: int = 120) -> Optional[pd.DataFrame]:
    end, start = dt.datetime.now(), dt.datetime.now() - dt.timedelta(days=days)
    try:
        df = fdr.DataReader(tk, start, end)
        if df.empty:
            logging.warning(f"{tk}: 데이터 없음")
            return None
        df = df.reset_index()
        first_col = df.columns[0]
        if first_col.lower() in ('index', ''):
            df.rename(columns={first_col: 'Date'}, inplace=True)
        if 'date' in df.columns and 'Date' not in df.columns:
            df.rename(columns={'date': 'Date'}, inplace=True)
        return df
    except Exception as e:
        logging.error(f"{tk}: 조회 실패 - {e}")
        return None

# ───── 차트 ───── #

def make_chart(df: pd.DataFrame, tk: str) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={'height_ratios':[3,1]})
    ax1.plot(df["Date"], df["Close"], label="Close", linewidth=1.2)
    ax1.plot(df["Date"], df["Close"].rolling(20).mean(), linestyle="--", linewidth=0.8, label="MA20")
    ax1.set_title(f"{tk} ({ETF_KR.get(tk, tk)}) Price", fontproperties=font_prop)
    ax1.grid(True, linestyle=":", linewidth=0.4)
    ax1.legend(loc="upper left", prop=font_prop)

    ax2.plot(df["Date"], df["CompK"], label="Composite K", linewidth=1.2)
    ax2.plot(df["Date"], df["CompD"], label="Composite D", linewidth=1.2)
    ax2.axhline(0, linewidth=0.5)
    ax2.set_title("Composite Lines (MACD+Slow%K / D)", fontproperties=font_prop)
    ax2.grid(True, linestyle=":", linewidth=0.4)
    ax2.legend(loc="upper left", prop=font_prop)

    ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    fig.tight_layout()

    path = f"{tk}_comp_chart.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

# ───── Telegram ───── #

def tg_text(msg: str):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    for chunk in [msg[i:i+3500] for i in range(0, len(msg), 3500)]:
        requests.post(url, json={"chat_id": CHAT_ID, "text": chunk}, timeout=10)

def tg_photo(path: str, caption: str = ""):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    with open(path, "rb") as img:
        requests.post(url, data={"chat_id": CHAT_ID, "caption": caption}, files={"photo": img}, timeout=20)

# ───── 메인 ───── #

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

        img_path = make_chart(df.tail(120), tk)
        tg_photo(img_path, caption=caption)

        if SAVE_CSV:
            df.to_csv(f"{tk}_history.csv", index=False)

    if alerts:
        tg_text("\n".join(alerts))
    else:
        tg_text("크로스 신호 없음 – No crossover detected.")

if __name__ == "__main__":
    main()
