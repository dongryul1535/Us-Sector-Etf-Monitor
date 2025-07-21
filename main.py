#!/usr/bin/env python3
"""
US Major Sector ETF Monitor + Composite Indicator (MACD + Slow %D)
────────────────────────────────────────────────────────────────────
📌 **무엇이 달라졌나요?**
- 기존 **MACD·Stochastic·Bollinger Bands** 신호 로직은 그대로 유지.
- 추가로 **MACD + Slow %D**(Stochastic) 합성지표를 계산해 `df["MACD_SlowD"]` 컬럼에 저장합니다.
  - ✔️ _MACD 라인_ 과 _Stochastic Slow %D(=Slow %K의 3일 평균)_ 을 단순 합산한 값입니다.
  - 필요에 따라 `scale_macd=True` 옵션을 주면 MACD 값을 **0‑100 범위로 정규화** 후 합산하여 두 지표의 스케일을 맞출 수 있습니다.
- Telegram 알림에는 여전히 매수·매도 신호만 전송하지만, `save_csv=True` 로 실행하면
  최근 90일 데이터를 **CSV** 로 저장해 합성지표를 직접 차트로 시각화할 수 있습니다.

ETF(Exchange Traded Fund)는 주식처럼 거래소에서 매매되는 펀드 상품으로, 특정 지수나 섹터를 추종하여 분산투자 및 리스크 관리에 유용합니다.
이 스크립트에서는 S&P 500 섹터 지수를 기반으로 구성된 11개 섹터 ETF를 기본으로 사용합니다.
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
# 환경변수 ETF_LIST="XLF,XLK" 형태로 지정 가능
DEFAULT_ETFS = [
    "XLB","XLE","XLF","XLI","XLK",
    "XLP","XLRE","XLU","XLV","XLY","XLC"
]
ETFS = [s.strip().upper() for s in os.getenv("ETF_LIST", ",".join(DEFAULT_ETFS)).split(",") if s.strip()]

# 실행 옵션: CSV 저장 여부·MACD 스케일링 여부를 환경변수로 제어
SAVE_CSV     = os.getenv("SAVE_CSV", "false").lower() == "true"
SCALE_MACD   = os.getenv("SCALE_MACD", "false").lower() == "true"

if not (TOKEN and CHAT_ID and ETFS):
    sys.exit("필수 환경변수 누락: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ETF_LIST")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ───── 헬퍼 함수 ───── #

def latest(series: pd.Series) -> Optional[float]:
    val = series.iloc[-1]
    return None if pd.isna(val) else float(val)


def add_indicators(df: pd.DataFrame, scale_macd: bool = False) -> pd.DataFrame:
    """MACD, Stochastic Slow %D, Bollinger, Composite(MACD+Slow D) 추가"""
    macd = MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
    st   = StochasticOscillator(df["Close"], df["High"], df["Low"], window=14, smooth_window=3)
    bb   = BollingerBands(df["Close"], window=20, window_dev=2)

    df["MACD"]      = macd.macd()
    df["MACD_SIG"]  = macd.macd_signal()
    df["SlowD"]     = st.stoch_signal()
    df["BB_UP"]     = bb.bollinger_hband()
    df["BB_LW"]     = bb.bollinger_lband()

    macd_scaled = df["MACD"]
    if scale_macd:
        # 0‒100 정규화 → Stoch와 스케일 맞춤
        min_m, max_m = macd_scaled.min(), macd_scaled.max()
        macd_scaled  = 100 * (macd_scaled - min_m) / (max_m - min_m)

    df["MACD_SlowD"] = macd_scaled + df["SlowD"]
    return df


def signal(df: pd.DataFrame) -> Tuple[Optional[str], int]:
    """기존 점수 로직(MACD·Stoch·BB 5:4:1)"""
    m = 1 if latest(df["MACD"])     > latest(df["MACD_SIG"]) else -1
    s = 1 if latest(df["SlowD"])    > 50                    else -1  # Slow D 50선 기준
    c = latest(df["Close"])
    b = 1 if c < latest(df["BB_LW"]) else -1 if c > latest(df["BB_UP"]) else 0
    score = 5*m + 4*s + b
    if score >= 7:  return "BUY", score
    if score <= -7: return "SELL", score
    return None, score


# ───── 가격 데이터 조회 ───── #

def fetch_daily(ticker: str, days: int = 90) -> Optional[pd.DataFrame]:
    end, start = dt.datetime.now(), dt.datetime.now() - dt.timedelta(days=days)
    try:
        df = fdr.DataReader(ticker, start, end)
        if df.empty or len(df) < days * 0.5:
            logging.warning(f"{ticker}: 데이터 부족({len(df)})")
            return None
        df = df.reset_index()
        df.columns = [c.capitalize() for c in df.columns]  # Date, Open, High...
        return df
    except Exception as e:
        logging.error(f"{ticker}: 조회 실패 - {e}")
        return None


# ───── Telegram 알림 ───── #

def tg(msg: str):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    for chunk in [msg[i:i+3500] for i in range(0, len(msg), 3500)]:
        requests.post(url, json={"chat_id": CHAT_ID, "text": chunk}, timeout=10)


# ───── 메인 ───── #

def main():
    alerts: List[str] = []
    for tk in ETFS:
        df = fetch_daily(tk)
        if df is None or len(df) < 40:
            alerts.append(f"{tk}: 데이터 부족")
            continue

        df = add_indicators(df, scale_macd=SCALE_MACD)
        sig, score = signal(df)
        if sig:
            alerts.append(f"{tk}: **{sig}** (score {score:+d}) – MACD_SlowD {latest(df['MACD_SlowD']):.2f}")

        if SAVE_CSV:
            csv_name = f"{tk}_history.csv"
            df.to_csv(csv_name, index=False)
            logging.info(f"{csv_name} saved.")

    tg("\n".join(alerts) if alerts else "No BUY/SELL signals detected.")


if __name__ == "__main__":
    main()
