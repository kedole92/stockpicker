# intraday_fetch_60m.py
import os
import pandas as pd
import yfinance as yf
import pytz

SYMBOLS = [
    "RELIANCE.NS","TCS.NS","TATASTEEL.NS","HDFCBANK.NS","INFY.NS"
]  # put your 5 or 100 here

DATA_DIR = os.path.join("data", "intraday_60m")
PERIOD   = "60d"
INTERVAL = "60m"
IST = pytz.timezone("Asia/Kolkata")
os.makedirs(DATA_DIR, exist_ok=True)

def fetch(sym: str):
    df = yf.download(
        sym, interval=INTERVAL, period=PERIOD, progress=False,
        auto_adjust=False, group_by="column", prepost=False, threads=False
    )
    if df is None or df.empty:
        print(f"[warn] no data for {sym}")
        return

    # tz → IST
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(IST)
    else:
        df = df.tz_convert(IST)

    # normalize columns
    df = df.rename(columns={
        "open":"Open","high":"High","low":"Low","close":"Close",
        "volume":"Volume","adj close":"Adj Close","Adj Close":"Adj Close"
    })

    # fallback rename if columns like "RELIANCE.NS Open"
    needed = {"Open","High","Low","Close","Volume"}
    if not needed.issubset(set(df.columns)):
        rename_map = {}
        for c in df.columns:
            parts = str(c).split()
            if len(parts)>=2 and parts[-1] in needed:
                rename_map[c] = parts[-1]
        if rename_map:
            df = df.rename(columns=rename_map)

    if not needed.issubset(set(df.columns)):
        raise KeyError(f"Missing OHLCV for {sym}: {list(df.columns)}")

    df = df.reset_index().rename(columns={"index":"DateTime","Datetime":"DateTime"})
    df = df[["DateTime","Open","High","Low","Close","Volume"]]

    out_path = os.path.join(DATA_DIR, f"{sym.replace('.','_')}.csv")
    df.to_csv(out_path, index=False)
    print(f"[ok] {sym}: {len(df)} rows -> {out_path}")

if __name__ == "__main__":
    for s in SYMBOLS:
        try:
            fetch(s)
        except Exception as e:
            print(f"[err] {s}: {e}")

