# intraday_fetch_60m.py  (robust, single-level columns)
import os
import pandas as pd
import yfinance as yf
import pytz

# --- CONFIG ---
# Load symbols from file
with open(r"D:\\AI\\TIZ\\data\\nifty100.txt") as f:
    SYMBOLS = [line.strip() for line in f if line.strip()]

print(f"[info] Loaded {len(SYMBOLS)} symbols from nifty100.txt")

DATA_DIR = os.path.join("data", "intraday_60m")
PERIOD   = "60d"
INTERVAL = "60m"
IST = pytz.timezone("Asia/Kolkata")

os.makedirs(DATA_DIR, exist_ok=True)

def fetch(sym: str):
    # Force single-level columns
    df = yf.download(
        sym,
        interval=INTERVAL,
        period=PERIOD,
        progress=False,
        auto_adjust=False,
        group_by="column",
        prepost=False,
        threads=False,
    )

    if df is None or df.empty:
        print(f"[warn] no data for {sym}")
        return

    # Make index tz-aware → IST
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(IST)
    else:
        df = df.tz_convert(IST)

    # Normalize columns (sometimes case/spacing varies)
    df = df.rename(columns={
        "open":"Open","high":"High","low":"Low","close":"Close",
        "volume":"Volume","adj close":"Adj Close","Adj Close":"Adj Close"
    })

    # Keep required cols if present
    needed = ["Open","High","Low","Close","Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        # Some builds put columns as e.g. 'RELIANCE.NS Open' etc. Handle that:
        # Try to split by space and take last token.
        rename_map = {}
        for c in df.columns:
            parts = str(c).split()
            if len(parts) >= 2 and parts[-1] in ["Open","High","Low","Close","Volume"]:
                rename_map[c] = parts[-1]
        if rename_map:
            df = df.rename(columns=rename_map)
            missing = [c for c in needed if c not in df.columns]

    if missing:
        raise KeyError(f"Missing expected columns after fetch: {missing} (got: {list(df.columns)})")

    # Move index to a column named DateTime (IST)
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
