# discover_top3.py  (STRICT base; 4H; outputs entry/target/stop)
import os, numpy as np, pandas as pd
from dataclasses import dataclass

DATA_DIR = os.path.join("data","intraday_60m")
OUT_DIR  = "out_alerts"; os.makedirs(OUT_DIR, exist_ok=True)

# --- params ---
PIVOT_BODY_MAX   = 0.50
LEGOUT_BODY_MIN  = 0.50
BASE_MAX         = 3
ATR_N            = 14
ENTRY_K_ATR      = 0.10
SL_K_ATR         = 0.10
TARGET_R         = 2.0
RECENT_ZONE_LOOKBACK_BARS = 120
NEAR_LOWER_K_ATR = 0.20
NEAR_UPPER_K_ATR = 0.10
NEAR_SUPPLY_K_ATR= 0.20

@dataclass
class Zone:
    date: pd.Timestamp
    ztype: str
    proximal: float
    distal: float
    base_end_idx: int

def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def atr(df, n=ATR_N):
    h = pd.to_numeric(df["High"], errors="coerce")
    l = pd.to_numeric(df["Low"], errors="coerce")
    c = pd.to_numeric(df["Close"], errors="coerce")
    prev = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    return ema(tr, n)

def body_pct_row(r):
    hi, lo = float(r.High), float(r.Low)
    op, cl = float(r.Open), float(r.Close)
    rng = max(hi-lo, 1e-9)
    return abs(cl-op)/rng

def _coerce_numeric(df):
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_60m_csv(path):
    df = pd.read_csv(path)
    dt_col = "DateTime" if "DateTime" in df.columns else ("Datetime" if "Datetime" in df.columns else None)
    if not dt_col: raise KeyError(f"'{path}' has no DateTime column")
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.rename(columns={dt_col: "DateTime"})
    df = _coerce_numeric(df)
    df = df.sort_values("DateTime").dropna(subset=["DateTime","Open","High","Low","Close"]).reset_index(drop=True)
    df4 = df.set_index("DateTime").resample("4h").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna().reset_index()
    return _coerce_numeric(df4)

def detect_zones_strict(df4):
    zones = []
    def is_green(r): return float(r.Close) > float(r.Open)
    def is_red(r):   return float(r.Close) < float(r.Open)
    for i in range(5, len(df4)-2):
        if i+1 >= len(df4): break
        legout = df4.iloc[i+1]
        if body_pct_row(legout) < LEGOUT_BODY_MIN: continue
        want_green = is_green(legout); want_red = is_red(legout)
        if not (want_green or want_red): continue
        j=i; base_idx=[]
        while j>=0 and len(base_idx)<10:
            r=df4.iloc[j]
            if body_pct_row(r) >= PIVOT_BODY_MAX: break
            if want_green and not is_green(r): break   # STRICT: only green
            if want_red   and not is_red(r):   break   # STRICT: only red
            base_idx.append(j); j-=1
        base_idx=base_idx[::-1]
        if len(base_idx)==0 or len(base_idx)>BASE_MAX: continue
        base_end=base_idx[-1]
        opens  = df4.loc[base_idx,"Open"].to_numpy(float)
        closes = df4.loc[base_idx,"Close"].to_numpy(float)
        lows   = df4.loc[base_idx,"Low"].to_numpy(float)
        highs  = df4.loc[base_idx,"High"].to_numpy(float)
        body_highs = np.maximum(opens,closes)
        body_lows  = np.minimum(opens,closes)
        if want_green:
            ztype="Demand"; proximal=float(body_highs.max()); distal=float(lows.min())
        else:
            ztype="Supply"; proximal=float(body_lows.min());  distal=float(highs.max())
        zones.append(Zone(date=pd.Timestamp(df4.loc[base_end,"DateTime"]),
                          ztype=ztype, proximal=proximal, distal=distal, base_end_idx=base_end))
    return zones

def score_and_build_entries(df4, zones):
    out=[]
    A = atr(df4).fillna(method="bfill").fillna(method="ffill")
    last_idx = len(df4)-1
    last_close = float(df4.iloc[last_idx]["Close"])
    atr_now = float(A.iloc[last_idx]) if last_idx >= 0 else 0.0

    for z in zones:
        if z.base_end_idx < last_idx - RECENT_ZONE_LOOKBACK_BARS: continue
        if atr_now <= 1e-9: continue
        zone_h = abs(z.proximal - z.distal)
        if zone_h > 1.5*atr_now: continue  # quality filter

        if z.ztype=="Demand":
            ok = (last_close >= (z.proximal - NEAR_LOWER_K_ATR*atr_now)
                  and last_close <= (z.proximal + NEAR_UPPER_K_ATR*atr_now)
                  and last_close > z.distal)
            if not ok: continue
            entry  = z.proximal + ENTRY_K_ATR*atr_now
            stop   = z.distal   - SL_K_ATR*atr_now
            target = entry + TARGET_R*(entry - stop)
            bias   = "Buy"
        else:
            ok = (last_close <= (z.proximal + NEAR_SUPPLY_K_ATR*atr_now)
                  and last_close >= (z.proximal - NEAR_UPPER_K_ATR*atr_now)
                  and last_close < z.distal)
            if not ok: continue
            entry  = z.proximal - ENTRY_K_ATR*atr_now
            stop   = z.distal   + SL_K_ATR*atr_now
            target = entry - TARGET_R*(stop - entry)
            bias   = "Sell"

        dist = abs(last_close - z.proximal) / max(atr_now,1e-9)
        score = 100 - 50*dist - 10*(zone_h/atr_now)
        out.append({
            "Symbol": df4.attrs.get("Symbol",""),
            "AsOf": pd.Timestamp(df4.iloc[last_idx]["DateTime"]),
            "Type": z.ztype, "Bias": bias,
            "Entry": round(entry,2), "Target": round(target,2), "Stop": round(stop,2),
            "Proximal": round(z.proximal,2), "Distal": round(z.distal,2),
            "LastClose": round(last_close,2), "ATR": round(atr_now,2),
            "Score": round(score,2)
        })

    if not out: return pd.DataFrame()
    return pd.DataFrame(out).sort_values("Score", ascending=False)

def run_discovery_and_format():
    rows=[]
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".csv"): continue
        sym = fname.replace(".csv","").replace("_",".")
        df4 = load_60m_csv(os.path.join(DATA_DIR,fname))
        df4.attrs["Symbol"]=sym
        zones = detect_zones_strict(df4)
        cand = score_and_build_entries(df4, zones)
        if not cand.empty:
            rows.append(cand.head(2))  # top few per symbol
    if not rows:
        return pd.DataFrame(), ""

    big = pd.concat(rows, ignore_index=True)
    top3 = big.sort_values("Score", ascending=False).head(3).reset_index(drop=True)
    ts = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y%m%d_%H%M")
    out_path = os.path.join(OUT_DIR, f"top3_4h_strict_{ts}.csv")
    top3.to_csv(out_path, index=False)

    # WhatsApp text format
    lines = ["Good Morning!", "Below are top 3 high probability stocks for today."]
    for i, r in top3.iterrows():
        lines.append(f"{i+1}) {r['Bias']} {r['Symbol']} {r['Entry']} {r['Target']} {r['Stop']}")
    msg = "\n".join(lines)
    print(msg)
    return top3, out_path, msg

if __name__ == "__main__":
    df, path, msg = run_discovery_and_format()
    if isinstance(df, pd.DataFrame) and not df.empty:
        print(f"[Top3 CSV] {path}")
    else:
        print("[info] No near-zone candidates right now.")

