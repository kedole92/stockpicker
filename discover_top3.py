# discover_top3.py  (STRICT base; verbose; always writes a CSV)
import os, numpy as np, pandas as pd
from dataclasses import dataclass
from datetime import time as dtime


DATA_DIR = os.path.join("data","intraday_60m")
OUT_DIR  = "out_alerts"; os.makedirs(OUT_DIR, exist_ok=True)

# ---- Controls ----
VERBOSE = True       # console debug
SAVE_EMPTY = True    # write an _EMPTY.csv if no candidates (never leave folder blank)

# ---- Params (mildly relaxed so you see picks; tune later) ----
PIVOT_BODY_MAX   = 0.50
LEGOUT_BODY_MIN  = 0.50
BASE_MAX         = 3
ATR_N            = 14
ENTRY_K_ATR      = 0.10
SL_K_ATR         = 0.10
TARGET_R         = 2.0
RECENT_ZONE_LOOKBACK_BARS = 180
NEAR_LOWER_K_ATR = 0.25   # demand lower band
NEAR_UPPER_K_ATR = 0.12   # both sides small buffer
NEAR_SUPPLY_K_ATR= 0.25
ZONE_WIDTH_MAX_ATR = 1.8  # skip zones wider than this * ATR

@dataclass
class Zone:
    date: pd.Timestamp
    ztype: str
    proximal: float
    distal: float
    base_end_idx: int

def _dbg(msg):
    if VERBOSE:
        print(">>", msg)

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
    import pytz
    IST = pytz.timezone("Asia/Kolkata")

    df = pd.read_csv(path)
    dt_col = "DateTime" if "DateTime" in df.columns else ("Datetime" if "Datetime" in df.columns else None)
    if not dt_col: raise KeyError(f"'{path}' has no DateTime column")
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.rename(columns={dt_col: "DateTime"})
    # ensure tz-aware IST
    if df["DateTime"].dt.tz is None:
        df["DateTime"] = df["DateTime"].dt.tz_localize("UTC").dt.tz_convert(IST)
    else:
        df["DateTime"] = df["DateTime"].dt.tz_convert(IST)

    # keep only market hours 09:15–15:30
    df = df[
        (df["DateTime"].dt.time >= pd.to_datetime("09:15").time()) &
        (df["DateTime"].dt.time <= pd.to_datetime("15:30").time())
    ].copy()

    # numeric
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["DateTime","Open","High","Low","Close"]).sort_values("DateTime")

    # --- 4H resample anchored to 09:15 ---
    # Trick: shift timestamps by -15 minutes so 09:15 becomes 09:00,
    # resample to 4H with offset so bins are [09:00,13:00), [13:00,17:00),
    # then shift back +15m → effectively [09:15,13:15) and [13:15,17:15).
    df = df.set_index("DateTime")
    df.index = df.index - pd.Timedelta(minutes=15)

    df4 = df.resample(
        "4h",
        label="right", closed="right",  # typical TV-like behavior
        offset="0min"                   # we already shifted explicitly
    ).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()

    # shift back to original clock
    df4.index = df4.index + pd.Timedelta(minutes=15)
    df4 = df4.reset_index().rename(columns={"DateTime":"DateTime"})

    # final numeric coercion
    for c in ["Open","High","Low","Close","Volume"]:
        df4[c] = pd.to_numeric(df4[c], errors="coerce")

    return df4


def detect_zones_strict(df4):
    zones = []
    def is_green(r): return float(r.Close) > float(r.Open)
    def is_red(r):   return float(r.Close) < float(r.Open)
    for i in range(5, len(df4)-2):
        if i+1 >= len(df4): break
        legout = df4.iloc[i+1]
        if body_pct_row(legout) < LEGOUT_BODY_MIN: continue
        want_green = is_green(legout)  # Demand
        want_red   = is_red(legout)    # Supply
        if not (want_green or want_red): continue
        # STRICT base: only green for demand; only red for supply
        j=i; base_idx=[]
        while j>=0 and len(base_idx)<10:
            r=df4.iloc[j]
            if body_pct_row(r) >= PIVOT_BODY_MAX: break
            if want_green and not is_green(r): break
            if want_red   and not is_red(r):   break
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
        zones.append(Zone(
            date=pd.Timestamp(df4.loc[base_end,"DateTime"]),
            ztype=ztype, proximal=proximal, distal=distal, base_end_idx=base_end
        ))
    return zones

def score_and_build_entries(df4, zones):
    out=[]
    A = atr(df4); A = A.bfill().ffill()
    last_idx = len(df4)-1
    last_close = float(df4.iloc[last_idx]["Close"])
    atr_now = float(A.iloc[last_idx]) if last_idx >= 0 else 0.0
    for z in zones:
        if z.base_end_idx < last_idx - RECENT_ZONE_LOOKBACK_BARS: 
            continue
        if atr_now <= 1e-9:
            continue
        zone_h = abs(z.proximal - z.distal)
        if zone_h > ZONE_WIDTH_MAX_ATR * atr_now:
            continue
        if z.ztype=="Demand":
            ok = (last_close >= (z.proximal - NEAR_LOWER_K_ATR*atr_now)
                  and last_close <= (z.proximal + NEAR_UPPER_K_ATR*atr_now)
                  and last_close > z.distal)
            if not ok: 
                continue
            entry  = z.proximal + ENTRY_K_ATR*atr_now
            stop   = z.distal   - SL_K_ATR*atr_now
            target = entry + TARGET_R*(entry - stop)
            bias   = "Buy"
            
            end_ts = pd.Timestamp(df4.iloc[last_idx]["DateTime"]).tz_convert("Asia/Kolkata")
            t = end_ts.time()
            action = "SameDay" if t == dtime(12, 15) else "NextDay"

        else:
            ok = (last_close <= (z.proximal + NEAR_SUPPLY_K_ATR*atr_now)
                  and last_close >= (z.proximal - NEAR_UPPER_K_ATR*atr_now)
                  and last_close < z.distal)
            if not ok:
                continue
            
            entry  = z.proximal - ENTRY_K_ATR*atr_now
            stop   = z.distal   + SL_K_ATR*atr_now
            target = entry - TARGET_R*(stop - entry)
            bias   = "Sell"
            
            end_ts = pd.Timestamp(df4.iloc[last_idx]["DateTime"]).tz_convert("Asia/Kolkata")
            t = end_ts.time()
            action = "SameDay" if t == dtime(12, 15) else "NextDay"
            
        dist = abs(last_close - z.proximal) / max(atr_now,1e-9)
        score = 100 - 50*dist - 10*(zone_h/atr_now)
        out.append({
            "Symbol": df4.attrs.get("Symbol",""),
            "AsOf": pd.Timestamp(df4.iloc[last_idx]["DateTime"]),
            "Type": z.ztype, "Bias": bias,
            "Entry": round(entry,2), "Target": round(target,2), "Stop": round(stop,2),
            "Proximal": round(z.proximal,2), "Distal": round(z.distal,2),
            "LastClose": round(last_close,2), "ATR": round(atr_now,2),
            "ZoneHeight": round(zone_h,2),
            "Score": round(score,2),
            "ActionDay": action,
        })
    if not out:
        cols = ["Symbol","AsOf","Type","Bias","Proximal","Distal",
                "LastClose","ATR","ZoneHeight","Score","Entry","Target","Stop"]
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(out).sort_values("Score", ascending=False)

def run_discovery_and_format():
    rows=[]; stats=[]
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".csv"): continue
        sym = fname.replace(".csv","").replace("_",".")
        path = os.path.join(DATA_DIR,fname)
        try:
            df4 = load_60m_csv(path)
        except Exception as e:
            _dbg(f"{sym}: load error: {e}")
            continue
        df4.attrs["Symbol"]=sym
        zones = detect_zones_strict(df4)
        cand  = score_and_build_entries(df4, zones)
        nz = len(zones)
        nc = 0 if not isinstance(cand, pd.DataFrame) else len(cand)
        stats.append({"Symbol": sym, "Zones": nz, "NearCandidates": nc})
        _dbg(f"{sym}: zones={nz}, near={nc}")
        if isinstance(cand, pd.DataFrame) and not cand.empty:
            rows.append(cand.head(2))

    if stats:
        _dbg(pd.DataFrame(stats).sort_values("NearCandidates", ascending=False).to_string(index=False))

    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    ts = now_ist.strftime("%Y%m%d_%H%M")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Time-aware greeting
    greet = "Good Morning!" if now_ist.hour < 12 else "Good Afternoon!"
    dateline = now_ist.strftime("(%d-%b-%Y)")

    if not rows:
        # Always write an EMPTY file so folder isn’t blank
        empty_path = os.path.join(OUT_DIR, f"top3_4h_strict_{ts}_EMPTY.csv")
        pd.DataFrame(columns=["Symbol","Bias","Entry","Target","Stop"]).to_csv(empty_path, index=False)
        msg = f"{greet}\nNo near-zone setups today. {dateline}"
        print(msg)
        print(f"[EMPTY CSV] {empty_path}")
        return pd.DataFrame(), empty_path, msg

    big = pd.concat(rows, ignore_index=True)
    top3 = big.sort_values("Score", ascending=False).head(3).reset_index(drop=True)

    out_path = os.path.join(OUT_DIR, f"top3_4h_strict_{ts}.csv")
    top3.to_csv(out_path, index=False)

    # WhatsApp-style text
    lines = [greet, f"Below are top 3 high probability stocks for today. {dateline}"]
    for i, r in top3.iterrows():
        sym = r['Symbol'].replace(".NS","")
        lines.append(f"{i+1}) {r['Bias']} {sym} {r['Entry']} {r['Target']} {r['Stop']} ({r['ActionDay']})")

    msg = "\n".join(lines)
    print(msg)
    print(f"[Top3 CSV] {out_path}")
    return top3, out_path, msg

if __name__ == "__main__":
    df, path, msg = run_discovery_and_format()
