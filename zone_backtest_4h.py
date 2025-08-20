# zone_backtest_4h.py  (4H zones; choose doji-allowed OR strict-pure-color)
import os, sys, argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import time as dtime
from typing import List
import pytz


DATA_DIR = os.path.join("data","intraday_60m")
OUT_DIR  = "out_zones_4h"; os.makedirs(OUT_DIR, exist_ok=True)
# --- Marking method ---
# "body_to_wick": (Demand) prox = max(body highs), distal = min(wicks lows)
#                 (Supply) prox = min(body lows),  distal = max(wicks highs)
# "wick_to_wick": (Demand) prox = max(wicks highs), distal = min(wicks lows)
#                 (Supply) prox = min(wicks lows),  distal = max(wicks highs)
MARKING_METHOD = "body_to_wick"   # or "wick_to_wick"


# ---- Params (tune as needed) ----
PIVOT_BODY_MAX   = 0.50   # base = candles with body% < 50% of range
LEGOUT_BODY_MIN  = 0.50   # leg-out must be "exciting"
BASE_MAX         = 3      # prefer compact bases
ATR_N            = 14
ENTRY_K_ATR      = 0.10   # entry buffer = k * ATR
SL_K_ATR         = 0.10   # SL buffer   = k * ATR
TARGET_R         = 2.0
LOOKAHEAD_4H     = 12     # forward bars to evaluate outcome
REQUIRE_CONFIRM  = True   # confirmation close beyond proximal/distal
SKIP_TOUCHED     = True   # only use each zone once

# ---- Choose base-color rule ----
# DOJI_ALLOWED = True  -> allow doji in base, forbid only opposite color
# DOJI_ALLOWED = False -> strict pure color (no doji)
DOJI_ALLOWED = True  # will be overridden by CLI if provided


# --- IST helpers & ZoneDate formatting (candle start/end) ---
IST = pytz.timezone("Asia/Kolkata")
ZONE_DATE_SHOW = "end"  # "start" (recommended) or "end"

def _ensure_ist(ts):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("Asia/Kolkata")
    else:
        ts = ts.tz_convert("Asia/Kolkata")
    return ts

def _candle_start_end(ts_end):
    # our 4h bars are label='right': timestamp is END of the 4h bar
    ts_end = _ensure_ist(ts_end)
    ts_start = ts_end - pd.Timedelta(hours=4)
    return ts_start, ts_end

def _fmt_ts_with_colon_offset(ts):
    ts = _ensure_ist(ts)
    return ts.strftime("%Y-%m-%d %H:%M:%S IST")


@dataclass
class Zone:
    date: pd.Timestamp
    ztype: str   # "Demand" / "Supply"
    proximal: float
    distal: float

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
    IST = pytz.timezone("Asia/Kolkata")

    df = pd.read_csv(path)
    dt_col = "DateTime" if "DateTime" in df.columns else ("Datetime" if "Datetime" in df.columns else None)
    if not dt_col:
        raise KeyError(f"'{path}' has no DateTime column")
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.rename(columns={dt_col: "DateTime"})

    # tz → IST
    if df["DateTime"].dt.tz is None:
        df["DateTime"] = df["DateTime"].dt.tz_localize("UTC").dt.tz_convert(IST)
    else:
        df["DateTime"] = df["DateTime"].dt.tz_convert(IST)

    # Keep only NSE session 09:15–15:30
    df = df[
        (df["DateTime"].dt.time >= pd.to_datetime("09:15").time()) &
        (df["DateTime"].dt.time <= pd.to_datetime("15:30").time())
    ].copy()

    # numeric
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["DateTime","Open","High","Low","Close"]).sort_values("DateTime")

    # --- Anchor to 09:15: shift -15m → resample 4h (label='right') → shift +15m ---
    df = df.set_index("DateTime")
    df.index = df.index - pd.Timedelta(minutes=15)
    df4 = df.resample("4h", label="right", closed="right").agg(
        {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    ).dropna()
    df4.index = df4.index + pd.Timedelta(minutes=15)

    df4 = df4.reset_index().rename(columns={"index": "DateTime"})
    # optional: reduce tiny feed diffs
    df4[["Open","High","Low","Close"]] = df4[["Open","High","Low","Close"]].round(2)
    return df4


def detect_zones_4h(df4: pd.DataFrame, doji_allowed: bool) -> List[Zone]:
    zones: List[Zone] = []

    def is_green_row(r): return float(r.Close) > float(r.Open)
    def is_red_row(r):   return float(r.Close) < float(r.Open)
    def is_dojiish(r):
        if not doji_allowed:  # strict mode -> never treat as doji
            return False
        hi, lo, op, cl = map(float, [r.High, r.Low, r.Open, r.Close])
        rng = max(hi - lo, 1e-9)
        return abs(cl - op) / rng <= 0.10  # doji if body <= 10% of range

    for i in range(5, len(df4)-2):
        if i + 1 >= len(df4): break
        legout = df4.iloc[i + 1]
        if body_pct_row(legout) < LEGOUT_BODY_MIN:
            continue

        want_green_base = is_green_row(legout)  # Demand
        want_red_base   = is_red_row(legout)    # Supply
        if not (want_green_base or want_red_base):
            continue

        ztype = "Demand" if want_green_base else "Supply"

        # Build base backward with BOTH rules:
        # 1) base body% < PIVOT_BODY_MAX
        # 2) NO opposite color inside base (doji allowed only if doji_allowed=True)
        j = i
        base_idx = []
        while j >= 0 and len(base_idx) < 10:
            r = df4.iloc[j]
            if body_pct_row(r) >= PIVOT_BODY_MAX:
                break
            if want_green_base:
                # demand base: forbid red; allow green (+doji if allowed)
                if is_red_row(r) and not is_dojiish(r):
                    break
                if not doji_allowed and not is_green_row(r):
                    break  # strict mode: only green
            else:
                # supply base: forbid green; allow red (+doji if allowed)
                if is_green_row(r) and not is_dojiish(r):
                    break
                if not doji_allowed and not is_red_row(r):
                    break  # strict mode: only red
            base_idx.append(j)
            j -= 1

        base_idx = base_idx[::-1]
        if len(base_idx) == 0 or len(base_idx) > BASE_MAX:
            continue

        base_end = base_idx[-1]

        # Mark zone (PDF body→wick OR wick→wick)
        opens  = df4.loc[base_idx,"Open"].to_numpy(float)
        closes = df4.loc[base_idx,"Close"].to_numpy(float)
        lows   = df4.loc[base_idx,"Low"].to_numpy(float)
        highs  = df4.loc[base_idx,"High"].to_numpy(float)
        body_highs = np.maximum(opens, closes)
        body_lows  = np.minimum(opens, closes)

        if MARKING_METHOD == "wick_to_wick":
            if ztype == "Demand":
                proximal = float(highs.max())   # highest wick of base
                distal   = float(lows.min())    # lowest wick of base
            else:  # Supply
                proximal = float(lows.min())    # lowest wick of base
                distal   = float(highs.max())   # highest wick of base
        else:  # body_to_wick
            if ztype == "Demand":
                proximal = float(body_highs.max())   # highest body of base
                distal   = float(lows.min())         # lowest wick of base
            else:  # Supply
                proximal = float(body_lows.min())    # lowest body of base
                distal   = float(highs.max())        # highest wick of base

        zones.append(Zone(
            date=pd.Timestamp(df4.loc[base_end, "DateTime"]),
            ztype=ztype, proximal=proximal, distal=distal
        ))

    return zones

def simulate(df4: pd.DataFrame, zones: List[Zone]):
    A = atr(df4)
    trades=[]
    touched_levels=set()

    for z in zones:
        key=(z.ztype, round(z.proximal,2), round(z.distal,2))
        if SKIP_TOUCHED and key in touched_levels: 
            continue

        # ---- tz-safe start index (avoid naive vs aware issues) ----
        # find first bar after the zone bar (>= z.date) then +1
        idx_found = df4.index[df4["DateTime"] >= z.date]
        if len(idx_found) == 0:
            continue
        start = int(idx_found[0]) + 1
        if start >= len(df4):
            continue

        atr_series = A.copy(); atr_series.index = df4.index
        outcome="none"; bars=None

        for t in range(start, len(df4)):
            row = df4.iloc[t]
            px_open, px_high, px_low, px_close = map(float,[row.Open,row.High,row.Low,row.Close])
            atr_now = float(atr_series.iloc[t])

            if z.ztype=="Demand":
                # touch?
                if px_low <= z.proximal and px_high >= z.distal:
                    if REQUIRE_CONFIRM:
                        conf_ok=False; conf_end=t
                        for k in range(t, min(t+3, len(df4))):
                            if float(df4.iloc[k].Close) > z.proximal:
                                conf_ok=True; conf_end=k; break
                        if not conf_ok: break
                        t = conf_end
                        row = df4.iloc[t]
                        px_open, px_high, px_low, px_close = map(float,[row.Open,row.High,row.Low,row.Close])
                        atr_now = float(atr_series.iloc[t])

                    entry = z.proximal + ENTRY_K_ATR*atr_now
                    sl    = z.distal   - SL_K_ATR*atr_now
                    target= entry + TARGET_R*(entry - sl)
                    for k in range(t, min(t+LOOKAHEAD_4H, len(df4))):
                        hh=float(df4.iloc[k].High); ll=float(df4.iloc[k].Low)
                        if ll <= sl: outcome="stop"; bars=k-t; break
                        if hh >= target: outcome="target"; bars=k-t; break
                    break

            else:  # Supply
                if px_high >= z.proximal and px_low <= z.distal:
                    if REQUIRE_CONFIRM:
                        conf_ok=False; conf_end=t
                        for k in range(t, min(t+3, len(df4))):
                            if float(df4.iloc[k].Close) < z.proximal:
                                conf_ok=True; conf_end=k; break
                        if not conf_ok: break
                        t = conf_end
                        row = df4.iloc[t]
                        px_open, px_high, px_low, px_close = map(float,[row.Open,row.High,row.Low,row.Close])
                        atr_now = float(atr_series.iloc[t])

                    entry = z.proximal - ENTRY_K_ATR*atr_now
                    sl    = z.distal   + SL_K_ATR*atr_now
                    target= entry - TARGET_R*(sl - entry)
                    for k in range(t, min(t+LOOKAHEAD_4H, len(df4))):
                        hh=float(df4.iloc[k].High); ll=float(df4.iloc[k].Low)
                        if hh >= sl: outcome="stop"; bars=k-t; break
                        if ll <= target: outcome="target"; bars=k-t; break
                    break

        if outcome!="none":
            touched_levels.add(key)

            # --- ZoneDate as candle START (or END) with +05:30 formatted ---
            start_ts, end_ts = _candle_start_end(z.date)
            zone_ts = start_ts if ZONE_DATE_SHOW == "start" else end_ts
            zone_time_str = _fmt_ts_with_colon_offset(zone_ts)

            end_ts = _ensure_ist(z.date)
            t = end_ts.time()
            action = "SameDay" if t == dtime(12, 15) else "NextDay"

            trades.append({
                "Symbol": df4.attrs.get("Symbol",""),
                "ZoneDate": zone_time_str,
                "ActionDay": action,         # <--- new
                "Type": z.ztype,
                "Proximal": round(z.proximal,2),
                "Distal": round(z.distal,2),
                "Outcome": outcome,
                "BarsToOutcome": bars
            })


    hit = (pd.Series([t["Outcome"] for t in trades])=="target").mean() if trades else np.nan
    return hit, pd.DataFrame(trades)

def run_symbol(csv_path, doji_allowed: bool):
    sym = os.path.basename(csv_path).replace(".csv","").replace("_",".")
    df4 = load_60m_csv(csv_path)
    df4.attrs["Symbol"] = sym
    zones = detect_zones_4h(df4, doji_allowed=doji_allowed)
    hit, tdf = simulate(df4, zones)
    suffix = "_doji" if doji_allowed else "_strict"
    if tdf is not None and not tdf.empty:
        tdf.to_csv(os.path.join(OUT_DIR, f"{sym}_4h_zone_trades{suffix}.csv"), index=False)
    return hit, len(zones)

def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--doji", action="store_true", help="Allow doji in base (forbid only opposite color) [default]")
    g.add_argument("--strict", action="store_true", help="Strict base color (no doji)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    doji_mode = DOJI_ALLOWED
    if args.doji:   doji_mode = True
    if args.strict: doji_mode = False

    rows=[]
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".csv"): continue
        hit, nz = run_symbol(os.path.join(DATA_DIR,fname), doji_allowed=doji_mode)
        rows.append({"Symbol": fname.replace(".csv","").replace("_","."), 
                     "Zones": nz, "HitRate": hit, "Mode": "doji" if doji_mode else "strict"})
    if rows:
        suffix = "_doji" if doji_mode else "_strict"
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"summary_4h{suffix}.csv"), index=False)
        print(f"Done. See {OUT_DIR}/ (mode={ 'doji' if doji_mode else 'strict' })")
    else:
        print("[warn] no symbols processed")

