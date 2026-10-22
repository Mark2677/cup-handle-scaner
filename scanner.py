"""
Daily S&P 500 cup-and-handle scanner.
Runs in GitHub Actions after the US close, writes docs/index.html (Hebrew dashboard)
and docs/data/<date>.json.

Detection follows William O'Neil's classic rules (approximate, algorithmic):
  * prior uptrend  >= 30% into the left rim
  * cup length 7-65 weeks (35-325 trading days), depth 12-35%, U-shaped (not V)
  * right rim recovers to within 10% of the left rim
  * handle 1-6 weeks (5-30 trading days), depth <= 15%, in the upper half of the cup
  * pivot (buy point) = handle high + 0.10
"""
import io
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")

# ---------------------------------------------------------------- parameters
P = dict(
    prior_uptrend=0.30,      # min rise into the left rim
    prior_lookback=150,      # days before the left rim to measure the uptrend
    cup_min=35, cup_max=325, # cup length in trading days (7-65 weeks)
    depth_min=0.12, depth_max=0.35,
    rim_tol_low=0.90,        # right rim >= 90% of left rim
    rim_tol_high=1.05,       # right rim <= 105% of left rim
    handle_min=5, handle_max=30,
    handle_depth_max=0.15,
    u_bottom_frac=0.38,      # min share of cup days in the bottom third (U ~0.5, straight V ~0.33)
    near_pivot=0.05,         # "near pivot" = within 5% below pivot
    breakout_vol=1.4,        # breakout volume >= 1.4x 50-day average
    min_price=10.0,
)

SECTOR_HE = {
    "Information Technology": "טכנולוגיה", "Health Care": "בריאות", "Financials": "פיננסים",
    "Consumer Discretionary": "צריכה מחזורית", "Consumer Staples": "צריכה בסיסית",
    "Industrials": "תעשייה", "Energy": "אנרגיה", "Materials": "חומרי גלם",
    "Utilities": "תשתיות", "Real Estate": 'נדל"ן', "Communication Services": "תקשורת",
}


# ---------------------------------------------------------------- universe
def get_universe():
    """Current S&P 500 list from Wikipedia; falls back to the bundled CSV."""
    try:
        import requests
        html = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (cup-handle-scanner)"}, timeout=30,
        ).text
        df = pd.read_html(io.StringIO(html), attrs={"id": "constituents"})[0]
        source = "Wikipedia"
    except Exception as e:  # noqa: BLE001
        print("Wikipedia failed, using fallback:", e)
        df = pd.read_csv(os.path.join(HERE, "sp500_fallback.csv"))
        source = "fallback"
    df = df.rename(columns={"Symbol": "sym", "Security": "name", "GICS Sector": "sector"})
    df["yf"] = df["sym"].str.replace(".", "-", regex=False)
    print(f"Universe: {len(df)} tickers ({source})")
    return df[["sym", "yf", "name", "sector"]], source


# ---------------------------------------------------------------- data
def download(tickers):
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        for attempt in range(3):
            try:
                d = yf.download(chunk, period="2y", interval="1d", auto_adjust=True,
                                group_by="ticker", progress=False, threads=True)
                break
            except Exception as e:  # noqa: BLE001
                print("retry", attempt, e)
                time.sleep(10)
        else:
            continue
        for t in chunk:
            try:
                df = d[t].dropna()
                if len(df) > 200:
                    out[t] = df
            except KeyError:
                pass
        time.sleep(2)
    print(f"Downloaded {len(out)}/{len(tickers)}")
    return out


# ---------------------------------------------------------------- indicators
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


# ---------------------------------------------------------------- detection
def find_cup_handle(df):
    """Return the best cup-with-handle ending today, or None."""
    h = df["High"].to_numpy()
    l = df["Low"].to_numpy()
    c = df["Close"].to_numpy()
    v = df["Volume"].to_numpy().astype(float)
    n = len(c)
    if c[-1] < P["min_price"]:
        return None

    best = None
    # right rim r: the handle runs from r to today
    for hl in range(P["handle_min"], P["handle_max"] + 1):
        r = n - 1 - hl
        if r < P["cup_min"] + 20:
            break
        handle_h = h[r:]
        # r must be the handle's high (excluding a possible breakout today)
        if h[r] < handle_h[:-1].max():
            continue
        rim_r = h[r]
        handle_low = l[r + 1:].min()
        handle_depth = (rim_r - handle_low) / rim_r
        if handle_depth > P["handle_depth_max"] or handle_depth <= 0.005:
            continue

        lo_L = max(0, r - P["cup_max"])
        hi_L = r - P["cup_min"]
        if hi_L <= lo_L:
            continue
        # candidate left rims: local highs (highest within +-10 days)
        for L in range(hi_L, lo_L - 1, -1):
            win = h[max(0, L - 10):L + 11]
            if h[L] < win.max():
                continue
            rim_l = h[L]
            if not (P["rim_tol_low"] * rim_l <= rim_r <= P["rim_tol_high"] * rim_l):
                continue
            seg_h = h[L + 1:r]
            if seg_h.max() > max(rim_l, rim_r) * 1.01:   # nothing poking above the rims
                continue
            seg_l = l[L:r + 1]
            b = L + int(seg_l.argmin())
            bottom = l[b]
            top = min(rim_l, rim_r)
            depth = (rim_l - bottom) / rim_l
            if not (P["depth_min"] <= depth <= P["depth_max"]):
                continue
            cup_len = r - L
            pos = (b - L) / cup_len
            if not (0.2 <= pos <= 0.8):                    # bottom roughly centred
                continue
            third = bottom + (top - bottom) / 3
            u_frac = float((c[L:r + 1] <= third).mean())
            if u_frac < P["u_bottom_frac"]:                # too V-shaped
                continue
            mid = bottom + (rim_r - bottom) / 2
            if handle_low < mid:                           # handle in upper half
                continue
            pl = max(0, L - P["prior_lookback"])
            if L - pl < 40:
                continue
            prior_low = l[pl:L].min()
            prior = rim_l / prior_low - 1
            if prior < P["prior_uptrend"]:
                continue

            vol50 = v[-51:-1].mean()
            handle_vol = v[r + 1:-1].mean() if hl > 1 else v[-1]
            cup_vol = v[L:r + 1].mean()
            pivot = float(handle_h[:-1].max()) + 0.10
            # quality score (0-100)
            score = 50
            score += 10 if 0.15 <= depth <= 0.30 else 0
            score += 10 if handle_depth <= 0.12 else 0
            score += 10 if handle_vol < cup_vol else 0
            score += max(0, min(10, (u_frac - 0.38) * 70))
            score += 10 if rim_r >= 0.97 * rim_l else 0
            cand = dict(
                L=L, b=b, r=r, rim_l=rim_l, rim_r=rim_r, bottom=bottom,
                depth=depth, cup_weeks=cup_len / 5, handle_days=hl,
                handle_depth=handle_depth, prior=prior, u_frac=u_frac,
                pivot=pivot, vol_ratio=v[-1] / vol50 if vol50 else 0,
                handle_vol_dry=handle_vol < cup_vol, score=round(score),
            )
            if best is None or cand["score"] > best["score"]:
                best = cand
            break  # nearest valid left rim for this handle is enough
    return best


def classify(m, close):
    dist = close / m["pivot"] - 1
    m["dist"] = dist
    if dist > 0 and m["vol_ratio"] >= P["breakout_vol"]:
        m["status"] = "breakout"
    elif dist > 0:
        m["status"] = "above"
    elif dist >= -P["near_pivot"]:
        m["status"] = "near"
    else:
        m["status"] = "forming"
    return m


# ---------------------------------------------------------------- main
def run():
    uni, source = get_universe()
    data = download(uni["yf"].tolist())
    asof = max(df.index[-1] for df in data.values()).strftime("%Y-%m-%d")
    results = []
    for _, row in uni.iterrows():
        df = data.get(row["yf"])
        if df is None:
            continue
        try:
            m = find_cup_handle(df)
        except Exception as e:  # noqa: BLE001
            print(row["sym"], e)
            continue
        if not m:
            continue
        close = df["Close"]
        m = classify(m, float(close.iloc[-1]))
        start = m["L"] - 20 if m["L"] >= 20 else 0
        seg = df.iloc[start:]
        results.append(dict(
            sym=row["sym"], name=row["name"],
            sector=SECTOR_HE.get(row["sector"], row["sector"]),
            close=round(float(close.iloc[-1]), 2),
            chg=round(float(close.iloc[-1] / close.iloc[-2] - 1) * 100, 2),
            pivot=round(m["pivot"], 2), dist=round(m["dist"] * 100, 2),
            depth=round(m["depth"] * 100, 1), cup_weeks=round(m["cup_weeks"], 1),
            handle_days=m["handle_days"], handle_depth=round(m["handle_depth"] * 100, 1),
            prior=round(m["prior"] * 100, 0), vol_ratio=round(m["vol_ratio"], 2),
            dry=bool(m["handle_vol_dry"]), score=m["score"], status=m["status"],
            rsi14=round(float(rsi(close).iloc[-1]), 1),
            above50=bool(close.iloc[-1] > close.rolling(50).mean().iloc[-1]),
            above200=bool(close.iloc[-1] > close.rolling(200).mean().iloc[-1]),
            ema20=round(float(ema(close, 20).iloc[-1]), 2),
            ema50=round(float(ema(close, 50).iloc[-1]), 2),
            ema200=round(float(ema(close, 200).iloc[-1]), 2),
            chart=dict(
                d=[x.strftime("%Y-%m-%d") for x in seg.index],
                c=[round(float(x), 2) for x in seg["Close"]],
                iL=m["L"] - start, ib=m["b"] - start, ir=m["r"] - start,
            ),
        ))
    order = {"breakout": 0, "above": 1, "near": 2, "forming": 3}
    results.sort(key=lambda x: (order[x["status"]], -x["score"]))
    payload = dict(
        asof=asof, generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        scanned=len(data), universe=len(uni), source=source, params=P, results=results,
    )
    os.makedirs(os.path.join(DOCS, "data"), exist_ok=True)
    with open(os.path.join(DOCS, "data", f"{asof}.json"), "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    # history index (dates + counts) for the history tab
    hist = []
    for fn in sorted(os.listdir(os.path.join(DOCS, "data")), reverse=True):
        if fn.endswith(".json") and fn[0].isdigit():
            try:
                with open(os.path.join(DOCS, "data", fn)) as f:
                    p = json.load(f)
                hist.append(dict(date=p["asof"],
                                 total=len(p["results"]),
                                 breakout=[r["sym"] for r in p["results"] if r["status"] == "breakout"],
                                 near=[r["sym"] for r in p["results"] if r["status"] == "near"]))
            except Exception:  # noqa: BLE001
                pass
    payload["history"] = hist[:60]
    with open(os.path.join(HERE, "template.html"), encoding="utf-8") as f:
        tpl = f.read()
    html = tpl.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False))
    with open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"{asof}: {len(results)} patterns "
          f"({sum(r['status'] == 'breakout' for r in results)} breakouts)")


if __name__ == "__main__":
    run()
