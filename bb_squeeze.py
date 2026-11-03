"""
Daily Bollinger squeeze scanner: S&P 500 + Nasdaq-100 + Dow 30 (deduplicated).
Runs in GitHub Actions at 23:10 Israel time on US trading days.
Writes docs/squeeze/index.html (Hebrew dashboard) and docs/squeeze/data/<date>.json.

A stock is "in a squeeze" when all of these hold at today's close:
  1. Bollinger(20, 2) is inside the Keltner Channel (20, 1.5 x ATR)  -> TTM-style squeeze
     and has been for at least MIN_DAYS bars in a row,
  2. Bollinger band width is in the lowest WIDTH_PCT % of the last LOOKBACK bars,
  3. the width is still contracting (today < WIDTH_SLOPE bars ago).
Each squeeze gets a bias (bullish / bearish / neutral) from momentum + trend, and
alert levels for the breakout (upper band, squeeze high) and breakdown (lower band, squeeze low).
"Fired" = the squeeze was on yesterday (>= MIN_DAYS) and released today.
Prices are split-adjusted, not dividend-adjusted (matches TradingView / Finviz).
"""
import json
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from channel import get_universe
from ema_cross import download
from scanner import SECTOR_HE, rsi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "squeeze")

P = dict(
    bb_len=20, bb_mult=2.0,       # Bollinger
    kc_len=20, kc_mult=1.5,       # Keltner (SMA20 +- 1.5 ATR20), as in TTM Squeeze
    min_days=3,                   # squeeze on for at least this many consecutive bars
    lookback=126, width_pct=20,   # band width in the lowest 20% of ~6 months
    width_slope=5,                # width today < width 5 bars ago (still contracting)
    mom_len=20,                   # TTM momentum (linear regression) length
)


def linreg_last(y):
    """value of the least-squares line at the last point (TradingView ta.linreg(y, n, 0))"""
    n = len(y)
    x = np.arange(n)
    a, b = np.polyfit(x, y, 1)
    return a * (n - 1) + b


def analyse(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    cn = c.to_numpy(float)
    mid = c.rolling(P["bb_len"]).mean()
    sd = c.rolling(P["bb_len"]).std(ddof=0)
    bu, bl = mid + P["bb_mult"] * sd, mid - P["bb_mult"] * sd
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr20 = tr.rolling(P["kc_len"]).mean()          # TTM uses a simple average of TR
    kmid = c.rolling(P["kc_len"]).mean()
    ku, kl = kmid + P["kc_mult"] * atr20, kmid - P["kc_mult"] * atr20
    on = ((bu < ku) & (bl > kl)).to_numpy()
    width = ((bu - bl) / mid * 100).to_numpy()
    # TTM momentum: linreg of close - avg(avg(HH20, LL20), SMA20)
    n = P["mom_len"]
    base = ((h.rolling(n).max() + l.rolling(n).min()) / 2 + c.rolling(n).mean()) / 2
    dev = (c - base).to_numpy(float)
    mom = np.full(len(cn), np.nan)
    for i in range(len(cn) - 30, len(cn)):
        seg = dev[i - n + 1:i + 1]
        if not np.isnan(seg).any():
            mom[i] = linreg_last(seg)
    return dict(c=cn, h=h.to_numpy(float), l=l.to_numpy(float), mid=mid.to_numpy(), bu=bu.to_numpy(),
                bl=bl.to_numpy(), ku=ku.to_numpy(), kl=kl.to_numpy(), on=on, width=width, mom=mom)


def run_len(mask, end):
    k = end
    while k >= 0 and mask[k]:
        k -= 1
    return end - k


def run():
    uni, source = get_universe()
    data = download(uni["yf"].tolist())
    if not data:
        sys.exit("No data downloaded")
    last = max(df.index[-1] for df in data.values()).date()
    now_ny = datetime.now(ZoneInfo("America/New_York"))
    if os.environ.get("FORCE") != "1" and last != now_ny.date():
        print(f"Latest bar {last} != today {now_ny.date()} (holiday / no session). Skipping.")
        return
    asof = last.strftime("%Y-%m-%d")

    results, stale, squeeze_on = [], 0, 0
    for _, row in uni.iterrows():
        df = data.get(row["yf"])
        if df is None:
            continue
        if df.index[-1].date() != last or len(df) < P["lookback"] + 40:
            stale += 1
            continue
        a = analyse(df)
        on, width, mom = a["on"], a["width"], a["mom"]
        if np.isnan(width[-1]) or np.isnan(mom[-1]):
            continue
        days_today = run_len(on, len(on) - 1)
        days_yday = run_len(on, len(on) - 2)
        if days_today:
            squeeze_on += 1
        hist = width[-P["lookback"]:]
        pct = float((hist < width[-1]).mean() * 100)          # percentile of today's width
        contracting = bool(width[-1] < width[-1 - P["width_slope"]])
        c, close = a["c"], float(a["c"][-1])
        s = pd.Series(c)
        e20, e50, e200 = (float(s.ewm(span=k, adjust=False).mean().iloc[-1]) for k in (20, 50, 200))
        mom_up = bool(mom[-1] > mom[-2])
        kind = None
        if days_today >= P["min_days"] and pct <= P["width_pct"] and contracting:
            # bias from momentum (sign + direction) and trend
            bull_pts = int(mom[-1] > 0) + int(mom_up) + int(close > e50) + int(e50 > e200)
            kind = "sq_bull" if bull_pts >= 3 else "sq_bear" if bull_pts <= 1 else "sq_neutral"
            sq_days = days_today
        elif days_today == 0 and days_yday >= P["min_days"]:
            if mom[-1] > 0 and close > a["mid"][-1]:
                kind = "fire_up"
            elif mom[-1] < 0 and close < a["mid"][-1]:
                kind = "fire_down"
            sq_days = days_yday
        if not kind:
            continue
        start = len(c) - sq_days - (0 if days_today else 1)
        sq_hi, sq_lo = float(a["h"][start:].max()), float(a["l"][start:].min())
        v = df["Volume"].astype(float)
        vol50 = float(v.iloc[-51:-1].mean())
        sma = lambda n: float(s.rolling(n).mean().iloc[-1]) if len(s) >= n else None  # noqa: E731
        s50, s200, s150, s20 = sma(50), sma(200), sma(150), sma(20)
        tr = np.maximum(a["h"][1:] - a["l"][1:], np.maximum(abs(a["h"][1:] - c[:-1]), abs(a["l"][1:] - c[:-1])))
        atr14 = float(pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
        seg = slice(-90, None)
        rnd = lambda arr: [None if np.isnan(x) else round(float(x), 2) for x in arr]  # noqa: E731
        results.append(dict(
            sym=row["sym"], name=row["name"], kind=kind, idx=row["idx"],
            sector=SECTOR_HE.get(row["sector"], row["sector"]) or "—",
            close=round(close, 2), chg=round((close / c[-2] - 1) * 100, 2),
            sq_days=int(sq_days), width=round(float(width[-1]), 2), width_pct=round(pct, 1),
            width_min=round(float(np.nanmin(hist)), 2),
            mom=round(float(mom[-1]), 3), mom_up=mom_up,
            bb_up=round(float(a["bu"][-1]), 2), bb_lo=round(float(a["bl"][-1]), 2),
            sq_hi=round(sq_hi, 2), sq_lo=round(sq_lo, 2),
            to_up=round((sq_hi / close - 1) * 100, 2), to_lo=round((sq_lo / close - 1) * 100, 2),
            atr=round(atr14, 2),
            vol_ratio=round(float(v.iloc[-1]) / vol50, 2) if vol50 else None,
            vol10=round(float(v.iloc[-10:].mean()) / vol50, 2) if vol50 else None,
            rsi14=round(float(rsi(df["Close"], 14).iloc[-1]), 1),
            rsi35_ema10=round(float(rsi(df["Close"], 35).ewm(span=10, adjust=False).mean().iloc[-1]), 1),
            ema20=round(e20, 2), ema30=round(float(s.ewm(span=30, adjust=False).mean().iloc[-1]), 2),
            ema50=round(e50, 2), ema200=round(e200, 2),
            sma20=round(s20, 2) if s20 else None, sma150=round(s150, 2) if s150 else None,
            above50=bool(s50 and close > s50), above200=bool(s200 and close > s200),
            chart=dict(c=rnd(c[seg]), h=rnd(a["h"][seg]), l=rnd(a["l"][seg]), bu=rnd(a["bu"][seg]),
                       bl=rnd(a["bl"][seg]), ku=rnd(a["ku"][seg]), kl=rnd(a["kl"][seg]),
                       on=[bool(x) for x in on[seg]]),
        ))
    order = {"fire_up": 0, "fire_down": 1}
    results.sort(key=lambda x: (order.get(x["kind"], 2), x["width_pct"], -x["sq_days"]))
    payload = dict(
        asof=asof, generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        intraday=bool(last == now_ny.date() and now_ny.hour < 16),
        scanned=len(data) - stale, universe=len(uni), stale=stale, source=source,
        squeeze_on=squeeze_on, params=P, results=results,
    )
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    with open(os.path.join(OUT, "data", f"{asof}.json"), "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    kinds = ("sq_bull", "sq_bear", "sq_neutral", "fire_up", "fire_down")
    hist = []
    for fn in sorted(os.listdir(os.path.join(OUT, "data")), reverse=True)[:60]:
        if fn.endswith(".json"):
            with open(os.path.join(OUT, "data", fn)) as f:
                p = json.load(f)
            hist.append(dict(date=p["asof"], **{k: [r["sym"] for r in p["results"] if r["kind"] == k] for k in kinds}))
    payload["history"] = hist
    with open(os.path.join(HERE, "template_squeeze.html"), encoding="utf-8") as f:
        tpl = f.read()
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(tpl.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False)))
    print(f"{asof}: squeeze on {squeeze_on}; " + ", ".join(f"{k} {sum(r['kind'] == k for r in results)}" for k in kinds))


if __name__ == "__main__":
    run()
