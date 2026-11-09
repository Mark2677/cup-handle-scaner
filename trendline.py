"""
Daily trendline-break scanner: S&P 500 + Nasdaq-100 + Dow 30 (deduplicated).
Runs in GitHub Actions at 23:10 Israel time on US trading days.
Writes docs/trendline/index.html (Hebrew dashboard) and docs/trendline/data/<date>.json.

Descending-highs line (resistance): a line through >= 2 lower swing highs, 20-126 bars long,
  with no close above it (wicks may poke 0.25 ATR) from the first high until yesterday.
  BREAKOUT  : today's close above the line (yesterday's close was not).
Ascending-lows line (support): mirror image through higher swing lows.
  BREAKDOWN : today's close below the line (yesterday's close was not).
Watch lists : the line is still intact and today's close is within 0.5 ATR of it.
The best line per stock (most touches, then longest) is used.
Prices are split-adjusted, not dividend-adjusted (matches TradingView / Finviz).
"""
import json
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from channel import get_universe, swings, visits
from ema_cross import download
from scanner import SECTOR_HE, rsi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "trendline")

P = dict(
    min_len=20, max_len=126,   # line life: ~1 to ~6 months
    swing=3,                   # swing high/low = extreme of +-3 bars
    min_gap=5,                 # bars between the two anchors
    min_touch=2,               # separate touches of the line (anchors included)
    touch=0.3,                 # a swing point within 0.3 ATR of the line is a touch
    pierce=0.25,               # wicks may poke through the line by 0.25 ATR
    near=0.5,                  # watch list: close within 0.5 ATR of the intact line
    min_move=1.5,              # the line must fall (rise) at least 1.5 ATR over its life
)
KINDS = ("break_up", "break_down", "near_up", "near_down")


def wilder_atr(h, l, c, n=14):
    tr = np.r_[h[0] - l[0], np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))]
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def best_line(h, l, c, atr, side):
    """side='res': descending line over swing highs; side='sup': ascending line over swing lows.
    Returns the best line that is intact through yesterday's close, or None."""
    n = len(c)
    k = P["swing"]
    src = h if side == "res" else l
    piv = [i for i in swings(src, k, "high" if side == "res" else "low")
           if n - P["max_len"] <= i <= n - 1 - k]
    best = None
    for ai, p in enumerate(piv):
        if n - 1 - p < P["min_len"]:
            continue
        a = float(np.median(atr[p:n]))
        t = np.arange(p, n)
        for q in piv[ai + 1:]:
            if q - p < P["min_gap"]:
                continue
            slope = (src[q] - src[p]) / (q - p)
            if (side == "res" and slope >= 0) or (side == "sup" and slope <= 0):
                continue
            if abs(slope) * (n - 1 - p) < P["min_move"] * a:
                continue                                 # practically flat: not a trendline
            line = src[p] + slope * (t - p)
            hist = slice(p, n - 1)                       # until yesterday
            lh = line[:-1]
            if side == "res":
                if (c[hist] > lh).any() or (h[hist] > lh + P["pierce"] * a).any():
                    continue
                tp = [i for i in piv if i >= p and abs(h[i] - line[i - p]) <= P["touch"] * a]
            else:
                if (c[hist] < lh).any() or (l[hist] < lh - P["pierce"] * a).any():
                    continue
                tp = [i for i in piv if i >= p and abs(l[i] - line[i - p]) <= P["touch"] * a]
            tv = visits(np.isin(np.arange(n), tp), P["min_gap"])
            if len(tv) < P["min_touch"]:
                continue
            score = (len(tv), n - 1 - p)
            if best is None or score > best["score"]:
                best = dict(score=score, start=p, slope=float(slope), line=line, touches=tv)
    return best


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

    results, stale, lines = [], 0, dict(res=0, sup=0)
    for _, row in uni.iterrows():
        df = data.get(row["yf"])
        if df is None:
            continue
        if df.index[-1].date() != last or len(df) < P["max_len"] + 30:
            stale += 1
            continue
        c = df["Close"]
        cn, h, l = c.to_numpy(float), df["High"].to_numpy(float), df["Low"].to_numpy(float)
        atr = wilder_atr(h, l, cn)
        a_now, close = float(atr[-1]), float(cn[-1])
        for side in ("res", "sup"):
            ln = best_line(h, l, cn, atr, side)
            if not ln:
                continue
            lines[side] += 1
            lv = float(ln["line"][-1])
            d = (close - lv) / a_now                         # >0 = above the line
            if side == "res":
                kind = "break_up" if d > 0 else "near_up" if d >= -P["near"] else None
            else:
                kind = "break_down" if d < 0 else "near_down" if d <= P["near"] else None
            if not kind:
                continue
            v = df["Volume"].astype(float)
            vol50 = float(v.iloc[-51:-1].mean())
            ema = lambda n_: round(float(c.ewm(span=n_, adjust=False).mean().iloc[-1]), 2)  # noqa: E731
            sma = lambda n_: float(c.rolling(n_).mean().iloc[-1]) if len(c) >= n_ else None  # noqa: E731
            s50, s200, s150, s20 = sma(50), sma(200), sma(150), sma(20)
            n = len(cn)
            view = max(0, min(ln["start"] - 15, n - 90))
            rnd = lambda arr: [round(float(x), 2) for x in arr]  # noqa: E731
            results.append(dict(
                sym=row["sym"], name=row["name"], kind=kind, side=side, idx=row["idx"],
                sector=SECTOR_HE.get(row["sector"], row["sector"]) or "—",
                close=round(close, 2), chg=round((close / cn[-2] - 1) * 100, 2),
                line=round(lv, 2), dist_atr=round(d, 2), dist_pct=round((close / lv - 1) * 100, 2),
                slope_m=round(ln["slope"] * 21 / close * 100, 2), life=int(n - 1 - ln["start"]),
                touches=len(ln["touches"]), last_touch=int(n - 1 - ln["touches"][-1]),
                atr=round(a_now, 2), atr_pct=round(a_now / close * 100, 2),
                vol_ratio=round(float(v.iloc[-1]) / vol50, 2) if vol50 else None,
                rsi14=round(float(rsi(c, 14).iloc[-1]), 1),
                rsi35_ema10=round(float(rsi(c, 35).ewm(span=10, adjust=False).mean().iloc[-1]), 1),
                ema20=ema(20), ema30=ema(30), ema50=ema(50), ema200=ema(200),
                sma20=round(s20, 2) if s20 else None, sma150=round(s150, 2) if s150 else None,
                above50=bool(s50 and close > s50), above200=bool(s200 and close > s200),
                chart=dict(c=rnd(cn[view:]), h=rnd(h[view:]), l=rnd(l[view:]), off=int(ln["start"] - view),
                           line=rnd(ln["line"]), t=[i - view for i in ln["touches"]]),
            ))
    order = {k: i for i, k in enumerate(KINDS)}
    results.sort(key=lambda x: (order[x["kind"]], -(x["vol_ratio"] or 0) if x["kind"].startswith("break")
                                else abs(x["dist_atr"])))
    payload = dict(
        asof=asof, generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        intraday=bool(last == now_ny.date() and now_ny.hour < 16),
        scanned=len(data) - stale, universe=len(uni), stale=stale, source=source,
        lines=lines, params=P, results=results,
    )
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    with open(os.path.join(OUT, "data", f"{asof}.json"), "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    hist = []
    for fn in sorted(os.listdir(os.path.join(OUT, "data")), reverse=True)[:60]:
        if fn.endswith(".json"):
            with open(os.path.join(OUT, "data", fn)) as f:
                p = json.load(f)
            hist.append(dict(date=p["asof"], **{k: [r["sym"] for r in p["results"] if r["kind"] == k] for k in KINDS}))
    payload["history"] = hist
    with open(os.path.join(HERE, "template_trendline.html"), encoding="utf-8") as f:
        tpl = f.read()
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(tpl.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False)))
    print(f"{asof}: lines {lines}; " + ", ".join(f"{k} {sum(r['kind'] == k for r in results)}" for k in KINDS))


if __name__ == "__main__":
    run()
