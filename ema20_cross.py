"""
Daily S&P 500 scanner: EMA(20) crosses SMA(20) + RSI(10) filter.
Runs in GitHub Actions at 23:10 Israel time on US trading days.
Writes docs/cross20/index.html (Hebrew dashboard) and docs/cross20/data/<date>.json.

LONG ONLY.
Cross up : EMA20 crossed above SMA20 at today's close.
Above    : EMA20 is above SMA20 (crossed earlier).
Filter   : RSI(10) at today's close between RSI_LO and RSI_HI (inclusive).
Prices are split-adjusted but NOT dividend-adjusted, to match TradingView / Finviz charts.
"""
import json
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ema_cross import download
from scanner import SECTOR_HE, get_universe, rsi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "cross20")

FAST, SLOW = 20, 20          # EMA length, SMA length
RSI_LEN, RSI_LO, RSI_HI = 10, 40.0, 60.0
RSI_SLOPE = 3    # shown on the cards only: RSI10 change vs 3 bars ago
RANGE_DAYS = 60  # shown on the cards only: position in the 60-day range


def run():
    uni, source = get_universe()
    data = download(uni["yf"].tolist())
    if not data:
        sys.exit("No data downloaded")
    last = max(df.index[-1] for df in data.values()).date()
    today_ny = datetime.now(ZoneInfo("America/New_York")).date()
    if os.environ.get("FORCE") != "1" and last != today_ny:
        print(f"Latest bar {last} != today {today_ny} (holiday / no session). Skipping.")
        return
    asof = last.strftime("%Y-%m-%d")

    results, stale, filtered_out, trend_out, fresh_out, cands = [], 0, 0, 0, {}, []
    for _, row in uni.iterrows():
        df = data.get(row["yf"])
        if df is None:
            continue
        if df.index[-1].date() != last:
            stale += 1
            continue
        c, v = df["Close"], df["Volume"].astype(float)
        e9 = c.ewm(span=FAST, adjust=False).mean()
        s20 = c.rolling(SLOW).mean()
        d0, d1 = e9.iloc[-1] - s20.iloc[-1], e9.iloc[-2] - s20.iloc[-2]
        if np.isnan(d0) or np.isnan(d1):
            continue
        if not d0 > 0:
            continue                              # long only: EMA20 must be above SMA20
        sign = np.sign((e9 - s20).dropna().to_numpy())
        j = len(sign) - 1
        while j >= 0 and sign[j] == sign[-1]:
            j -= 1
        since = len(sign) - 2 - j                 # 0 = crossed today
        kind = "up" if since == 0 else "above"
        k = j
        while k >= 0 and sign[k] == sign[j]:
            k -= 1
        prior_days = j - k                        # length of the regime before the cross
        r10s = rsi(c, RSI_LEN)
        r10 = float(r10s.iloc[-1])
        close = float(c.iloc[-1])
        e20n = float(e9.iloc[-1])
        e50n = float(c.ewm(span=50, adjust=False).mean().iloc[-1])
        e200n = float(c.ewm(span=200, adjust=False).mean().iloc[-1])
        bull = kind == "up"
        h, l = df["High"].to_numpy(float), df["Low"].to_numpy(float)
        cn = c.to_numpy(float)
        tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - cn[:-1]), abs(l[1:] - cn[:-1])))
        atr = float(pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
        rsi_d = float(r10s.iloc[-1] - r10s.iloc[-1 - RSI_SLOPE])
        rsi_d1 = float(r10s.iloc[-1] - r10s.iloc[-2])
        s20n = float(s20.iloc[-1])
        ext = (close - s20n) / atr
        hi60, lo60 = float(h[-RANGE_DAYS:].max()), float(l[-RANGE_DAYS:].min())
        pos60 = (close - lo60) / (hi60 - lo60) * 100 if hi60 > lo60 else 50.0
        m = dict(sym=row["sym"], bull=bull, since=since, prior=prior_days, r10=r10, rsi_d=rsi_d,
                 rsi_d1=rsi_d1, close=close, e20=e20n, s20=s20n, e50=e50n, e200=e200n, ext=ext, pos=pos60)
        cands.append(m)
        why = None if RSI_LO <= r10 <= RSI_HI else "rsi_range"
        if why == "rsi_range":
            filtered_out += 1
            continue
        if why == "trend":
            trend_out += 1
            continue
        if why:
            fresh_out[why] = fresh_out.get(why, 0) + 1
            continue
        ema = lambda n: float(c.ewm(span=n, adjust=False).mean().iloc[-1])  # noqa: E731
        sma = lambda n: float(c.rolling(n).mean().iloc[-1]) if len(c) >= n else None  # noqa: E731
        close = float(c.iloc[-1])
        sma50, sma200 = sma(50), sma(200)
        r14 = rsi(c, 14)
        vol50 = float(v.iloc[-51:-1].mean())
        seg = df.iloc[-60:]
        results.append(dict(
            sym=row["sym"], name=row["name"], kind=kind,
            sector=SECTOR_HE.get(row["sector"], row["sector"]),
            close=round(close, 2),
            chg=round((close / float(c.iloc[-2]) - 1) * 100, 2),
            fast=round(float(e9.iloc[-1]), 2), sma20=round(float(s20.iloc[-1]), 2),
            gap=round(d0 / float(s20.iloc[-1]) * 100, 2),
            vol_ratio=round(float(v.iloc[-1]) / vol50, 2) if vol50 else None,
            rsi10=round(r10, 1),
            rsi14=round(float(r14.iloc[-1]), 1),
            rsi35_ema10=round(float(rsi(c, 35).ewm(span=10, adjust=False).mean().iloc[-1]), 1),
            ema20=round(ema(20), 2), ema30=round(ema(30), 2), ema50=round(ema(50), 2),
            ema200=round(ema(200), 2), sma150=round(sma(150), 2) if sma(150) else None,
            above50=bool(sma50 and close > sma50), above200=bool(sma200 and close > sma200),
            prior_days=int(prior_days), since=int(since),
            atr=round(atr, 2), ext=round(ext, 2), pos60=round(pos60, 1), rsi_d=round(rsi_d, 1),
            rsi_d1=round(rsi_d1, 1),
            chart=dict(
                d=[x.strftime("%m-%d") for x in seg.index],
                c=[round(float(x), 2) for x in seg["Close"]],
                f=[round(float(x), 2) for x in e9.iloc[-60:]],
                s20=[None if np.isnan(x) else round(float(x), 2) for x in s20.iloc[-60:]],
            ),
        ))
    results.sort(key=lambda x: (x["since"], -(x["vol_ratio"] or 0)))
    payload = dict(
        asof=asof, generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        intraday=bool(last == today_ny and datetime.now(ZoneInfo("America/New_York")).hour < 16),
        filtered_out=filtered_out, trend_out=trend_out, fresh_out=fresh_out,
        params=dict(fast=FAST, slow=SLOW, rsi_len=RSI_LEN, rsi_lo=RSI_LO, rsi_hi=RSI_HI,
                    long_only=True),
        scanned=len(data) - stale, universe=len(uni), stale=stale, source=source,
        results=results,
    )
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    with open(os.path.join(OUT, "data", f"{asof}.json"), "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    hist = []
    for fn in sorted(os.listdir(os.path.join(OUT, "data")), reverse=True)[:60]:
        if fn.endswith(".json"):
            with open(os.path.join(OUT, "data", fn)) as f:
                p = json.load(f)
            hist.append(dict(date=p["asof"],
                             **{k: [r["sym"] for r in p["results"] if r["kind"] == k]
                                for k in ("up", "above")}))
    payload["history"] = hist
    with open(os.path.join(HERE, "template_cross20.html"), encoding="utf-8") as f:
        tpl = f.read()
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(tpl.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False)))
    print(f"{asof}: " + ", ".join(f"{k} {sum(r['kind'] == k for r in results)}"
                                  for k in ("up", "above")))


if __name__ == "__main__":
    run()
