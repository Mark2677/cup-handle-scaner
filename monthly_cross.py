"""
Monthly EMA5 / EMA8 cross scanner — S&P 500.
Runs in GitHub Actions at 23:10 Israel time on US trading days.
Writes docs/monthly/index.html (Hebrew dashboard) and docs/monthly/data/<date>.json.

Monthly candles (split-adjusted, not dividend-adjusted, like TradingView / Finviz).
The current month is still open, so its candle closes at today's price:
  golden / death        : EMA5 crossed above / below EMA8 on the CURRENT (open) month
                          vs the last closed month. Provisional until the month closes.
  golden_conf/death_conf: the cross happened on the LAST CLOSED month (final).
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ema_cross import download as download_daily
from scanner import SECTOR_HE, get_universe, rsi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "monthly")
FAST, SLOW = 5, 8
MIN_MONTHS = 30
KINDS = ("golden", "death", "golden_conf", "death_conf")


def download_monthly(tickers):
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        for attempt in range(3):
            try:
                d = yf.download(chunk, period="15y", interval="1mo", auto_adjust=False,
                                group_by="ticker", progress=False, threads=True)
                break
            except Exception as e:  # noqa: BLE001
                print("retry", attempt, e)
                time.sleep(10)
        else:
            continue
        for t in chunk:
            try:
                df = d[t].dropna(subset=["Close"])
                if len(df) >= MIN_MONTHS:
                    out[t] = df
            except KeyError:
                pass
        time.sleep(2)
    print(f"Monthly: downloaded {len(out)}/{len(tickers)}")
    return out


def run():
    uni, source = get_universe()
    daily = download_daily(uni["yf"].tolist())
    if not daily:
        sys.exit("No daily data downloaded")
    last = max(df.index[-1] for df in daily.values()).date()
    now_ny = datetime.now(ZoneInfo("America/New_York"))
    if os.environ.get("FORCE") != "1" and last != now_ny.date():
        print(f"Latest bar {last} != today {now_ny.date()} (holiday / no session). Skipping.")
        return
    asof = last.strftime("%Y-%m-%d")
    monthly = download_monthly(uni["yf"].tolist())
    # trading days into the current month, and whether today is its last trading day (weekends only)
    month_days = int(sum(1 for d in pd.bdate_range(last.replace(day=1), last)))
    last_day = (pd.Timestamp(last) + pd.offsets.BDay(1)).month != last.month

    results, stale = [], 0
    for _, row in uni.iterrows():
        m, d = monthly.get(row["yf"]), daily.get(row["yf"])
        if m is None or d is None:
            continue
        if d.index[-1].date() != last:
            stale += 1
            continue
        m = m.copy()
        ym = m.index.to_period("M")
        m = m[~ym.duplicated(keep="first")]            # some yfinance versions add a 2nd row for the open month
        # make sure the open month's candle ends at today's daily close
        if m.index[-1].year == last.year and m.index[-1].month == last.month:
            m.iloc[-1, m.columns.get_loc("Close")] = float(d["Close"].iloc[-1])
        else:
            stale += 1
            continue
        c = m["Close"].astype(float)
        e5 = c.ewm(span=FAST, adjust=False).mean()
        e8 = c.ewm(span=SLOW, adjust=False).mean()
        g = (e5 - e8).to_numpy()
        s0, s1, s2 = np.sign(g[-1]), np.sign(g[-2]), np.sign(g[-3])
        if s0 > 0 >= s1:
            kind = "golden"
        elif s0 < 0 <= s1:
            kind = "death"
        elif s1 > 0 >= s2:
            kind = "golden_conf"
        elif s1 < 0 <= s2:
            kind = "death_conf"
        else:
            continue
        # months the previous regime lasted (before the cross)
        ref = len(g) - (1 if kind in ("golden", "death") else 2)   # index of the cross candle
        side = np.sign(g[ref - 1])
        k = ref - 1
        while k >= 0 and np.sign(g[k]) == side:
            k -= 1
        prior = ref - 1 - k
        dc = d["Close"]
        close = float(dc.iloc[-1])
        mclose_prev = float(c.iloc[-2])
        ema = lambda n: round(float(dc.ewm(span=n, adjust=False).mean().iloc[-1]), 2)  # noqa: E731
        sma = lambda n: float(dc.rolling(n).mean().iloc[-1]) if len(dc) >= n else None  # noqa: E731
        s50, s200, s150, s20 = sma(50), sma(200), sma(150), sma(20)
        # how far today's close is from flipping the open month back (price where EMA5 == EMA8)
        a5, a8 = 2 / (FAST + 1), 2 / (SLOW + 1)
        p5, p8 = float(e5.iloc[-2]), float(e8.iloc[-2])
        flip = (a8 * 0 + (1 - a8) * p8 - (1 - a5) * p5) / (a5 - a8)
        seg = m.iloc[-36:]
        rnd = lambda arr: [round(float(x), 2) for x in arr]  # noqa: E731
        results.append(dict(
            sym=row["sym"], name=row["name"], kind=kind,
            sector=SECTOR_HE.get(row["sector"], row["sector"]),
            close=round(close, 2), chg=round((close / float(dc.iloc[-2]) - 1) * 100, 2),
            mchg=round((close / mclose_prev - 1) * 100, 2),
            ema5=round(float(e5.iloc[-1]), 2), ema8=round(float(e8.iloc[-1]), 2),
            gap=round(float(g[-1]) / float(e8.iloc[-1]) * 100, 2),
            flip=round(flip, 2) if kind in ("golden", "death") else None,
            flip_pct=round((flip / close - 1) * 100, 2) if kind in ("golden", "death") else None,
            prior=int(prior),
            mrsi14=round(float(rsi(c, 14).iloc[-1]), 1),
            rsi14=round(float(rsi(dc, 14).iloc[-1]), 1),
            rsi35_ema10=round(float(rsi(dc, 35).ewm(span=10, adjust=False).mean().iloc[-1]), 1),
            ema20=ema(20), ema30=ema(30), ema50=ema(50), ema200=ema(200),
            sma20=round(s20, 2) if s20 else None, sma150=round(s150, 2) if s150 else None,
            above50=bool(s50 and close > s50), above200=bool(s200 and close > s200),
            chart=dict(d=[x.strftime("%m/%y") for x in seg.index], c=rnd(seg["Close"]),
                       h=rnd(seg["High"]), l=rnd(seg["Low"]), o=rnd(seg["Open"]),
                       e5=rnd(e5.iloc[-36:]), e8=rnd(e8.iloc[-36:])),
        ))
    order = {k: i for i, k in enumerate(KINDS)}
    results.sort(key=lambda x: (order[x["kind"]], -abs(x["gap"])))
    payload = dict(
        asof=asof, generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        intraday=bool(last == now_ny.date() and now_ny.hour < 16),
        month_days=month_days, last_day=bool(last_day),
        scanned=len(monthly) - stale, universe=len(uni), stale=stale, source=source,
        params=dict(fast=FAST, slow=SLOW), results=results,
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
    with open(os.path.join(HERE, "template_monthly.html"), encoding="utf-8") as f:
        tpl = f.read()
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(tpl.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False)))
    print(f"{asof}: " + ", ".join(f"{k} {sum(r['kind'] == k for r in results)}" for k in KINDS))


if __name__ == "__main__":
    run()
