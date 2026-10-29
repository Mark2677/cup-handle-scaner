"""
Daily S&P 500 scanner: EMA30/EMA50 trend + RSI(35) vs its EMA(10) trigger.
Runs in GitHub Actions at 23:10 Israel time on US trading days.
Writes docs/trend/index.html (Hebrew dashboard) and docs/trend/data/<date>.json.

LONG  (both must hold at today's close):
  A. EMA30 > EMA50                         (tagged "cross" if it crossed up today)
  B. RSI35 crossed above its EMA10 today   -> "rsi_cross"
     or RSI35 is still below its EMA10, the gap is <= APPROACH points
     and the gap narrowed vs yesterday     -> "rsi_near"  (just before the cross)
SHORT: mirror image (EMA30 < EMA50; RSI35 crossing / about to cross below its EMA10).
Prices are split-adjusted, not dividend-adjusted (matches TradingView / Finviz).
"""
import json
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

from ema_cross import download
from scanner import SECTOR_HE, get_universe, rsi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "trend")

EMA_FAST, EMA_SLOW = 30, 50
RSI_LEN, RSI_SIG = 35, 10
APPROACH = float(os.environ.get("APPROACH", "0"))   # 0 = only real crosses; >0 also lists "about to cross" (RSI points)


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

    results, stale = [], 0
    for _, row in uni.iterrows():
        df = data.get(row["yf"])
        if df is None:
            continue
        if df.index[-1].date() != last or len(df) < 120:
            stale += 1
            continue
        c, v = df["Close"], df["Volume"].astype(float)
        e30 = c.ewm(span=EMA_FAST, adjust=False).mean()
        e50 = c.ewm(span=EMA_SLOW, adjust=False).mean()
        r35 = rsi(c, RSI_LEN)
        sig = r35.ewm(span=RSI_SIG, adjust=False).mean()
        t0, t1 = e30.iloc[-1] - e50.iloc[-1], e30.iloc[-2] - e50.iloc[-2]
        g0, g1 = r35.iloc[-1] - sig.iloc[-1], r35.iloc[-2] - sig.iloc[-2]
        if any(np.isnan(x) for x in (t0, t1, g0, g1)):
            continue
        kind = rsi_state = None
        if t0 > 0:                                   # long side
            if g0 > 0 >= g1:
                rsi_state = "cross"
            elif APPROACH > 0 and -APPROACH <= g0 < 0 and g0 > g1:
                rsi_state = "near"
            if rsi_state:
                kind, ema_cross = "long", t1 <= 0
        elif t0 < 0:                                 # short side
            if g0 < 0 <= g1:
                rsi_state = "cross"
            elif APPROACH > 0 and 0 < g0 <= APPROACH and g0 < g1:
                rsi_state = "near"
            if rsi_state:
                kind, ema_cross = "short", t1 >= 0
        if not kind:
            continue
        close = float(c.iloc[-1])
        ema = lambda n: round(float(c.ewm(span=n, adjust=False).mean().iloc[-1]), 2)  # noqa: E731
        sma = lambda n: float(c.rolling(n).mean().iloc[-1]) if len(c) >= n else None  # noqa: E731
        s50, s200, s150, s20 = sma(50), sma(200), sma(150), sma(20)
        vol50 = float(v.iloc[-51:-1].mean())
        # how many days EMA30 has been on the current side of EMA50
        side = np.sign((e30 - e50).to_numpy())
        k = len(side) - 1
        while k >= 0 and side[k] == side[-1]:
            k -= 1
        trend_days = len(side) - 1 - k
        rnd = lambda a: [round(float(x), 2) for x in a]  # noqa: E731
        results.append(dict(
            sym=row["sym"], name=row["name"], kind=kind,
            sector=SECTOR_HE.get(row["sector"], row["sector"]),
            ema_cross=bool(ema_cross), rsi_state=rsi_state,
            close=round(close, 2), chg=round((close / float(c.iloc[-2]) - 1) * 100, 2),
            ema30=round(float(e30.iloc[-1]), 2), ema50_t=round(float(e50.iloc[-1]), 2),
            trend_gap=round(t0 / float(e50.iloc[-1]) * 100, 2), trend_days=int(trend_days),
            rsi35=round(float(r35.iloc[-1]), 2), rsi35_sig=round(float(sig.iloc[-1]), 2),
            rsi_gap=round(float(g0), 2),
            vol_ratio=round(float(v.iloc[-1]) / vol50, 2) if vol50 else None,
            rsi14=round(float(rsi(c, 14).iloc[-1]), 1),
            ema20=ema(20), ema200=ema(200),
            sma20=round(s20, 2) if s20 else None, sma150=round(s150, 2) if s150 else None,
            above50=bool(s50 and close > s50), above200=bool(s200 and close > s200),
            chart=dict(c=rnd(c.iloc[-90:]), e30=rnd(e30.iloc[-90:]), e50=rnd(e50.iloc[-90:]),
                       r=rnd(r35.iloc[-90:]), s=rnd(sig.iloc[-90:])),
        ))
    rank = {"cross": 0, "near": 1}
    results.sort(key=lambda x: (not x["ema_cross"], rank[x["rsi_state"]], -(x["vol_ratio"] or 0)))
    payload = dict(
        asof=asof, generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        intraday=bool(last == now_ny.date() and now_ny.hour < 16),
        scanned=len(data) - stale, universe=len(uni), stale=stale, source=source,
        params=dict(ema_fast=EMA_FAST, ema_slow=EMA_SLOW, rsi_len=RSI_LEN, rsi_sig=RSI_SIG,
                    approach=APPROACH),
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
                             long=[r["sym"] for r in p["results"] if r["kind"] == "long"],
                             short=[r["sym"] for r in p["results"] if r["kind"] == "short"]))
    payload["history"] = hist
    with open(os.path.join(HERE, "template_trend.html"), encoding="utf-8") as f:
        tpl = f.read()
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(tpl.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False)))
    by = lambda k, s: sum(r["kind"] == k and r["rsi_state"] == s for r in results)  # noqa: E731
    print(f"{asof}: long {by('long','cross')} cross + {by('long','near')} near | "
          f"short {by('short','cross')} cross + {by('short','near')} near")


if __name__ == "__main__":
    run()
