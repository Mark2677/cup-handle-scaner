"""
Daily S&P 500 scanner: EMA(9) crosses SMA(9), no RSI condition.
Runs in GitHub Actions at 23:10 Israel time on US trading days.
Writes docs/cross9/index.html (Hebrew dashboard) and docs/cross9/data/<date>.json.

Cross up   : EMA9 > SMA9 at today's close, and EMA9 <= SMA9 at yesterday's close.
Cross down : EMA9 < SMA9 at today's close, and EMA9 >= SMA9 at yesterday's close.
Prices are split-adjusted but NOT dividend-adjusted, to match TradingView / Finviz charts.
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
OUT = os.path.join(HERE, "docs", "cross9")

FAST, SLOW = 9, 9            # EMA length, SMA length
RSI_LEN, RSI_LO, RSI_HI = 10, 0.0, 100.0   # RSI shown for info only (no filter)


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

    results, stale, filtered_out = [], 0, 0
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
        if d0 > 0 >= d1:
            kind = "up"
        elif d0 < 0 <= d1:
            kind = "down"
        else:
            continue
        r10 = float(rsi(c, RSI_LEN).iloc[-1])
        if not (RSI_LO <= r10 <= RSI_HI):
            filtered_out += 1
            continue
        ema = lambda n: float(c.ewm(span=n, adjust=False).mean().iloc[-1])  # noqa: E731
        sma = lambda n: float(c.rolling(n).mean().iloc[-1]) if len(c) >= n else None  # noqa: E731
        close = float(c.iloc[-1])
        sma50, sma200 = sma(50), sma(200)
        r14 = rsi(c, 14)
        vol50 = float(v.iloc[-51:-1].mean())
        # days since the previous opposite cross (how long the prior regime lasted)
        sign = np.sign((e9 - s20).dropna().to_numpy())
        k = len(sign) - 2
        while k >= 0 and sign[k] == sign[-2]:
            k -= 1
        prior_days = len(sign) - 2 - k
        seg = df.iloc[-60:]
        results.append(dict(
            sym=row["sym"], name=row["name"], kind=kind,
            sector=SECTOR_HE.get(row["sector"], row["sector"]),
            close=round(close, 2),
            chg=round((close / float(c.iloc[-2]) - 1) * 100, 2),
            fast=round(float(e9.iloc[-1]), 2), slow=round(float(s20.iloc[-1]), 2), sma20=sma(20) and round(sma(20), 2),
            gap=round(d0 / float(s20.iloc[-1]) * 100, 2),
            vol_ratio=round(float(v.iloc[-1]) / vol50, 2) if vol50 else None,
            rsi10=round(r10, 1),
            rsi14=round(float(r14.iloc[-1]), 1),
            rsi35_ema10=round(float(rsi(c, 35).ewm(span=10, adjust=False).mean().iloc[-1]), 1),
            ema20=round(ema(20), 2), ema30=round(ema(30), 2), ema50=round(ema(50), 2),
            ema200=round(ema(200), 2), sma150=round(sma(150), 2) if sma(150) else None,
            above50=bool(sma50 and close > sma50), above200=bool(sma200 and close > sma200),
            prior_days=int(prior_days),
            chart=dict(
                d=[x.strftime("%m-%d") for x in seg.index],
                c=[round(float(x), 2) for x in seg["Close"]],
                f=[round(float(x), 2) for x in e9.iloc[-60:]],
                s20=[None if np.isnan(x) else round(float(x), 2) for x in s20.iloc[-60:]],
            ),
        ))
    results.sort(key=lambda x: -(x["vol_ratio"] or 0))
    payload = dict(
        asof=asof, generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        intraday=bool(last == today_ny and datetime.now(ZoneInfo("America/New_York")).hour < 16),
        filtered_out=filtered_out,
        params=dict(fast=FAST, slow=SLOW, rsi_len=RSI_LEN, rsi_lo=RSI_LO, rsi_hi=RSI_HI),
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
                             up=[r["sym"] for r in p["results"] if r["kind"] == "up"],
                             down=[r["sym"] for r in p["results"] if r["kind"] == "down"]))
    payload["history"] = hist
    with open(os.path.join(HERE, "template_cross9.html"), encoding="utf-8") as f:
        tpl = f.read()
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(tpl.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False)))
    print(f"{asof}: {sum(r['kind'] == 'up' for r in results)} up, "
          f"{sum(r['kind'] == 'down' for r in results)} down")


if __name__ == "__main__":
    run()
