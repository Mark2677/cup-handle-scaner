"""
Daily S&P 500 scanner: Bollinger Band rejection + Parabolic SAR flip.
Runs in GitHub Actions at 23:10 Israel time on US trading days.
Writes docs/bbsar/index.html (Hebrew dashboard) and docs/bbsar/data/<date>.json.

LONG  : within the last TOUCH_WINDOW bars a low touched/pierced the lower band,
        today's close is back inside the bands (between lower and upper),
        and today the Parabolic SAR flipped from above price to below price.
SHORT : mirror image on the upper band (close inside the bands), SAR flips from below to above price.
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
OUT = os.path.join(HERE, "docs", "bbsar")

BB_LEN, BB_MULT = 20, 2.0
SAR_START, SAR_INC, SAR_MAX = 0.05, 0.05, 0.5
TOUCH_WINDOW = 3  # today + 2 previous bars


def psar(high, low, close, start=SAR_START, inc=SAR_INC, mx=SAR_MAX):
    """Parabolic SAR, a line-by-line port of TradingView's ta.sar().
    Returns (sar values, is_below) where is_below=True means SAR is under price (uptrend)."""
    n = len(close)
    sar = np.full(n, np.nan)
    below = np.zeros(n, dtype=bool)
    result = max_min = acc = np.nan
    is_below = False
    for i in range(1, n):
        first = False
        if i == 1:
            if close[1] > close[0]:
                is_below, max_min, result = True, high[1], low[0]
            else:
                is_below, max_min, result = False, low[1], high[0]
            first, acc = True, start
        result = result + acc * (max_min - result)
        if is_below:
            if result > low[i]:
                first, is_below = True, False
                result, max_min, acc = max(high[i], max_min), low[i], start
        else:
            if result < high[i]:
                first, is_below = True, True
                result, max_min, acc = min(low[i], max_min), high[i], start
        if not first:
            if is_below:
                if high[i] > max_min:
                    max_min, acc = high[i], min(acc + inc, mx)
            else:
                if low[i] < max_min:
                    max_min, acc = low[i], min(acc + inc, mx)
        if is_below:
            result = min(result, low[i - 1])
            if i > 1:
                result = min(result, low[i - 2])
        else:
            result = max(result, high[i - 1])
            if i > 1:
                result = max(result, high[i - 2])
        sar[i], below[i] = result, is_below
    return sar, below


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
        if df.index[-1].date() != last:
            stale += 1
            continue
        c = df["Close"]
        h, l = df["High"].to_numpy(float), df["Low"].to_numpy(float)
        mid = c.rolling(BB_LEN).mean()
        sd = c.rolling(BB_LEN).std(ddof=0)          # TradingView uses population stdev
        up, lo = mid + BB_MULT * sd, mid - BB_MULT * sd
        sar, below = psar(h, l, c.to_numpy(float))
        if np.isnan(lo.iloc[-1]) or np.isnan(sar[-1]):
            continue
        w = slice(-TOUCH_WINDOW, None)
        close = float(c.iloc[-1])
        kind = None
        if below[-1] and not below[-2]:                       # SAR flipped under price
            touch = l[w] <= lo.to_numpy()[w]
            if touch.any() and lo.iloc[-1] < close < up.iloc[-1]:
                kind = "long"
        elif not below[-1] and below[-2]:                     # SAR flipped over price
            touch = h[w] >= up.to_numpy()[w]
            if touch.any() and lo.iloc[-1] < close < up.iloc[-1]:
                kind = "short"
        if not kind:
            continue
        touch_ago = int(TOUCH_WINDOW - 1 - np.where(touch)[0].max())
        v = df["Volume"].astype(float)
        vol50 = float(v.iloc[-51:-1].mean())
        ema = lambda n: round(float(c.ewm(span=n, adjust=False).mean().iloc[-1]), 2)  # noqa: E731
        sma = lambda n: float(c.rolling(n).mean().iloc[-1]) if len(c) >= n else None  # noqa: E731
        s50, s200, s150 = sma(50), sma(200), sma(150)
        width = float(up.iloc[-1] - lo.iloc[-1])
        seg = slice(-60, None)
        rnd = lambda a: [None if np.isnan(x) else round(float(x), 2) for x in a]  # noqa: E731
        results.append(dict(
            sym=row["sym"], name=row["name"], kind=kind,
            sector=SECTOR_HE.get(row["sector"], row["sector"]),
            close=round(close, 2), chg=round((close / float(c.iloc[-2]) - 1) * 100, 2),
            bb_up=round(float(up.iloc[-1]), 2), bb_mid=round(float(mid.iloc[-1]), 2),
            bb_lo=round(float(lo.iloc[-1]), 2),
            pctb=round((close - float(lo.iloc[-1])) / width * 100, 1) if width else None,
            bw=round(width / float(mid.iloc[-1]) * 100, 2),
            sar=round(float(sar[-1]), 2), sar_dist=round((float(sar[-1]) / close - 1) * 100, 2),
            touch_ago=touch_ago,
            vol_ratio=round(float(v.iloc[-1]) / vol50, 2) if vol50 else None,
            rsi14=round(float(rsi(c, 14).iloc[-1]), 1),
            rsi35_ema10=round(float(rsi(c, 35).ewm(span=10, adjust=False).mean().iloc[-1]), 1),
            ema20=ema(20), ema30=ema(30), ema50=ema(50), ema200=ema(200),
            sma150=round(s150, 2) if s150 else None,
            above50=bool(s50 and close > s50), above200=bool(s200 and close > s200),
            chart=dict(c=rnd(c.to_numpy()[seg]), h=rnd(h[seg]), l=rnd(l[seg]),
                       up=rnd(up.to_numpy()[seg]), mid=rnd(mid.to_numpy()[seg]),
                       lo=rnd(lo.to_numpy()[seg]), sar=rnd(sar[seg]),
                       below=[bool(x) for x in below[seg]]),
        ))
    results.sort(key=lambda x: -(x["vol_ratio"] or 0))
    payload = dict(
        asof=asof, generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        intraday=bool(last == now_ny.date() and now_ny.hour < 16),
        scanned=len(data) - stale, universe=len(uni), stale=stale, source=source,
        params=dict(bb_len=BB_LEN, bb_mult=BB_MULT, sar=[SAR_START, SAR_INC, SAR_MAX],
                    window=TOUCH_WINDOW),
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
    with open(os.path.join(HERE, "template_bbsar.html"), encoding="utf-8") as f:
        tpl = f.read()
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(tpl.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False)))
    print(f"{asof}: {sum(r['kind'] == 'long' for r in results)} long, "
          f"{sum(r['kind'] == 'short' for r in results)} short")


if __name__ == "__main__":
    run()
