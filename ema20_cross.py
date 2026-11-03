"""
Daily S&P 500 scanner: EMA(20) crosses SMA(20) + RSI(10) filter.
Runs in GitHub Actions at 23:10 Israel time on US trading days.
Writes docs/cross20/index.html (Hebrew dashboard) and docs/cross20/data/<date>.json.

Goal: stocks at the START of a momentum move.
Long : EMA20 crossed above SMA20 today or in the last FRESH bars, after the opposite regime lasted
       >= PRIOR_MIN bars; RSI(10) in 40-60 and rising; close above EMA20 and SMA20 but at most
       MAX_EXT_ATR ATR above SMA20; close not in the top of its 60-day range; long-term uptrend
       (EMA50 > EMA200, close > EMA200).
Short: mirror image.
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
TREND = True     # long: EMA50 > EMA200 and close > EMA200 (short: mirror)
# "start of momentum" rules
FRESH = 2        # EMA20 crossed SMA20 today or in the last FRESH bars (0 = today only)
PRIOR_MIN = 10   # the opposite regime before the cross lasted at least this many bars
RSI_SLOPE = 3    # RSI10 rising (long) / falling (short) vs this many bars ago
MAX_EXT_ATR = 1.5  # close at most this many ATR(14) away from SMA20 (not extended yet)
RANGE_DAYS, RANGE_MAX = 60, 70.0  # long: close in the lower 70% of the 60-day range; short: upper 70%

FRESH_MAX = 5    # crosses up to this old are collected (for the variant report in the log)


def reject(m, rsi_l=(RSI_LO, RSI_HI), rsi_s=(RSI_LO, RSI_HI), trend=TREND, fresh=FRESH):
    """Return the first rule a candidate fails (None = passes). m = metrics dict."""
    b = m["bull"]
    lo, hi = rsi_l if b else rsi_s
    if m["since"] > fresh:
        return "fresh"
    if not lo <= m["r10"] <= hi:
        return "rsi_range"
    if trend and not ((m["e50"] > m["e200"] and m["close"] > m["e200"]) if b
                      else (m["e50"] < m["e200"] and m["close"] < m["e200"])):
        return "trend"
    if m["prior"] < PRIOR_MIN:
        return "prior"                            # whipsaw, not a turn
    if (m["rsi_d"] <= 0 or m["rsi_d1"] <= 0) if b else (m["rsi_d"] >= 0 or m["rsi_d1"] >= 0):
        return "rsi"                              # momentum not building, or turning against us today
    if (m["close"] <= max(m["e20"], m["s20"])) if b else (m["close"] >= min(m["e20"], m["s20"])):
        return "side"                             # price not on the momentum side of both averages
    if (m["ext"] > MAX_EXT_ATR) if b else (m["ext"] < -MAX_EXT_ATR):
        return "ext"                              # already ran away from SMA20
    if (m["pos"] > RANGE_MAX) if b else (m["pos"] < 100 - RANGE_MAX):
        return "range"                            # already at the far end of its 60-day range
    return None


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
        if d0 == 0:
            continue
        kind = "up" if d0 > 0 else "down"
        sign = np.sign((e9 - s20).dropna().to_numpy())
        j = len(sign) - 1
        while j >= 0 and sign[j] == sign[-1]:
            j -= 1
        since = len(sign) - 2 - j                 # 0 = crossed today
        if since > FRESH_MAX:
            continue
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
        why = reject(m)
        if why == "fresh":
            continue
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
    variants = {
        "A: RSI 40-70 long / 30-60 short": dict(rsi_l=(40, 70), rsi_s=(30, 60)),
        "B: no long-term trend filter": dict(trend=False),
        "C: cross within 5 days": dict(fresh=4),
        "A+C": dict(rsi_l=(40, 70), rsi_s=(30, 60), fresh=4),
        "A+B+C": dict(rsi_l=(40, 70), rsi_s=(30, 60), fresh=4, trend=False),
    }
    for name, kw in variants.items():
        ok = [m for m in cands if reject(m, **kw) is None]
        print(f"variant {name}: long {[x['sym'] for x in ok if x['bull']]} short {[x['sym'] for x in ok if not x['bull']]}")
    payload = dict(
        asof=asof, generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        intraday=bool(last == today_ny and datetime.now(ZoneInfo("America/New_York")).hour < 16),
        filtered_out=filtered_out, trend_out=trend_out, fresh_out=fresh_out,
        params=dict(fast=FAST, slow=SLOW, rsi_len=RSI_LEN, rsi_lo=RSI_LO, rsi_hi=RSI_HI,
                    trend=TREND, fresh=FRESH, prior_min=PRIOR_MIN, rsi_slope=RSI_SLOPE,
                    max_ext=MAX_EXT_ATR, range_days=RANGE_DAYS, range_max=RANGE_MAX),
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
                                for k in ("up", "down")}))
    payload["history"] = hist
    with open(os.path.join(HERE, "template_cross20.html"), encoding="utf-8") as f:
        tpl = f.read()
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(tpl.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False)))
    print(f"{asof}: " + ", ".join(f"{k} {sum(r['kind'] == k for r in results)}"
                                  for k in ("up", "down")) + f" | rejected {fresh_out}")


if __name__ == "__main__":
    run()
