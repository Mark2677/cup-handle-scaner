"""
Pre-Sweep Scanner | LONG  —  S&P 500 + Nasdaq-100.
Runs in GitHub Actions at 23:10 Israel time on US trading days.
Writes docs/presweep/index.html (Hebrew dashboard) and docs/presweep/data/<date>.json.

Goal: at the close, find stocks sitting just above a support, so a bullish liquidity sweep
(a wick under the support that is bought back) may happen in the next session.
Alerts / execution are on the 15-minute chart; this scan only prepares the list.

Base filters : price > $15, 50-day avg volume >= 1M shares, market cap > $2B.
Scenario A   : horizontal support from wick lows, found in the last 60 / 120 / 252 bars:
               >= 2 separate touches, or one significant low followed by a >= 2 ATR rally
               within 10 bars. The level is intact (no low under it since it formed),
               today's close is above it by at most 0.5 ATR(14), and today's low did not cross it.
Scenario B   : SMA20 as dynamic support: close above SMA20 by at most 0.5 ATR, today's low
               above SMA20. Prior reactions to SMA20 (held + bounced) are counted and shown;
               overlap with a horizontal support is flagged.
Info only    : open gap-down above price, short float (flag >= 10% / >= 20%), ATR $ / %,
               distance from SMA20 and from the support (% and ATR), support type.
Prices are split-adjusted, not dividend-adjusted (matches TradingView / Finviz).
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from channel import NDX_FALLBACK, _index, _nasdaq100_api, _wiki_tables
from ema_cross import download
from scanner import SECTOR_HE, get_universe as sp500_universe, rsi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "presweep")
FUND = os.path.join(HERE, "data", "presweep_fund.json")

P = dict(
    min_price=15.0, min_vol=1_000_000, min_mcap=2e9,
    depths=[60, 120, 252],
    swing=3,               # pivot low = lowest of +-3 bars
    touch_tol=0.3,         # touches within 0.3 ATR above the level belong to it
    touch_gap=5,           # touches closer than 5 bars are one visit
    min_touches=2,
    react_atr=2.0, react_bars=10,   # single significant low: >= 2 ATR rally within 10 bars
    near_atr=0.5,          # close at most 0.5 ATR(14) above the support
    sma_len=20, sma_look=60, sma_tol=0.25, sma_bounce=1.0, sma_bounce_bars=5,
    gap_look=252,
    short_hi=20.0, short_mid=10.0,
)
FUND_MAX_AGE_DAYS = 7


def get_universe():
    sp, s1 = sp500_universe()
    ndx, s2 = _index("Nasdaq-100", NDX_FALLBACK, [
        _nasdaq100_api, lambda: _wiki_tables(["Nasdaq-100", "List_of_Nasdaq-100_companies"])])
    rows = {}
    for _, r in sp.iterrows():
        rows[r["sym"]] = dict(sym=r["sym"], name=r["name"], sector=r["sector"], idx=["S&P 500"])
    for _, r in ndx.iterrows():
        if r["sym"] in rows:
            rows[r["sym"]]["idx"].append("Nasdaq-100")
        else:
            rows[r["sym"]] = dict(sym=r["sym"], name=r["name"], sector=r["sector"], idx=["Nasdaq-100"])
    uni = pd.DataFrame(rows.values())
    uni["yf"] = uni["sym"].str.replace(".", "-", regex=False)
    src = f"S&P 500: {s1} · Nasdaq-100: {s2}"
    print(f"Universe: {len(uni)} unique tickers ({src})")
    return uni, src


def wilder_atr(h, l, c, n=14):
    tr = np.r_[h[0] - l[0], np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))]
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def pivots_low(l, k):
    return [i for i in range(k, len(l) - k) if l[i] == l[i - k:i + k + 1].min()]


def visits(idx, gap):
    out = []
    for i in sorted(idx):
        if not out or i - out[-1] >= gap:
            out.append(i)
    return out


def supports(h, l, c, atr):
    """All intact horizontal supports below today's close (today excluded from touches)."""
    n = len(c)
    k = P["swing"]
    piv = [i for i in pivots_low(l, k) if i >= n - max(P["depths"]) and i <= n - 2]
    a_now = atr[-1]
    found = {}
    for p in piv:
        L = l[p]
        if not L < c[-1] or not l[-1] > L:
            continue
        tch = [q for q in piv if L <= l[q] <= L + P["touch_tol"] * a_now]
        first = min(tch)
        if l[first:].min() < L:                     # something traded below it -> not intact
            continue
        vis = visits(tch, P["touch_gap"])
        rally = (h[p + 1:p + 1 + P["react_bars"]].max() - L) / atr[p] if p + 1 < n else 0.0
        if len(vis) < P["min_touches"] and rally < P["react_atr"]:
            continue
        key = round(L, 2)
        depth = next(d for d in P["depths"] if first >= n - d)
        if key not in found:
            found[key] = dict(level=float(L), touches=vis, n_touch=len(vis), rally=float(rally),
                              first=int(first), last=int(max(tch)), depth=int(depth),
                              kind="multi" if len(vis) >= P["min_touches"] else "single")
    return sorted(found.values(), key=lambda s: -s["level"])


def sma_reactions(h, l, c, sma, atr):
    """Past bars that tested SMA20 from above, held it at the close and bounced >= 1 ATR."""
    n = len(c)
    ev = []
    for i in range(n - P["sma_look"], n - 3):
        if np.isnan(sma[i]):
            continue
        if l[i] <= sma[i] + P["sma_tol"] * atr[i] and c[i] >= sma[i]:
            if h[i + 1:i + 1 + P["sma_bounce_bars"]].max() - c[i] >= P["sma_bounce"] * atr[i]:
                ev.append(i)
    return visits(ev, P["touch_gap"])


def open_gap_down(h, l, c):
    """Nearest unfilled gap-down zone above today's close in the last gap_look bars."""
    n = len(c)
    best = None
    for i in range(max(1, n - P["gap_look"]), n):
        if h[i] < l[i - 1]:
            top = l[i - 1]
            after = h[i + 1:].max() if i + 1 < n else -np.inf
            if after >= top:
                continue                                     # fully filled
            bottom = max(h[i], after)
            if bottom <= c[-1]:
                continue
            if best is None or bottom < best["bottom"]:
                best = dict(bottom=float(bottom), top=float(top), bar=i)
    return best


def load_fund():
    try:
        with open(FUND) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def fetch_fund(tickers, cache):
    import yfinance as yf
    now = time.time()
    todo = [t for t in tickers if t not in cache or now - cache[t].get("ts", 0) > FUND_MAX_AGE_DAYS * 86400]
    print(f"Fundamentals: {len(tickers) - len(todo)} cached, fetching {len(todo)}")
    for t in todo:
        for attempt in range(3):
            try:
                info = yf.Ticker(t).get_info()
                sf = info.get("shortPercentOfFloat")
                cache[t] = dict(ts=now, mcap=info.get("marketCap"),
                                short_float=None if sf is None else round(float(sf) * 100, 2),
                                short_date=info.get("dateShortInterest"))
                break
            except Exception as e:  # noqa: BLE001
                print(t, "info retry", attempt, str(e)[:80])
                time.sleep(5 * (attempt + 1))
        time.sleep(0.4)
    os.makedirs(os.path.dirname(FUND), exist_ok=True)
    with open(FUND, "w") as f:
        json.dump(cache, f)
    return cache


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

    cands, stale, funnel = [], 0, dict(price=0, volume=0, setup=0, mcap=0)
    for _, row in uni.iterrows():
        df = data.get(row["yf"])
        if df is None:
            continue
        if df.index[-1].date() != last or len(df) < 130:
            stale += 1
            continue
        c, h, l = (df[k].to_numpy(float) for k in ("Close", "High", "Low"))
        v = df["Volume"].to_numpy(float)
        close = c[-1]
        if close <= P["min_price"]:
            continue
        funnel["price"] += 1
        vol50 = float(v[-51:-1].mean())
        if vol50 < P["min_vol"]:
            continue
        funnel["volume"] += 1
        atr = wilder_atr(h, l, c)
        a_now = float(atr[-1])
        sma = pd.Series(c).rolling(P["sma_len"]).mean().to_numpy()
        lv = supports(h, l, c, atr)
        near = [s for s in lv if close - s["level"] <= P["near_atr"] * a_now]
        A = near[0] if near else None
        s20 = float(sma[-1])
        B = bool(close > s20 and close - s20 <= P["near_atr"] * a_now and l[-1] > s20)
        if not A and not B:
            continue
        funnel["setup"] += 1
        overlap = next((s for s in lv if abs(s["level"] - s20) <= P["near_atr"] * a_now), None)
        kind = "AB" if (A and B) else "A" if A else "B"
        sup = A or overlap
        react = sma_reactions(h, l, c, sma, atr)
        gap = open_gap_down(h, l, c)
        cs = pd.Series(c)
        ema = lambda n: round(float(cs.ewm(span=n, adjust=False).mean().iloc[-1]), 2)  # noqa: E731
        smav = lambda n: float(cs.rolling(n).mean().iloc[-1]) if len(cs) >= n else None  # noqa: E731
        s50, s200, s150 = smav(50), smav(200), smav(150)
        view = 120
        off = len(c) - view
        rnd = lambda arr: [None if np.isnan(x) else round(float(x), 2) for x in arr]  # noqa: E731
        lvl = A["level"] if A else s20
        cands.append(dict(
            sym=row["sym"], yf=row["yf"], name=row["name"], idx=row["idx"], kind=kind,
            sector=SECTOR_HE.get(row["sector"], row["sector"]) or "—",
            close=round(close, 2), chg=round((close / c[-2] - 1) * 100, 2),
            atr=round(a_now, 2), atr_pct=round(a_now / close * 100, 2),
            vol50=int(vol50), vol_ratio=round(v[-1] / vol50, 2) if vol50 else None,
            sma20=round(s20, 2), sma20_atr=round((close - s20) / a_now, 2),
            sma20_pct=round((close / s20 - 1) * 100, 2),
            sma20_slope=round((s20 / sma[-6] - 1) * 100, 2),
            sma_react=len(react),
            sup=None if not sup else dict(
                level=round(sup["level"], 2), type=sup["kind"], touches=sup["n_touch"],
                rally=round(sup["rally"], 1), depth=sup["depth"], age=int(len(c) - 1 - sup["last"]),
                atr=round((close - sup["level"]) / a_now, 2), pct=round((close / sup["level"] - 1) * 100, 2)),
            overlap=bool(overlap) if B else bool(A and abs(A["level"] - s20) <= P["near_atr"] * a_now),
            alert=round(lvl, 2), dist_atr=round((close - lvl) / a_now, 2),
            gap=None if not gap else dict(bottom=round(gap["bottom"], 2), top=round(gap["top"], 2),
                                          dist=round((gap["bottom"] / close - 1) * 100, 2),
                                          ago=int(len(c) - 1 - gap["bar"])),
            rsi14=round(float(rsi(df["Close"], 14).iloc[-1]), 1),
            rsi35_ema10=round(float(rsi(df["Close"], 35).ewm(span=10, adjust=False).mean().iloc[-1]), 1),
            ema20=ema(20), ema30=ema(30), ema50=ema(50), ema200=ema(200),
            sma150=round(s150, 2) if s150 else None,
            above50=bool(s50 and close > s50), above200=bool(s200 and close > s200),
            chart=dict(c=rnd(c[off:]), h=rnd(h[off:]), l=rnd(l[off:]), sma=rnd(sma[off:]),
                       first=None if not sup else max(0, sup["first"] - off),
                       touch=[] if not sup else [t - off for t in sup["touches"] if t >= off],
                       react=[t - off for t in react if t >= off],
                       gap=None if not gap else [max(0, gap["bar"] - off), round(gap["bottom"], 2), round(gap["top"], 2)]),
        ))
    # market cap + short float only for the stocks that have a setup (few calls, cached 7 days)
    cache = fetch_fund([x["yf"] for x in cands], load_fund())
    results = []
    for x in cands:
        fd = cache.get(x["yf"], {})
        mc = fd.get("mcap")
        if mc is not None and mc < P["min_mcap"]:
            continue
        funnel["mcap"] += 1
        x["mcap"] = mc
        x["short_float"] = fd.get("short_float")
        x["short_date"] = fd.get("short_date")
        del x["yf"]
        results.append(x)
    order = {"AB": 0, "A": 1, "B": 2}
    results.sort(key=lambda r: (order[r["kind"]], r["dist_atr"]))
    payload = dict(
        asof=asof, generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        intraday=bool(last == now_ny.date() and now_ny.hour < 16),
        scanned=len(data) - stale, universe=len(uni), stale=stale, source=source,
        funnel=funnel, params=P, results=results,
    )
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    with open(os.path.join(OUT, "data", f"{asof}.json"), "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    hist = []
    for fn in sorted(os.listdir(os.path.join(OUT, "data")), reverse=True)[:60]:
        if fn.endswith(".json"):
            with open(os.path.join(OUT, "data", fn)) as f:
                p = json.load(f)
            hist.append(dict(date=p["asof"], **{k: [r["sym"] for r in p["results"] if r["kind"] == k]
                                                for k in ("AB", "A", "B")}))
    payload["history"] = hist
    with open(os.path.join(HERE, "template_presweep.html"), encoding="utf-8") as f:
        tpl = f.read()
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(tpl.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False)))
    print(f"{asof}: funnel {funnel}; " + ", ".join(f"{k} {sum(r['kind'] == k for r in results)}" for k in ("AB", "A", "B")))


if __name__ == "__main__":
    run()
