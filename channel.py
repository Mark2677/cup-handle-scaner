"""
Daily price-channel scanner: S&P 500 + Nasdaq-100 + Dow Jones 30 (deduplicated).
Runs in GitHub Actions at 23:10 Israel time on US trading days.
Writes docs/channel/index.html (Hebrew dashboard) and docs/channel/data/<date>.json.

A channel is drawn the way a trader draws one by hand:
  * one line through two swing lows (or two swing highs) inside the last 2-6 months,
  * a parallel line through the opposite extreme of the same period,
  * every bar since the first anchor stays inside (small wick/close tolerance),
  * at least 2 separate touches of each line and 5 in total,
  * the channel really slopes (rise/fall over its life >= 0.75 x its width).
The best channel per stock (most touches, then longest) is kept, and the stock is
listed when today's close is at / very near the lower or upper line
(at most 0.5 ATR inside the line, or at most 0.25 ATR outside it).
Prices are split-adjusted, not dividend-adjusted (matches TradingView / Finviz).
"""
import io
import json
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ema_cross import download
from scanner import SECTOR_HE, get_universe as sp500_universe, rsi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "channel")

P = dict(
    min_len=40, max_len=126,   # channel life: ~2 to ~6 months of daily bars
    swing=3,                   # swing high/low = extreme of +-3 bars
    min_gap=10,                # bars between the two anchors
    touch=0.35,                # a bar "touches" a line within 0.35 ATR
    pierce=0.5,                # wicks may poke through a line by 0.5 ATR
    close_out=0.25,            # closes may sit outside a line by 0.25 ATR
    min_width=2.5,             # channel at least 2.5 ATR wide
    trend=0.75,                # move over the channel life >= 0.75 x width
    min_side=2, min_total=5,   # touches per line / in total (before today)
    near_in=0.5,               # today's close at most 0.5 ATR inside the line ...
    near_out=0.25,             # ... or at most 0.25 ATR outside it
)

NDX_FALLBACK = """AAPL ABNB ADBE ADI ADP ADSK AEP AMAT AMD AMGN AMZN ANSS APP ARM ASML AVGO AXON AZN BIIB BKNG
BKR CCEP CDNS CDW CEG CHTR CMCSA COST CPRT CRWD CSCO CSGP CSX CTAS CTSH DASH DDOG DXCM EA EXC FANG FAST
FTNT GEHC GFS GILD GOOG GOOGL HON IDXX INTC INTU ISRG KDP KHC KLAC LIN LRCX LULU MAR MCHP MDLZ MELI META
MNST MRVL MSFT MSTR MU NFLX NVDA NXPI ODFL ON ORLY PANW PAYX PCAR PDD PEP PLTR PYPL QCOM REGN ROP ROST
SBUX SHOP SNPS TEAM TMUS TRI TSLA TTD TTWO TXN VRSK VRTX WBD WDAY XEL ZS""".split()
DOW_FALLBACK = """AAPL AMGN AMZN AXP BA CAT CRM CSCO CVX DIS GS HD HON IBM JNJ JPM KO MCD MMM MRK MSFT NKE NVDA
PG SHW TRV UNH V VZ WMT""".split()


def _wiki(page, fallback, label):
    try:
        import requests
        html = requests.get(f"https://en.wikipedia.org/wiki/{page}",
                            headers={"User-Agent": "Mozilla/5.0 (channel-scanner)"}, timeout=30).text
        for t in pd.read_html(io.StringIO(html)):
            cols = {str(c).strip(): c for c in t.columns}
            sc = next((cols[k] for k in ("Ticker", "Symbol") if k in cols), None)
            if sc is None or not (25 <= len(t) <= 120):
                continue
            nc = next((cols[k] for k in ("Company", "Security") if k in cols), sc)
            ec = next((cols[k] for k in ("GICS Sector", "Industry", "Sector") if k in cols), None)
            df = pd.DataFrame(dict(sym=t[sc].astype(str).str.strip(), name=t[nc].astype(str),
                                   sector=t[ec].astype(str) if ec is not None else ""))
            df = df[df["sym"].str.fullmatch(r"[A-Z.\-]{1,6}")]
            if len(df) >= 25:
                print(f"{label}: {len(df)} tickers (Wikipedia)")
                return df, "Wikipedia"
        raise ValueError("no constituents table")
    except Exception as e:  # noqa: BLE001
        print(f"{label}: Wikipedia failed ({e}); using bundled list")
        return pd.DataFrame(dict(sym=fallback, name=fallback, sector="")), "fallback"


def get_universe():
    sp, s1 = sp500_universe()
    ndx, s2 = _wiki("Nasdaq-100", NDX_FALLBACK, "Nasdaq-100")
    dow, s3 = _wiki("Dow_Jones_Industrial_Average", DOW_FALLBACK, "Dow 30")
    rows = {}
    for _, r in sp.iterrows():
        rows[r["sym"]] = dict(sym=r["sym"], name=r["name"], sector=r["sector"], idx=["S&P 500"])
    for df, tag in ((ndx, "Nasdaq-100"), (dow, "Dow 30")):
        for _, r in df.iterrows():
            if r["sym"] in rows:
                rows[r["sym"]]["idx"].append(tag)
            else:
                rows[r["sym"]] = dict(sym=r["sym"], name=r["name"], sector=r["sector"], idx=[tag])
    uni = pd.DataFrame(rows.values())
    uni["yf"] = uni["sym"].str.replace(".", "-", regex=False)
    src = f"S&P 500: {s1} · Nasdaq-100: {s2} · Dow 30: {s3}"
    print(f"Universe: {len(uni)} unique tickers ({src})")
    return uni, src


def swings(x, k, kind):
    out = []
    for i in range(k, len(x) - k):
        w = x[i - k:i + k + 1]
        if (kind == "low" and x[i] == w.min()) or (kind == "high" and x[i] == w.max()):
            out.append(i)
    return out


def visits(mask, gap=5):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    groups = np.split(idx, np.flatnonzero(np.diff(idx) >= gap) + 1)
    return [int(g[0]) for g in groups]


def find_channel(h, l, c, atr):
    """Best pivot channel ending today, or None. Bars 0..n-1, today = n-1."""
    n = len(c)
    k = P["swing"]
    lo_piv = [i for i in swings(l, k, "low") if i >= n - P["max_len"]]
    hi_piv = [i for i in swings(h, k, "high") if i >= n - P["max_len"]]
    best = None
    for anchor, piv in (("low", lo_piv), ("high", hi_piv)):
        src = l if anchor == "low" else h
        for ai in range(len(piv)):
            p = piv[ai]
            life = n - 1 - p
            if life < P["min_len"]:
                continue
            a = float(np.median(atr[p:n]))
            if not a > 0:
                continue
            seg = slice(p, n - 1)                      # channel history, today excluded
            t = np.arange(p, n)
            for q in piv[ai + 1:]:
                if q - p < P["min_gap"] or q > n - 1 - k:
                    continue
                slope = (src[q] - src[p]) / (q - p)
                base = src[p] + slope * (t - p)        # anchor line over p..today
                b = base[:-1]
                if anchor == "low":
                    if (l[seg] < b - P["pierce"] * a).any():
                        continue
                    width = float((h[seg] - b).max())
                    lower, upper = base, base + width
                else:
                    if (h[seg] > b + P["pierce"] * a).any():
                        continue
                    width = float((b - l[seg]).max())
                    lower, upper = base - width, base
                if width < P["min_width"] * a:
                    continue
                if abs(slope) * life < P["trend"] * width:
                    continue
                lo_h, up_h = lower[:-1], upper[:-1]
                cs = c[seg]
                if (cs < lo_h - P["close_out"] * a).any() or (cs > up_h + P["close_out"] * a).any():
                    continue
                if (l[seg] < lo_h - P["pierce"] * a).any() or (h[seg] > up_h + P["pierce"] * a).any():
                    continue
                tl = visits(l[seg] <= lo_h + P["touch"] * a)
                tu = visits(h[seg] >= up_h - P["touch"] * a)
                if min(len(tl), len(tu)) < P["min_side"] or len(tl) + len(tu) < P["min_total"]:
                    continue
                score = (len(tl) + len(tu), life)
                if best is None or score > best["score"]:
                    best = dict(score=score, start=p, anchor=anchor, slope=float(slope),
                                lower=lower, upper=upper, width=width, atr_med=a,
                                t_low=[p + i for i in tl], t_up=[p + i for i in tu])
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

    results, stale, with_channel = [], 0, 0
    for _, row in uni.iterrows():
        df = data.get(row["yf"])
        if df is None:
            continue
        if df.index[-1].date() != last or len(df) < P["max_len"] + 30:
            stale += 1
            continue
        c = df["Close"]
        cn, h, l = c.to_numpy(float), df["High"].to_numpy(float), df["Low"].to_numpy(float)
        tr = np.r_[h[0] - l[0], np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - cn[:-1]), abs(l[1:] - cn[:-1])))]
        atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
        ch = find_channel(h, l, cn, atr)
        if not ch:
            continue
        with_channel += 1
        close, a_now = cn[-1], float(atr[-1])
        lo0, up0 = float(ch["lower"][-1]), float(ch["upper"][-1])
        d_lo, d_up = (close - lo0) / a_now, (up0 - close) / a_now   # >0 = inside
        trend = "up" if ch["slope"] > 0 else "down"
        if -P["near_out"] <= d_lo <= P["near_in"]:
            edge, line, dist = "low", lo0, d_lo
        elif -P["near_out"] <= d_up <= P["near_in"]:
            edge, line, dist = "high", up0, d_up
        else:
            continue
        kind = f"{trend}_{edge}"
        v = df["Volume"].astype(float)
        vol50 = float(v.iloc[-51:-1].mean())
        ema = lambda n: round(float(c.ewm(span=n, adjust=False).mean().iloc[-1]), 2)  # noqa: E731
        sma = lambda n: float(c.rolling(n).mean().iloc[-1]) if len(c) >= n else None  # noqa: E731
        s50, s200, s150, s20 = sma(50), sma(200), sma(150), sma(20)
        n = len(cn)
        view = max(0, min(ch["start"] - 15, n - 90))
        off = ch["start"] - view
        rnd = lambda arr: [round(float(x), 2) for x in arr]  # noqa: E731
        results.append(dict(
            sym=row["sym"], name=row["name"], kind=kind, trend=trend, edge=edge,
            idx=row["idx"], sector=SECTOR_HE.get(row["sector"], row["sector"]) or "—",
            close=round(float(close), 2), chg=round((close / cn[-2] - 1) * 100, 2),
            line=round(line, 2), other=round(up0 if edge == "low" else lo0, 2),
            dist_atr=round(float(dist), 2), dist_pct=round((close / line - 1) * 100, 2),
            touched=bool(l[-1] <= lo0 + P["touch"] * a_now) if edge == "low"
            else bool(h[-1] >= up0 - P["touch"] * a_now),
            pos=round((close - lo0) / (up0 - lo0) * 100, 1),
            width_pct=round((up0 - lo0) / close * 100, 2),
            slope_m=round(ch["slope"] * 21 / close * 100, 2),        # % per month
            life=int(n - 1 - ch["start"]), anchor=ch["anchor"],
            t_low=len(ch["t_low"]), t_up=len(ch["t_up"]),
            atr=round(a_now, 2),
            vol_ratio=round(float(v.iloc[-1]) / vol50, 2) if vol50 else None,
            rsi14=round(float(rsi(c, 14).iloc[-1]), 1),
            rsi35_ema10=round(float(rsi(c, 35).ewm(span=10, adjust=False).mean().iloc[-1]), 1),
            ema20=ema(20), ema30=ema(30), ema50=ema(50), ema200=ema(200),
            sma20=round(s20, 2) if s20 else None, sma150=round(s150, 2) if s150 else None,
            above50=bool(s50 and close > s50), above200=bool(s200 and close > s200),
            chart=dict(c=rnd(cn[view:]), h=rnd(h[view:]), l=rnd(l[view:]), off=int(off),
                       lo=rnd(ch["lower"]), up=rnd(ch["upper"]),
                       tl=[i - view for i in ch["t_low"]], tu=[i - view for i in ch["t_up"]]),
        ))
    results.sort(key=lambda x: (abs(x["dist_atr"]), -(x["t_low"] + x["t_up"])))
    payload = dict(
        asof=asof, generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        intraday=bool(last == now_ny.date() and now_ny.hour < 16),
        scanned=len(data) - stale, universe=len(uni), stale=stale, source=source,
        with_channel=with_channel, params=P, results=results,
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
                                                for k in ("up_low", "up_high", "down_low", "down_high")}))
    payload["history"] = hist
    with open(os.path.join(HERE, "template_channel.html"), encoding="utf-8") as f:
        tpl = f.read()
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(tpl.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False)))
    cnt = {k: sum(r["kind"] == k for r in results) for k in ("up_low", "up_high", "down_low", "down_high")}
    print(f"{asof}: {with_channel} stocks in a channel; at an edge today: {cnt}")


if __name__ == "__main__":
    run()
