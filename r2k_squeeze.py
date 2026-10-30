"""
Russell 2000 scanner: heavily-shorted, high-beta small caps that are building a base.

Stage 1 - filters (Finviz-style):
    Beta > 2, Short Float > 20%, Short Ratio > 2, Inst Own < 50%, Avg Volume (3m) > 500K
Stage 2 - base patterns (daily candles, ~2 years):
    * cup with handle            (reuses scanner.find_cup_handle)
    * double / triple / multi bottom
    * flat base / consolidation
Runs in GitHub Actions at 23:10 Israel time. Writes docs/r2k/index.html and docs/r2k/data/<date>.json.

Data: universe = iShares IWM holdings (cached in data/r2000.csv).
Prices from Yahoo (split-adjusted). Short Float / Short Ratio / Inst Own / Beta from Yahoo
quote summary, cached in data/r2k_fundamentals.json and refreshed weekly (short interest is
published only twice a month).
"""
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import scanner as CUP
from scanner import rsi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "r2k")
DATA = os.path.join(HERE, "data")

F = dict(beta=2.0, short_float=20.0, short_ratio=2.0, inst_own=50.0, avg_vol=500_000)
FUND_MAX_AGE_DAYS = 7
MIN_PRICE = 1.0

# pattern parameters
MB = dict(lookback=150, swing=5, tol=0.05, min_sep=10, min_rally=0.10, recent=60)
FB = dict(min_len=25, max_len=65, max_range=0.20)   # + no trend inside the base
NEAR = 0.05          # "near pivot" = within 5% below
BREAKOUT_VOL = 1.4   # breakout volume vs 50-day average

SECTOR_HE = {
    "Information Technology": "טכנולוגיה", "Health Care": "בריאות", "Financials": "פיננסים",
    "Consumer Discretionary": "צריכה מחזורית", "Consumer Staples": "צריכה בסיסית",
    "Industrials": "תעשייה", "Energy": "אנרגיה", "Materials": "חומרי גלם",
    "Utilities": "תשתיות", "Real Estate": 'נדל"ן', "Communication": "תקשורת",
    "Communication Services": "תקשורת",
}


# ------------------------------------------------------------------ universe
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "text/csv,application/json,text/html;q=0.9,*/*;q=0.8",
      "Accept-Language": "en-US,en;q=0.9"}


def _ishares():
    import requests
    ses = requests.Session()
    ses.headers.update(UA)
    page = "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf"
    ses.get(page + "?switchLocale=y&siteEntryPassthrough=true", timeout=60)
    url = (page + "/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
           "&siteEntryPassthrough=true")
    r = ses.get(url, headers={"Referer": page}, timeout=60)
    txt = r.content.decode("utf-8-sig", errors="replace")
    print(f"iShares: HTTP {r.status_code}, {len(txt)} chars, starts {txt[:80]!r}")
    lines = txt.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.lstrip('"').startswith("Ticker"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), on_bad_lines="skip")
    df = df[df["Asset Class"] == "Equity"]
    return pd.DataFrame({"sym": df["Ticker"].astype(str).str.strip(),
                         "name": df["Name"].astype(str).str.title(), "sector": df["Sector"]})


def _vanguard():
    import requests
    rows, start = [], 1
    while start < 3000:
        url = ("https://investor.vanguard.com/investment-products/etfs/profile/api/VTWO/"
               f"portfolio-holding/stock?start={start}&count=500")
        r = requests.get(url, headers=UA, timeout=60)
        print(f"Vanguard: HTTP {r.status_code}, {len(r.text)} chars")
        ent = r.json().get("fund", {}).get("entity", [])
        if not ent:
            break
        rows += [dict(sym=e.get("ticker"), name=str(e.get("longName", "")).title(),
                      sector=e.get("sectorName") or e.get("sector") or "") for e in ent]
        start += 500
    return pd.DataFrame(rows)


def get_universe():
    os.makedirs(DATA, exist_ok=True)
    cache = os.path.join(DATA, "r2000.csv")
    df, source = None, None
    for name, fn in (("iShares IWM", _ishares), ("Vanguard VTWO", _vanguard)):
        try:
            d = fn()
            d = d[d["sym"].astype(str).str.match(r"^[A-Z][A-Z0-9.\- ]*$")]
            if len(d) >= 1500:
                df, source = d, name
                break
            print(f"{name}: only {len(d)} rows")
        except Exception as e:  # noqa: BLE001
            print(f"{name} failed: {type(e).__name__} {e}")
    if df is not None:
        df.to_csv(cache, index=False)
    elif os.path.exists(cache):
        df, source = pd.read_csv(cache), "cached list"
    else:
        sys.exit("No Russell 2000 list available")
    df["yf"] = df["sym"].str.replace(r"[. ]", "-", regex=True)
    print(f"Universe: {len(df)} ({source})")
    return df.reset_index(drop=True), source


# ------------------------------------------------------------------ prices
def download(tickers, period, interval="1d", min_len=60):
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        for attempt in range(3):
            try:
                d = yf.download(chunk, period=period, interval=interval, auto_adjust=False,
                                group_by="ticker", progress=False, threads=True)
                break
            except Exception as e:  # noqa: BLE001
                print("retry", attempt, e)
                time.sleep(15)
        else:
            continue
        for t in chunk:
            try:
                df = d[t].dropna(subset=["Close"])
                if len(df) >= min_len:
                    out[t] = df
            except KeyError:
                pass
        time.sleep(2)
    print(f"Downloaded {len(out)}/{len(tickers)} ({period} {interval})")
    return out


def calc_beta(monthly, spy):
    """5-year monthly beta vs SPY (same method as Yahoo / Finviz)."""
    r = monthly["Close"].pct_change()
    m = spy["Close"].pct_change()
    j = pd.concat([r, m], axis=1, join="inner").dropna().iloc[-60:]
    if len(j) < 24:
        return None
    return float(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / np.var(j.iloc[:, 1], ddof=1))


# ------------------------------------------------------------------ fundamentals
def load_fund():
    p = os.path.join(DATA, "r2k_fundamentals.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_fund(d):
    with open(os.path.join(DATA, "r2k_fundamentals.json"), "w") as f:
        json.dump(d, f, indent=0, sort_keys=True)


def fetch_fund(tickers, cache):
    import yfinance as yf
    now = time.time()
    todo = [t for t in tickers
            if t not in cache or now - cache[t].get("ts", 0) > FUND_MAX_AGE_DAYS * 86400]
    print(f"Fundamentals: {len(tickers) - len(todo)} cached, fetching {len(todo)}")
    for n, t in enumerate(todo):
        info = None
        for attempt in range(3):
            try:
                info = yf.Ticker(t).get_info()
                break
            except Exception as e:  # noqa: BLE001
                print(t, "info retry", attempt, str(e)[:80])
                time.sleep(5 * (attempt + 1))
        if info:
            pct = lambda k: None if info.get(k) is None else round(float(info[k]) * 100, 2)  # noqa: E731
            cache[t] = dict(ts=now, short_float=pct("shortPercentOfFloat"),
                            short_ratio=info.get("shortRatio"),
                            inst_own=pct("heldPercentInstitutions"),
                            beta_y=info.get("beta"), float_sh=info.get("floatShares"),
                            short_sh=info.get("sharesShort"),
                            short_date=info.get("dateShortInterest"),
                            mcap=info.get("marketCap"), industry=info.get("industry"))
        time.sleep(0.4)
        if n % 50 == 49:
            save_fund(cache)
    save_fund(cache)
    return cache


# ------------------------------------------------------------------ patterns
def swing_lows(l, k):
    idx = []
    for i in range(k, len(l) - 1):
        lo, hi = max(0, i - k), min(len(l), i + k + 1)
        if l[i] == l[lo:hi].min():
            idx.append(i)
    return idx


def multi_bottom(df):
    """Two or more swing lows at about the same level, a rally between them, price still above."""
    h, l, c = (df[k].to_numpy(float) for k in ("High", "Low", "Close"))
    n = len(c)
    s = max(0, n - MB["lookback"])
    lows = [i + s for i in swing_lows(l[s:], MB["swing"])]
    best = None
    for a in range(len(lows)):
        base = l[lows[a]]
        grp = [lows[a]]
        for b in lows[a + 1:]:
            if abs(l[b] / base - 1) > MB["tol"] or b - grp[-1] < MB["min_sep"]:
                continue
            # a real rally between consecutive bottoms (the middle of the "W")
            if h[grp[-1]:b + 1].max() < max(l[grp[-1]], l[b]) * (1 + MB["min_rally"]):
                continue
            grp.append(b)
        if len(grp) < 2 or n - 1 - grp[-1] > MB["recent"]:
            continue
        floor = min(l[g] for g in grp)
        if l[grp[0]:].min() < floor * 0.99:            # broke below the bottoms -> invalid
            continue
        neck = h[grp[0]:grp[-1] + 1].max()
        if neck / max(l[g] for g in grp) - 1 < MB["min_rally"]:
            continue
        if c[-1] <= floor:
            continue
        cand = dict(type="multi_bottom", bottoms=grp, pivot=float(neck),
                    floor=float(floor), count=len(grp), start=grp[0], end=n - 1,
                    depth=float(neck / floor - 1))
        if best is None or cand["count"] > best["count"] or (
                cand["count"] == best["count"] and cand["start"] > best["start"]):
            best = cand
    return best


def touches(mask, gap=5):
    """number of separate visits to a zone (visits closer than `gap` bars are merged)"""
    idx = np.flatnonzero(mask)
    return 0 if not len(idx) else 1 + int((np.diff(idx) >= gap).sum())


def flat_base(df):
    h, l, c = (df[k].to_numpy(float) for k in ("High", "Low", "Close"))
    n = len(c)
    best = None
    for L in range(FB["min_len"], FB["max_len"] + 1):
        if L + 1 > n:
            break
        seg_h, seg_l = h[n - 1 - L:n - 1].max(), l[n - 1 - L:n - 1].min()  # excluding today
        rng = seg_h / seg_l - 1
        seg_c = c[n - 1 - L:n - 1]
        drift = abs(np.polyfit(np.arange(L), seg_c, 1)[0]) * L        # trend inside the "base"
        if (rng <= FB["max_range"] and drift <= 0.5 * (seg_h - seg_l)
                and touches(l[n - 1 - L:n - 1] <= seg_l + 0.25 * (seg_h - seg_l)) >= 2
                and touches(h[n - 1 - L:n - 1] >= seg_h - 0.25 * (seg_h - seg_l)) >= 2):
            best = dict(type="flat_base", pivot=float(seg_h), floor=float(seg_l),
                        start=n - 1 - L, end=n - 1, weeks=L / 5, depth=float(rng))
    return best


def cup(df):
    CUP.P["min_price"] = MIN_PRICE
    m = CUP.find_cup_handle(df)
    if not m:
        return None
    return dict(type="cup", pivot=float(m["pivot"]), start=m["L"], end=len(df) - 1,
                bottoms=[m["b"]], rims=[m["L"], m["r"]], depth=float(m["depth"]),
                weeks=m["cup_weeks"], handle_depth=float(m["handle_depth"]), score=m["score"])


def status(close, prev, pivot, vol_ratio):
    if close > pivot and prev <= pivot and vol_ratio >= BREAKOUT_VOL:
        return "breakout"
    if close > pivot:
        return "above"
    if close >= pivot * (1 - NEAR):
        return "near"
    return "forming"


# ------------------------------------------------------------------ main
def run():
    uni, source = get_universe()
    daily = download(uni["yf"].tolist(), "2y", "1d", 120)
    if not daily:
        sys.exit("No price data")
    last = max(df.index[-1] for df in daily.values()).date()
    now_ny = datetime.now(ZoneInfo("America/New_York"))
    if os.environ.get("FORCE") != "1" and last != now_ny.date():
        print(f"Latest bar {last} != today {now_ny.date()}. Skipping.")
        return
    asof = last.strftime("%Y-%m-%d")

    # stage 1a: liquidity + price (from price data)
    liq = {}
    for t, df in daily.items():
        av = float(df["Volume"].iloc[-63:].mean())
        if av > F["avg_vol"] and float(df["Close"].iloc[-1]) >= MIN_PRICE:
            liq[t] = av
    print(f"Avg volume > {F['avg_vol']:,}: {len(liq)}")

    # stage 1b: beta (5y monthly vs SPY) as a pre-filter
    monthly = download(list(liq) + ["SPY"], "5y", "1mo", 24)
    spy = monthly.pop("SPY", None)
    betas = {t: calc_beta(m, spy) for t, m in monthly.items()} if spy is not None else {}
    pre = [t for t in liq if (betas.get(t) or 0) >= F["beta"] - 0.4]
    print(f"Beta pre-filter (>= {F['beta'] - 0.4}): {len(pre)}")

    # stage 1c: short interest / institutions / Yahoo beta
    fund = fetch_fund(pre, load_fund())
    passed = []
    for t in pre:
        fd = fund.get(t, {})
        beta = fd.get("beta_y") if fd.get("beta_y") is not None else betas.get(t)
        checks = dict(
            beta=beta is not None and beta > F["beta"],
            short_float=(fd.get("short_float") or 0) > F["short_float"],
            short_ratio=(fd.get("short_ratio") or 0) > F["short_ratio"],
            inst_own=fd.get("inst_own") is not None and fd["inst_own"] < F["inst_own"],
        )
        if all(checks.values()):
            passed.append((t, beta, fd))
    print(f"Passed all filters: {len(passed)}")

    meta = uni.set_index("yf")
    results = []
    for t, beta, fd in passed:
        df = daily[t]
        c, v = df["Close"], df["Volume"].astype(float)
        close, prev = float(c.iloc[-1]), float(c.iloc[-2])
        vol50 = float(v.iloc[-51:-1].mean())
        vr = float(v.iloc[-1]) / vol50 if vol50 else 0
        pats = []
        for fn in (cup, multi_bottom, flat_base):
            try:
                p = fn(df)
            except Exception as e:  # noqa: BLE001
                print(t, fn.__name__, e)
                p = None
            if p:
                p["status"] = status(close, prev, p["pivot"], vr)
                p["dist"] = round((close / p["pivot"] - 1) * 100, 2)
                pats.append(p)
        order = {"breakout": 0, "above": 1, "near": 2, "forming": 3}
        pats.sort(key=lambda p: (order[p["status"]], abs(p["dist"])))
        n = len(c)
        start = max(0, min([p["start"] for p in pats] + [n - 130]) - 15)
        ema = lambda k: round(float(c.ewm(span=k, adjust=False).mean().iloc[-1]), 2)  # noqa: E731
        sma = lambda k: round(float(c.rolling(k).mean().iloc[-1]), 2) if n >= k else None  # noqa: E731
        row = meta.loc[t]
        results.append(dict(
            sym=row["sym"], name=row["name"], sector=SECTOR_HE.get(row["sector"], row["sector"]),
            industry=fd.get("industry"), close=round(close, 2),
            chg=round((close / prev - 1) * 100, 2), vol_ratio=round(vr, 2),
            beta=round(beta, 2), beta_calc=None if betas.get(t) is None else round(betas[t], 2),
            short_float=fd.get("short_float"), short_ratio=fd.get("short_ratio"),
            inst_own=fd.get("inst_own"), avg_vol=int(liq[t]), mcap=fd.get("mcap"),
            short_date=fd.get("short_date"),
            rsi14=round(float(rsi(c, 14).iloc[-1]), 1),
            rsi35_ema10=round(float(rsi(c, 35).ewm(span=10, adjust=False).mean().iloc[-1]), 1),
            ema20=ema(20), ema30=ema(30), ema50=ema(50), ema200=ema(200),
            sma20=sma(20), sma150=sma(150),
            patterns=[dict(type=p["type"], status=p["status"], pivot=round(p["pivot"], 2),
                           dist=p["dist"], depth=round(p["depth"] * 100, 1),
                           count=p.get("count"), weeks=round(p["weeks"], 1) if p.get("weeks") else
                           round((p["end"] - p["start"]) / 5, 1),
                           floor=round(p["floor"], 2) if p.get("floor") else None,
                           marks=[i - start for i in p.get("bottoms", [])],
                           rims=[i - start for i in p.get("rims", [])],
                           s=p["start"] - start)
                      for p in pats],
            chart=dict(c=[round(float(x), 2) for x in c.iloc[start:]],
                       d=[x.strftime("%Y-%m-%d") for x in df.index[start:]]),
        ))
    rank = {"breakout": 0, "above": 1, "near": 2, "forming": 3}
    results.sort(key=lambda r: (0 if r["patterns"] else 1,
                                rank[r["patterns"][0]["status"]] if r["patterns"] else 9,
                                -(r["short_float"] or 0)))
    payload = dict(
        asof=asof, generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        intraday=bool(last == now_ny.date() and now_ny.hour < 16),
        universe=len(uni), scanned=len(daily), source=source,
        funnel=dict(liquid=len(liq), beta_pre=len(pre), passed=len(passed),
                    with_pattern=sum(1 for r in results if r["patterns"])),
        filters=F, mb=MB, fb=FB, results=results,
    )
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    with open(os.path.join(OUT, "data", f"{asof}.json"), "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    hist = []
    for fn in sorted(os.listdir(os.path.join(OUT, "data")), reverse=True)[:60]:
        if fn.endswith(".json"):
            with open(os.path.join(OUT, "data", fn)) as f:
                p = json.load(f)
            hist.append(dict(date=p["asof"], passed=len(p["results"]),
                             patterns=[r["sym"] for r in p["results"] if r["patterns"]],
                             breakout=[r["sym"] for r in p["results"]
                                       if any(x["status"] == "breakout" for x in r["patterns"])]))
    payload["history"] = hist
    with open(os.path.join(HERE, "template_r2k.html"), encoding="utf-8") as f:
        tpl = f.read()
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(tpl.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False)))
    print(f"{asof}: {len(results)} passed filters, "
          f"{payload['funnel']['with_pattern']} with a base pattern")


if __name__ == "__main__":
    run()
