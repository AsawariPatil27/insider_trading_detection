#!/usr/bin/env python3
"""
Synthetic Insider-Trading Surveillance Dataset Generator  --  v3
==================================================================
Same trader/trade/label engine as v2, plus one addition:

 PRE-TRADE RETURNS (point-in-time, no look-ahead)
    For every trade we add pre_trade_return_{1,3,5}d = the stock's own return
    over the N trading days *ending the day before* the trade (never includes
    the trade's own day, so it cannot leak the trade-day/event-day jump).
    This is a real feature (momentum going into the trade), unlike
    price_return_1d/3d which are intentionally forward-looking (validation only).

Everything else (trader generation, insider injection, coordinated groups,
label noise, all v2 derived features) is UNCHANGED from testing_v2.py, same
SEED, so if you run this against the same events file, profile.csv /
trade.csv's original columns / synthetic_metadata.csv / coordinated_groups.csv
reproduce exactly -- this script only adds the pre_trade_return_* columns.

Events source (pick one via env var):
  EVENTS_FILE             -- a real events_clean_24-26.xlsx-style file
  REAL_MASTER_FILE        -- an events_real_master.csv-style file (article-level:
                             one row per event+news-article pair; collapsed to
                             one row per event using event_id, with
                             primary_event_type driving the price-jump bias)
  RECONSTRUCT_EVENTS_FROM -- a directory with a prior run's synthetic_metadata.csv
                             + trade.csv, used as a fallback to rebuild an
                             approximate events table when no real file exists

Usage:  python testing_v3.py   (set the relevant env var above, plus OUT_DIR)
"""
import os
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIG  (unchanged v2 knobs + new one for pre-trade returns)
# --------------------------------------------------------------------------
EVENTS_FILE = os.environ.get("EVENTS_FILE", "events_clean_24-26.xlsx")
RECONSTRUCT_EVENTS_FROM = os.environ.get("RECONSTRUCT_EVENTS_FROM", "")  # dir with old metadata/trade csvs
REAL_MASTER_FILE = os.environ.get("REAL_MASTER_FILE", "")  # events_real_master.csv: real events + real scraped news, article-level
OUT_DIR     = os.environ.get("OUT_DIR", "outputs")

N_TRADERS        = 1000
N_TRADES         = 20000
INSIDER_FRACTION = 0.18
GROUP_SHARE      = 0.30
PRE_EVENT_WINDOW_HOURS = 72

P_LEGIT_LARGE        = 0.08
P_GENUINE_PREWINDOW  = 0.12
P_GENUINE_POSTEVENT  = 0.15
P_BURST              = 0.10
N_INSIDER_CLUSTERS   = 60
N_BENIGN_CLUSTERS    = 40
LABEL_FN_RATE        = 0.05
LABEL_FP_RATE        = 0.002
INTRADAY_NOISE       = 0.004

HIST_START = "2022-07-01"
MON_START  = "2024-01-01"
SEED = 42

# --- new: pre-trade return windows (trading days, ending the day before trade) ---
PRE_TRADE_RETURN_WINDOWS = [1, 3, 5]

rng = np.random.default_rng(SEED)

# --------------------------------------------------------------------------
# 1. EVENTS
# --------------------------------------------------------------------------
def load_real_events(path):
    ev = pd.read_excel(path)
    ev["event_date"] = pd.to_datetime(ev["timestamp_released"], format="%d-%b-%Y", errors="coerce")
    return ev

def load_real_master(path):
    """events_real_master.csv is article-level: one row per (event, news article) pair,
    real scraped coverage (publisher, title, link, pub_ist, phase, hours_from_outcome)
    around real NSE corporate-action disclosures. Collapse to one row per event for the
    generator, and keep the article-level rows separately for the enriched output --
    up to 100 real articles for one event can't be folded into a single cell."""
    raw = pd.read_csv(path)
    event_cols = ["isin", "symbol", "company_name", "event_type", "primary_event_type",
                 "event_date", "meeting_date_nse", "intimation_ts_first", "intimation_ts_last",
                 "outcome_ts", "reaction_date", "bm_purpose", "outcome_desc", "outcome_text",
                 "outcome_pdf", "raw_text", "merged_event_ids", "n_articles"]
    ev = raw.groupby("event_id", as_index=False)[event_cols].first()
    ev["event_date"] = pd.to_datetime(ev["event_date"], errors="coerce")
    ev = ev.rename(columns={"event_type": "event_type_detail"})
    ev["event_type"] = ev["primary_event_type"]   # clean single category drives bias/news logic
    art_cols = ["event_id", "isin", "article_id", "publisher", "publisher_url", "title",
               "link", "pub_ist", "phase", "hours_from_outcome", "kw_in_title"]
    art = raw.loc[raw["article_id"].notna(), art_cols].copy()
    art["pub_ist"] = pd.to_datetime(art["pub_ist"], errors="coerce")
    return ev, art

def reconstruct_events(src_dir):
    """Fallback when the real events file isn't available: rebuild an approximate
    events table from a *previous run's* synthetic_metadata.csv (+ trade.csv for
    company_name where missing). event_type is not recoverable, so we assign a
    plausible category per event, biased by the observed BUY/SELL mix of the
    insider trades already linked to that event_id (positive-news events drew
    mostly BUYs in the original generator, negative-news events mostly SELLs)."""
    meta = pd.read_csv(f"{src_dir}/synthetic_metadata__2_.csv" if os.path.exists(f"{src_dir}/synthetic_metadata__2_.csv")
                        else f"{src_dir}/synthetic_metadata.csv")
    ev = (meta.groupby("event_id")
              .agg(isin=("isin", "first"),
                   event_date=("event_timestamp", "first"),
                   buy_share=("direction", lambda s: (s == "BUY").mean()))
              .reset_index())
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    trade_path = f"{src_dir}/trade__2_.csv" if os.path.exists(f"{src_dir}/trade__2_.csv") else f"{src_dir}/trade.csv"
    tr = pd.read_csv(trade_path, usecols=["isin", "company_name"]).drop_duplicates("isin")
    ev = ev.merge(tr, on="isin", how="left")
    ev["company_name"] = ev["company_name"].fillna(ev["isin"])

    POS_TYPES = ["Bonus Issue", "Stock Split", "Share Buyback", "Interim Dividend", "Final Dividend"]
    NEG_TYPES = ["Fund Raising - Rights Issue", "Fund Raising - QIP", "Fund Raising - Preferential Allotment"]
    NEU_TYPES = ["Investor Update", "Board Meeting Outcome", "Change in Management"]

    def pick_type(b):
        r = rng.random()
        if b >= 0.65:      pool, p = POS_TYPES, 0.85
        elif b <= 0.35:    pool, p = NEG_TYPES, 0.85
        else:              pool, p = NEU_TYPES, 0.70
        if r < p:
            return rng.choice(pool)
        return rng.choice(POS_TYPES + NEG_TYPES + NEU_TYPES)

    ev["event_type"] = [pick_type(b) for b in ev["buy_share"].fillna(0.5)]
    ev["event_id"] = ev["event_id"]
    return ev[["event_id", "isin", "company_name", "event_date", "event_type"]]

real_articles = None
if REAL_MASTER_FILE:
    events, real_articles = load_real_master(REAL_MASTER_FILE)
elif RECONSTRUCT_EVENTS_FROM:
    events = reconstruct_events(RECONSTRUCT_EVENTS_FROM)
elif os.path.exists(EVENTS_FILE):
    events = load_real_events(EVENTS_FILE)
else:
    raise SystemExit(f"No events source: set EVENTS_FILE to a real events .xlsx, "
                      f"REAL_MASTER_FILE to an events_real_master.csv, "
                      f"or RECONSTRUCT_EVENTS_FROM to a directory with a prior run's CSVs.")

events = (events.dropna(subset=["event_date"])
                .drop_duplicates(subset=["isin", "event_date"])
                .reset_index(drop=True))
events_orig = events.copy()   # keep all original columns (raw_text, has_* flags, etc.) for the enriched output
if real_articles is not None:
    real_articles = real_articles[real_articles["event_id"].isin(events["event_id"])].reset_index(drop=True)

securities = events[["isin", "company_name"]].drop_duplicates("isin").reset_index(drop=True)
sec_of = {v: i for i, v in enumerate(securities["isin"])}
events["sec_i"] = events["isin"].map(sec_of)
n_sec = len(securities)

POSITIVE = ["Bonus", "Buyback", "Stock Split", "Dividend"]
NEGATIVE = ["Fund Raising"]

def event_bias(event_type):
    et = str(event_type)
    pos = any(k in et for k in POSITIVE)
    neg = any(k in et for k in NEGATIVE) and not pos
    if pos: return 0.80,  3.5, 3.0
    if neg: return 0.30, -2.0, 3.0
    return 0.55, 0.5, 2.5

bias = events["event_type"].apply(event_bias)
events["buy_prob"]  = [b[0] for b in bias]
events["jump_mean"] = [b[1] for b in bias]
events["jump_sd"]   = [b[2] for b in bias]
events_by_sec = {i: g.index.to_list() for i, g in events.groupby("sec_i")}

# --------------------------------------------------------------------------
# 2. TRADING CALENDAR + PRICE PATHS (weekdays only)
# --------------------------------------------------------------------------
MON_START_TS = pd.Timestamp(MON_START)
MON_END_TS   = events["event_date"].max()
bdays = pd.bdate_range(pd.Timestamp(HIST_START), MON_END_TS + pd.Timedelta(days=20))
n_days = len(bdays)
mon_first = int(bdays.searchsorted(MON_START_TS))
mon_last  = int(bdays.searchsorted(MON_END_TS, side="right") - 1)
events["day_i"] = bdays.searchsorted(events["event_date"]).astype(int)

sigma_px = rng.uniform(0.012, 0.028, n_sec)
rets = rng.standard_normal((n_sec, n_days)) * sigma_px[:, None]
for r in events.itertuples():
    if r.day_i < n_days:
        rets[r.sec_i, r.day_i] += rng.normal(r.jump_mean, r.jump_sd) / 100.0
P0 = np.exp(rng.uniform(np.log(60), np.log(3500), n_sec))
P = P0[:, None] * np.exp(np.cumsum(rets, axis=1))

def days_before_event(ev_date, offsets):
    out = []
    for k in offsets:
        d = ev_date - pd.Timedelta(days=int(k))
        if d.weekday() < 5:
            i = int(bdays.searchsorted(d))
            if i < n_days and mon_first <= i <= mon_last and bdays[i] == d:
                out.append(i)
    return out

# --------------------------------------------------------------------------
# 3. TRADERS
# --------------------------------------------------------------------------
ROLE_CLASSES = ["Promoter", "KMP", "Designated Person", "Director", "Employee", "Connected Person"]
ROLE_WEIGHTS = [0.05, 0.10, 0.20, 0.10, 0.45, 0.10]
DEPARTMENTS  = ["Finance", "Legal", "Compliance", "Sales", "IT",
                "Operations", "HR", "Strategy", "Investor Relations", "R&D"]
PRECLEAR_REQUIRED_PROB = {"Promoter": 0.95, "KMP": 0.90, "Designated Person": 0.85,
                          "Director": 0.90, "Employee": 0.15, "Connected Person": 0.40}

trader_ids = [f"TRD{str(i).zfill(5)}" for i in range(1, N_TRADERS + 1)]
role       = rng.choice(ROLE_CLASSES, size=N_TRADERS, p=ROLE_WEIGHTS)
department = rng.choice(DEPARTMENTS, size=N_TRADERS)
preclear   = np.array(["Required" if rng.random() < PRECLEAR_REQUIRED_PROB[r] else "Not Required" for r in role])
prone      = np.isin(role, ["Promoter", "KMP", "Designated Person", "Director"])
prone_idx, other_idx = np.where(prone)[0], np.where(~prone)[0]

lam       = rng.gamma(2.0, 0.5, N_TRADERS) + 0.15
base_qty  = rng.lognormal(6.0, 0.8, N_TRADERS)
size_sig  = rng.uniform(0.30, 0.70, N_TRADERS)
size_drift = rng.normal(0.0, 0.20, N_TRADERS)
watch = [rng.choice(n_sec, size=int(rng.integers(3, 9)), replace=False) for _ in range(N_TRADERS)]

def make_pan():
    L = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return ("".join(rng.choice(L, 5)) + "".join(rng.choice(list("0123456789"), 4)) + rng.choice(L))

pans = [make_pan() for _ in range(N_TRADERS)]

used = set()
def take(pool, k):
    cand = np.array([i for i in pool if int(i) not in used])
    pick = [int(x) for x in rng.choice(cand, size=k, replace=False)]
    used.update(pick)
    return pick

insider_clusters, benign_clusters = [], []
for _ in range(N_INSIDER_CLUSTERS):
    insider_clusters.append(take(prone_idx, 1) + take(other_idx, int(rng.integers(1, 4))))
for _ in range(N_BENIGN_CLUSTERS):
    benign_clusters.append(take(np.arange(N_TRADERS), int(rng.integers(2, 4))))
for cl in insider_clusters + benign_clusters:
    shared = make_pan()
    for ti in cl:
        pans[ti] = shared

# --------------------------------------------------------------------------
# 4. TRADE-SIZE PROCESS + PROFILE BASELINES FROM A SEPARATE HISTORY PERIOD
# --------------------------------------------------------------------------
def gm_vec(sigma, n):
    m = rng.lognormal(0.0, sigma, n)
    big = rng.random(n) < P_LEGIT_LARGE
    m[big] = rng.uniform(2.0, 6.0, int(big.sum()))
    return m

def gm_one(sigma):
    return float(gm_vec(sigma, 1)[0])

def dfac(ti, day):
    return float(np.exp(size_drift[ti] * (day - mon_first) / 252.0))

n_insider = int(round(N_TRADES * INSIDER_FRACTION))
n_gen     = N_TRADES - n_insider
n_mon_days = mon_last - mon_first + 1
hist_days  = mon_first
rate   = (n_gen * lam / lam.sum()) / n_mon_days
hist_n = np.maximum(5, rng.poisson(rate * hist_days))

avg_q, std_q, avg_v = np.zeros(N_TRADERS), np.zeros(N_TRADERS), np.zeros(N_TRADERS)
for ti in range(N_TRADERS):
    n = int(hist_n[ti])
    secs = np.where(rng.random(n) < 0.85, rng.choice(watch[ti], n), rng.integers(0, n_sec, n))
    days = rng.integers(0, hist_days, n)
    q = np.maximum(1, np.round(base_qty[ti] * gm_vec(size_sig[ti], n)))
    avg_q[ti], std_q[ti], avg_v[ti] = q.mean(), q.std(ddof=1), (q * P[secs, days]).mean()

profile = pd.DataFrame({
    "trader_id": trader_ids, "role_class": role, "department": department,
    "preclear_status": preclear, "associated_pan": pans,
    "avg_trade_quantity": np.maximum(1, np.round(avg_q)).astype(int),
    "avg_trade_value": np.round(avg_v, 2),
    "avg_daily_trades": np.round(hist_n / hist_days, 4),
    "std_trade_quantity": np.round(std_q, 1),
})

# --------------------------------------------------------------------------
# 5. TRADE GENERATION  (unchanged logic)
# --------------------------------------------------------------------------
rows = []
def add_trade(ti, sec, day, direction, qty, true_insider, inj_type="genuine", group_id=None, ev=None):
    minute = int(rng.integers(555, 930))
    ts = bdays[day] + pd.Timedelta(minutes=minute)
    price = round(float(P[sec, day] * np.exp(rng.normal(0.0, INTRADAY_NOISE))), 2)
    d1, d3 = min(day + 1, n_days - 1), min(day + 3, n_days - 1)
    qty = max(1, int(qty))
    rows.append(dict(
        trader_i=ti, sec_i=sec, day_i=day, timestamp=ts, trade_type=direction, quantity=qty, price=price,
        total_value=round(qty * price, 2),
        price_return_1d=round(100 * (P[sec, d1] / P[sec, day] - 1), 3),
        price_return_3d=round(100 * (P[sec, d3] / P[sec, day] - 1), 3),
        true_insider=true_insider, inj_type=inj_type, group_id=group_id,
        event_id=None if ev is None else ev["event_id"],
        event_ts=pd.NaT if ev is None else ev["event_date"],
        event_type=None if ev is None else ev["event_type"]))

n_group_target = int(round(n_insider * GROUP_SHARE))
n_group, gid = 0, 0
while n_group < n_group_target:
    cl = insider_clusters[int(rng.integers(len(insider_clusters)))]
    ev = events.loc[int(rng.integers(len(events)))]
    cands = days_before_event(ev["event_date"], range(3, 11))
    if not cands:
        continue
    base = int(rng.choice(cands))
    direction = "BUY" if rng.random() < ev["buy_prob"] else "SELL"
    gid += 1
    gname = f"GRP{gid:04d}"
    for ti in cl:
        for _ in range(int(rng.integers(1, 3))):
            day = min(base + int(rng.integers(0, 3)), int(ev["day_i"]) - 1)
            qty = round(avg_q[ti] * dfac(ti, day) * rng.uniform(1.2, 3.0))
            add_trade(ti, int(ev["sec_i"]), day, direction, qty, 1, "coordinated_group", gname, ev)
            n_group += 1

DIRECTION_RATIONALE = {"BUY": "Accumulation ahead of anticipated positive corporate action",
                       "SELL": "Disposal ahead of anticipated adverse/dilutive corporate action"}
n_solo, made = n_insider - n_group, 0
while made < n_solo:
    ti = int(rng.choice(prone_idx)) if rng.random() < 0.65 else int(rng.choice(other_idx))
    if rng.random() < 0.30:
        sec = int(rng.choice(watch[ti]))
        ev = events.loc[int(rng.choice(events_by_sec[sec]))]
    else:
        ev = events.loc[int(rng.integers(len(events)))]
    u = rng.random()
    offsets = [1, 2, 3] if u < 0.55 else (list(range(4, 8)) if u < 0.85 else list(range(8, 15)))
    cands = days_before_event(ev["event_date"], offsets)
    if not cands:
        continue
    day = int(rng.choice(cands))
    direction = "BUY" if rng.random() < ev["buy_prob"] else "SELL"
    v = rng.random()
    if v < 0.40:   mult, kind = rng.uniform(1.3, 2.0), "solo_subtle"
    elif v < 0.80: mult, kind = rng.uniform(2.0, 4.0), "solo_moderate"
    else:          mult, kind = rng.uniform(4.0, 8.0), "solo_obvious"
    add_trade(ti, int(ev["sec_i"]), day, direction, round(avg_q[ti] * dfac(ti, day) * mult), 1, kind, None, ev)
    made += 1

draw = rng.choice(N_TRADERS, size=n_gen, p=lam / lam.sum())
g_count = 0
for ti in draw:
    if g_count >= n_gen:
        break
    ti = int(ti)
    u, ev = rng.random(), None
    if u < P_GENUINE_PREWINDOW:
        e = events.loc[int(rng.integers(len(events)))]
        c = days_before_event(e["event_date"], [1, 2, 3])
        if c:
            ev, sec, day = e, int(e["sec_i"]), int(rng.choice(c))
    elif u < P_GENUINE_PREWINDOW + P_GENUINE_POSTEVENT:
        e = events.loc[int(rng.integers(len(events)))]
        d = int(e["day_i"]) + int(rng.integers(0, 6))
        if mon_first <= d <= mon_last:
            ev, sec, day = e, int(e["sec_i"]), d
    if ev is None:
        sec = int(rng.choice(watch[ti])) if rng.random() < 0.70 else int(rng.integers(n_sec))
        day = int(rng.integers(mon_first, mon_last + 1))
    direction = "BUY" if rng.random() < 0.5 else "SELL"
    qty = max(1, int(round(base_qty[ti] * dfac(ti, day) * gm_one(size_sig[ti]))))
    add_trade(ti, sec, day, direction, qty, 0)
    g_count += 1
    if rng.random() < P_BURST:
        d = day
        for _ in range(int(rng.integers(1, 4))):
            d += int(rng.integers(1, 5))
            if d > mon_last or g_count >= n_gen:
                break
            add_trade(ti, sec, d, direction, round(qty * rng.uniform(0.6, 1.4)), 0)
            g_count += 1

trade = pd.DataFrame(rows).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
trade.insert(0, "trade_id", [f"TRX{str(i).zfill(6)}" for i in range(1, len(trade) + 1)])
trade["trader_id"]    = [trader_ids[i] for i in trade["trader_i"]]
trade["isin"]         = securities["isin"].values[trade["sec_i"].values]
trade["company_name"] = securities["company_name"].values[trade["sec_i"].values]

true_lab = trade["true_insider"].values
obs = true_lab.copy()
ins_i, gen_i = np.where(true_lab == 1)[0], np.where(true_lab == 0)[0]
obs[ins_i[rng.random(len(ins_i)) < LABEL_FN_RATE]] = 0
obs[gen_i[rng.random(len(gen_i)) < LABEL_FP_RATE]] = 1
trade["is_insider"] = obs

# --------------------------------------------------------------------------
# 6. DERIVED FEATURES (point-in-time where possible)
# --------------------------------------------------------------------------
trade = trade.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
pr = profile.set_index("trader_id")

mu = pr.loc[trade["trader_id"], "avg_trade_quantity"].values.astype(float)
sd = np.maximum(pr.loc[trade["trader_id"], "std_trade_quantity"].values, 0.25 * mu)
trade["volume_zscore"] = np.round((trade["quantity"].values - mu) / sd, 3)

freq = np.zeros(len(trade), dtype=int)
for _, idx in trade.groupby("trader_id").indices.items():
    ts = trade["timestamp"].values[idx]
    freq[idx] = np.searchsorted(ts, ts, side="right") - np.searchsorted(ts, ts - np.timedelta64(7, "D"), side="left")
trade["trade_frequency_7d"] = freq

pct = np.full(len(trade), 50.0)
for _, idx in trade.groupby("isin").indices.items():
    q = trade["quantity"].values[idx]
    for k in range(4, len(idx)):
        pct[idx[k]] = 100.0 * ((q[:k] <= q[k]).sum() + 1) / (k + 1)
trade["order_size_percentile"] = np.round(pct, 2)

ev_arr = {k: np.sort(g["event_date"].values) for k, g in events.groupby("isin")}
ttn = np.full(len(trade), np.nan)
for k, idx in trade.groupby("isin").indices.items():
    arr, ts = ev_arr[k], trade["timestamp"].values[idx]
    j = np.searchsorted(arr, ts, side="left")
    ok = j < len(arr)
    ttn[idx[ok]] = (arr[j[ok]] - ts[ok]) / np.timedelta64(1, "h")
trade["time_to_next_event"] = np.round(ttn, 2)
trade["is_pre_event_window"] = trade["time_to_next_event"].notna() & (trade["time_to_next_event"] <= PRE_EVENT_WINDOW_HOURS)

# ---- NEW: pre-trade returns (point-in-time, window ends the day BEFORE the trade) ----
sec_arr = trade["sec_i"].values
day_arr = trade["day_i"].values
for w in PRE_TRADE_RETURN_WINDOWS:
    end = np.clip(day_arr - 1, 0, n_days - 1)      # day before the trade
    start = np.clip(day_arr - 1 - w, 0, n_days - 1)  # w trading days earlier
    ret = 100.0 * (P[sec_arr, end] / P[sec_arr, start] - 1.0)
    ret[day_arr - 1 - w < 0] = np.nan               # not enough history yet -> unknown, not zero
    trade[f"pre_trade_return_{w}d"] = np.round(ret, 3)

trade = trade.sort_values(["trader_id", "timestamp"], kind="mergesort").reset_index(drop=True)
pretrade_cols = [f"pre_trade_return_{w}d" for w in PRE_TRADE_RETURN_WINDOWS]
trade_final = trade[["trade_id", "trader_id", "company_name", "isin", "timestamp", "trade_type", "quantity",
                     "price", "total_value", "volume_zscore", "trade_frequency_7d", "order_size_percentile",
                     "time_to_next_event", "is_pre_event_window"] + pretrade_cols +
                     ["price_return_1d", "price_return_3d", "is_insider"]].copy()

# --------------------------------------------------------------------------
# 7. ANSWER KEYS  (unchanged)
# --------------------------------------------------------------------------
gen_mask = trade["true_insider"] == 0
g = trade[gen_mask].groupby("isin")["quantity"].agg(["mean", "std"])
gm_all, gs_all = trade.loc[gen_mask, "quantity"].mean(), trade.loc[gen_mask, "quantity"].std()

keep = trade[(trade["true_insider"] == 1) | ((trade["true_insider"] == 0) & (trade["is_insider"] == 1))].copy()
h_mean = keep["isin"].map(g["mean"]).fillna(gm_all)
h_std  = keep["isin"].map(g["std"]).fillna(gs_all)
is_true = keep["true_insider"] == 1
meta = pd.DataFrame({
    "synthetic_id": [f"SYN{str(i).zfill(5)}" for i in range(1, len(keep) + 1)],
    "trade_id": keep["trade_id"].values, "event_id": keep["event_id"].values,
    "trader_id": keep["trader_id"].values, "isin": keep["isin"].values,
    "event_timestamp": keep["event_ts"].values, "trade_timestamp": keep["timestamp"].values,
    "timing_offset_hours": np.round((keep["event_ts"] - keep["timestamp"]).dt.total_seconds() / 3600, 2).values,
    "isin_historical_mean_qty": np.round(h_mean.values, 2),
    "isin_historical_std_qty": np.round(h_std.values, 2),
    "injected_quantity": keep["quantity"].values,
    "volume_multiplier_vs_isin_mean": np.round(keep["quantity"].values / h_mean.values, 2),
    "volume_multiplier_vs_trader_baseline": np.round(
        keep["quantity"].values / pr.loc[keep["trader_id"], "avg_trade_quantity"].values, 2),
    "direction": keep["trade_type"].values,
    "direction_rationale": np.where(is_true, keep["trade_type"].map(DIRECTION_RATIONALE),
                                    "Genuine trade recorded as suspicious (label noise)"),
    "injection_type": np.where(is_true, keep["inj_type"], "label_noise_fp"),
    "group_id": keep["group_id"].values,
    "ground_truth_label": np.where(is_true, "Suspicious", "Genuine"),
    "observed_label": keep["is_insider"].values,
})

grp = trade[trade["inj_type"] == "coordinated_group"]
groups = (grp.groupby("group_id")
             .agg(event_id=("event_id", "first"), isin=("isin", "first"), company_name=("company_name", "first"),
                  event_date=("event_ts", "first"), n_traders=("trader_id", "nunique"), n_trades=("trade_id", "size"),
                  first_trade=("timestamp", "min"), last_trade=("timestamp", "max"),
                  trader_ids=("trader_id", lambda s: ";".join(sorted(set(s)))),
                  trade_ids=("trade_id", lambda s: ";".join(s)))
             .reset_index())
groups.insert(4, "shared_pan", [pr.loc[t.split(";")[0], "associated_pan"] for t in groups["trader_ids"]])


# --------------------------------------------------------------------------
# 8. SAVE + SELF-CHECK
# --------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
profile.to_csv(f"{OUT_DIR}/profile.csv", index=False)
trade_final.to_csv(f"{OUT_DIR}/trade.csv", index=False)
meta.to_csv(f"{OUT_DIR}/synthetic_metadata.csv", index=False)
groups.to_csv(f"{OUT_DIR}/coordinated_groups.csv", index=False)

print("profile.csv             :", profile.shape)
print("trade.csv               :", trade_final.shape, "columns:", list(trade_final.columns))
print("synthetic_metadata.csv  :", meta.shape)
print("coordinated_groups.csv  :", groups.shape)
print("\nis_insider (recorded label):\n", trade_final["is_insider"].value_counts().to_string())
print("true insider trades     :", int(trade["true_insider"].sum()), "| by type:",
      trade.loc[trade["true_insider"] == 1, "inj_type"].value_counts().to_dict())
print("weekend trades          :", int((trade["timestamp"].dt.dayofweek >= 5).sum()))
w = trade.groupby("true_insider")["is_pre_event_window"].mean().round(3).to_dict()
print("share inside 72h window : genuine", w.get(0), "| insider", w.get(1))
for wdw in PRE_TRADE_RETURN_WINDOWS:
    col = f"pre_trade_return_{wdw}d"
    print(f"{col} coverage/NaN     :", trade_final[col].notna().sum(), "/", len(trade_final),
          "| mean genuine", round(trade_final.loc[trade_final["is_insider"] == 0, col].mean(), 4),
          "| mean insider", round(trade_final.loc[trade_final["is_insider"] == 1, col].mean(), 4))
