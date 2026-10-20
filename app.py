"""
FastAPI backend for the Insider Trading Detection portal.

Run:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /                  Serve the frontend
    POST /api/upload        Upload trade + profile CSVs, run ML pipeline, return metrics
    POST /api/sentiment     Run FinBERT sentiment analysis on flagged trades
"""

import io, json, os, time, warnings
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, fbeta_score, roc_auc_score,
                             average_precision_score, confusion_matrix)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Config (matches insider_pipeline.py)
# ──────────────────────────────────────────────────────────────
SEED = 42
BETA = 2.0
TEST_SIZE = 0.20
N_FOLDS = 5
HIGH_RISK_PROB = 0.90

REQ_TRADE = [
    "trade_id", "trader_id", "isin", "timestamp", "trade_type", "quantity",
    "total_value", "volume_zscore", "trade_frequency_7d", "order_size_percentile",
    "time_to_next_event", "pre_trade_return_1d", "pre_trade_return_3d", "pre_trade_return_5d",
]
REQ_PROFILE = [
    "trader_id", "role_class", "preclear_status", "associated_pan",
    "avg_trade_quantity", "avg_trade_value",
]
ID_COLS = [
    "trade_id", "trader_id", "company_name", "isin", "timestamp",
    "trade_type", "quantity", "role_class",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────
app = FastAPI(title="Insider Trading Detection Portal")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# In-memory state for the current session
_state = {
    "trade_df": None,
    "profile_df": None,
    "scored_trades": None,
    "flagged_trades": None,
    "ml_results": None,
    "sentiment_results": None,
}


# ──────────────────────────────────────────────────────────────
# Feature engineering (from insider_pipeline.py)
# ──────────────────────────────────────────────────────────────
def build_features(trade, profile, use_event_timing=True, use_linked_accounts=True):
    """Point-in-time / retrospective features only."""
    trade = trade.copy()
    trade["timestamp"] = pd.to_datetime(trade["timestamp"])
    df = trade.merge(
        profile[REQ_PROFILE], on="trader_id", how="left", validate="many_to_one"
    ).reset_index(drop=True)

    if df["role_class"].isna().any():
        raise ValueError("Some trades belong to traders missing from the profile file")

    df["log_qty_ratio"]  = np.log(df.quantity / df.avg_trade_quantity).clip(-3, 4)
    df["log_val_ratio"]  = np.log(df.total_value / df.avg_trade_value).clip(-4, 5)
    df["vol_zscore"]     = df.volume_zscore.clip(-4, 6)
    df["order_size_pct"] = df.order_size_percentile
    df["freq_7d"]        = df.trade_frequency_7d
    df["preclear_req"]   = (df.preclear_status == "Required").astype(int)

    for w in (1, 3, 5):
        df[f"pre_ret_{w}d"] = df[f"pre_trade_return_{w}d"].clip(-15, 15)

    role_list = ["Connected Person", "Designated Person", "Director", "KMP", "Promoter"]
    for r in role_list:
        df[f"role_{r}"] = (df.role_class == r).astype(int)

    feats = (
        ["log_qty_ratio", "log_val_ratio", "vol_zscore", "order_size_pct",
         "freq_7d", "preclear_req", "pre_ret_1d", "pre_ret_3d", "pre_ret_5d"]
        + [f"role_{r}" for r in role_list]
    )

    if use_event_timing:
        df["log_hours_to_event"] = np.log1p(df.time_to_next_event.fillna(720).clip(upper=720))
        df["evt_within_3d"]  = (df.time_to_next_event <= 72).astype(int)
        df["evt_within_14d"] = (df.time_to_next_event <= 14 * 24).astype(int)
        feats += ["log_hours_to_event", "evt_within_3d", "evt_within_14d"]

    if use_linked_accounts:
        out = np.zeros(len(df))
        shared = df.groupby("associated_pan").trader_id.transform("nunique") > 1
        for _, g in df[shared].groupby(["associated_pan", "isin", "trade_type"]):
            ts = g.timestamp.values
            tr = g.trader_id.values
            pos = g.index.values
            for i in range(len(g)):
                out[pos[i]] = (
                    (np.abs((ts - ts[i]) / np.timedelta64(1, "D")) <= 7)
                    & (tr != tr[i])
                ).sum()
        df["linked_trades_7d"] = np.log1p(out)
        feats.append("linked_trades_7d")

    return df, feats


# ──────────────────────────────────────────────────────────────
# Model builders (from insider_pipeline.py)
# ──────────────────────────────────────────────────────────────
def make_lr():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.03, class_weight="balanced", max_iter=5000),
    )

def make_hgb():
    return HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=200,
        l2_regularization=1.0, min_samples_leaf=40,
        class_weight="balanced", random_state=SEED,
    )

def make_model():
    return VotingClassifier(
        [("lr", make_lr()), ("hgb", make_hgb())], voting="soft"
    )

def oof_proba(Xa, ya, ga, n_folds=N_FOLDS):
    oof = np.zeros(len(ya))
    for a, b in GroupKFold(n_folds).split(Xa, ya, ga):
        oof[b] = make_model().fit(Xa.iloc[a], ya[a]).predict_proba(Xa.iloc[b])[:, 1]
    return oof

def pick_threshold(yy, pr):
    grid = np.arange(0.05, 0.96, 0.01)
    f = np.array([
        fbeta_score(yy, (pr >= t).astype(int), beta=BETA, zero_division=0)
        for t in grid
    ])
    return float(np.median(grid[f >= f.max() - 0.01]))

def compute_metrics(yy, pr, thr):
    pred = (pr >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yy, pred, labels=[0, 1]).ravel()
    return {
        "precision": round(float(precision_score(yy, pred)), 4),
        "recall":    round(float(recall_score(yy, pred)), 4),
        "f1":        round(float(f1_score(yy, pred)), 4),
        "accuracy":  round(float(accuracy_score(yy, pred)), 4),
        "roc_auc":   round(float(roc_auc_score(yy, pr)), 4),
        "pr_auc":    round(float(average_precision_score(yy, pr)), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ──────────────────────────────────────────────────────────────
# Scoring / reasons (from insider_pipeline.py)
# ──────────────────────────────────────────────────────────────
def top_reasons(model, X, k=3):
    lr = model.named_estimators_["lr"]
    contrib = (
        lr.named_steps["standardscaler"].transform(X)
        * lr.named_steps["logisticregression"].coef_[0]
    )
    names = np.array(X.columns)
    idx = np.argsort(-contrib, axis=1)[:, :k]
    return [", ".join(names[i][contrib[r, i] > 0]) for r, i in enumerate(idx)]

def assemble(df, feats, prob, thr, model):
    out = df[[c for c in ID_COLS if c in df.columns]].copy()
    out["prob_insider"] = np.round(prob, 4)
    out["risk_score"] = np.round(100 * prob).astype(int)
    out["flag"] = (prob >= thr).astype(int)
    out["risk_level"] = np.where(
        prob >= HIGH_RISK_PROB, "High",
        np.where(prob >= thr, "Medium", "Low"),
    )
    out["top_reasons"] = top_reasons(model, df[feats])
    if "is_insider" in df.columns:
        out["is_insider"] = df["is_insider"].values
    return out


# ──────────────────────────────────────────────────────────────
# Serve frontend
# ──────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join(BASE_DIR, "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ──────────────────────────────────────────────────────────────
# Upload + ML Pipeline
# ──────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_and_run(
    trade_file: UploadFile = File(...),
    profile_file: UploadFile = File(...),
):
    """Upload trade + profile CSVs, run the ML pipeline, return metrics."""
    t0 = time.time()

    # 1. Read uploads
    try:
        trade_bytes = await trade_file.read()
        profile_bytes = await profile_file.read()
        trade_df = pd.read_csv(io.BytesIO(trade_bytes))
        profile_df = pd.read_csv(io.BytesIO(profile_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV files: {e}")

    # 2. Validate columns (not filenames)
    missing_trade = [c for c in REQ_TRADE if c not in trade_df.columns]
    missing_profile = [c for c in REQ_PROFILE if c not in profile_df.columns]
    errors = []
    if missing_trade:
        errors.append(f"Trade file missing columns: {missing_trade}")
    if missing_profile:
        errors.append(f"Profile file missing columns: {missing_profile}")
    if "is_insider" not in trade_df.columns:
        errors.append("Trade file needs an 'is_insider' column for training/evaluation")
    if errors:
        raise HTTPException(status_code=422, detail=" | ".join(errors))

    # 3. Feature engineering (with event timing features enabled to match Colab notebook)
    try:
        df, feats = build_features(trade_df, profile_df, use_event_timing=True)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {e}")

    X = df[feats]
    y = df["is_insider"].values
    g = df["trader_id"].values

    # 4. Hold-out split (unseen traders)
    tr, te = next(GroupShuffleSplit(1, test_size=TEST_SIZE, random_state=SEED).split(X, y, g))
    assert not set(g[tr]) & set(g[te]), "Trader leakage between train and test"

    # 5. Out-of-fold threshold on train
    oof_train = oof_proba(X.iloc[tr], y[tr], g[tr])
    thr_ho = pick_threshold(y[tr], oof_train)

    # 6. Fit hold-out model
    m_ho = make_model().fit(X.iloc[tr], y[tr])
    test_prob = m_ho.predict_proba(X.iloc[te])[:, 1]
    train_prob = m_ho.predict_proba(X.iloc[tr])[:, 1]

    test_met = compute_metrics(y[te], test_prob, thr_ho)
    train_met = compute_metrics(y[tr], train_prob, thr_ho)
    gap = round(train_met["pr_auc"] - test_met["pr_auc"], 4)

    # 7. Deployment model (refit on all data)
    oof_all = oof_proba(X, y, g)
    thr = pick_threshold(y, oof_all)
    final_model = make_model().fit(X, y)

    # 8. Score all trades with OOF probabilities
    scored = assemble(df, feats, oof_all, thr, final_model)

    # Store in session state
    _state["trade_df"] = trade_df
    _state["profile_df"] = profile_df
    _state["scored_trades"] = scored
    _state["flagged_trades"] = scored[scored.flag == 1].sort_values("prob_insider", ascending=False)
    _state["sentiment_results"] = None

    # 9. Build response
    risk_levels = scored.risk_level.value_counts().to_dict()
    flagged = _state["flagged_trades"]

    flagged_records = _safe_records(flagged.head(500))

    response = {
        "test_metrics": test_met,
        "train_metrics": train_met,
        "pr_auc_gap": gap,
        "threshold": round(thr, 4),
        "total_trades": len(scored),
        "flagged_count": len(flagged),
        "risk_levels": {k: int(v) for k, v in risk_levels.items()},
        "flagged_trades": flagged_records,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    # Save ML outputs to disk for persistence / reporting
    out_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    scored.to_csv(os.path.join(out_dir, "scored_trades.csv"), index=False)
    flagged.to_csv(os.path.join(out_dir, "flagged_trades.csv"), index=False)
    with open(os.path.join(out_dir, "ml_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(response, f, indent=2)

    _state["ml_results"] = response
    return JSONResponse(content=response)


# ──────────────────────────────────────────────────────────────
# Sentiment Analysis
# ──────────────────────────────────────────────────────────────
@app.post("/api/sentiment")
async def run_sentiment():
    """Run FinBERT sentiment analysis on flagged trades and save results to outputs/."""
    if _state["flagged_trades"] is None:
        raise HTTPException(status_code=400, detail="Run the ML pipeline first before sentiment analysis")

    from sentiment_analysis import run_sentiment_analysis

    try:
        result = run_sentiment_analysis(
            flagged_trades=_state["flagged_trades"],
            all_scored_trades=_state["scored_trades"],
            trade_data=_state["trade_df"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {e}")

    _state["sentiment_results"] = result

    # Save outputs to disk for reporting / persistence
    out_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "sentiment_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if result.get("refined_trades"):
        pd.DataFrame(result["refined_trades"]).to_csv(os.path.join(out_dir, "sentiment_refined_trades.csv"), index=False)
    if result.get("isin_sentiments"):
        pd.DataFrame(result["isin_sentiments"]).to_csv(os.path.join(out_dir, "isin_sentiments.csv"), index=False)

    return JSONResponse(content=result)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _safe_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list of dicts, handling numpy types."""
    records = []
    for _, row in df.iterrows():
        rec = {}
        for c in df.columns:
            v = row[c]
            if isinstance(v, (np.integer,)):
                rec[c] = int(v)
            elif isinstance(v, (np.floating,)):
                rec[c] = round(float(v), 4)
            elif isinstance(v, pd.Timestamp):
                rec[c] = str(v)
            elif pd.isna(v) if not isinstance(v, str) else False:
                rec[c] = None
            else:
                rec[c] = v
            # Final NaN check
            if isinstance(rec[c], float) and np.isnan(rec[c]):
                rec[c] = None
        records.append(rec)
    return records


# ──────────────────────────────────────────────────────────────
# Regulatory Explainability & PDF Endpoints
# ──────────────────────────────────────────────────────────────
def _get_truly_insider_flagged_trades() -> list[dict]:
    """
    Get trades that are TRULY insider (is_insider == 1) AND FLAGGED by sentiment analysis (sentiment_flag == 1).
    Enriches each trade with complete trade, profile, and news article publication details.
    """
    refined_records = []
    if _state["sentiment_results"] and "refined_trades" in _state["sentiment_results"]:
        refined_records = _state["sentiment_results"]["refined_trades"]
    else:
        out_csv = os.path.join(BASE_DIR, "outputs", "sentiment_refined_trades.csv")
        if os.path.exists(out_csv):
            df = pd.read_csv(out_csv)
            refined_records = _safe_records(df)

    if not refined_records:
        return []

    # STRICT FILTER: is_insider == 1 AND sentiment_flag == 1
    truly_insider = []
    for t in refined_records:
        is_ins = int(t.get("is_insider") or 0)
        sent_flg = int(t.get("sentiment_flag") or 0)
        if is_ins == 1 and sent_flg == 1:
            truly_insider.append(t)

    if not truly_insider:
        return []

    df_ti = pd.DataFrame(truly_insider)

    # 1. Merge trade.csv data if trade fields are missing/null
    trade_csv_path = os.path.join(BASE_DIR, "data", "trade.csv")
    if _state["trade_df"] is not None:
        df_t = _state["trade_df"]
    elif os.path.exists(trade_csv_path):
        df_t = pd.read_csv(trade_csv_path)
    else:
        df_t = None

    if df_t is not None:
        t_cols_to_merge = [c for c in df_t.columns if c not in ["is_insider"]]
        for c in t_cols_to_merge:
            if c != "trade_id" and (c not in df_ti.columns or df_ti[c].isnull().all()):
                if c in df_ti.columns:
                    df_ti = df_ti.drop(columns=[c])
        cols_for_join = [c for c in t_cols_to_merge if c not in df_ti.columns or c == "trade_id"]
        if len(cols_for_join) > 1:
            df_ti = df_ti.merge(df_t[cols_for_join].drop_duplicates("trade_id"), on="trade_id", how="left")

    # 2. Merge profile.csv data
    profile_csv_path = os.path.join(BASE_DIR, "data", "profile.csv")
    if _state["profile_df"] is not None:
        df_p = _state["profile_df"]
    elif os.path.exists(profile_csv_path):
        df_p = pd.read_csv(profile_csv_path)
    else:
        df_p = None

    if df_p is not None:
        for c in df_p.columns:
            if c != "trader_id" and (c not in df_ti.columns or df_ti[c].isnull().all()):
                if c in df_ti.columns:
                    df_ti = df_ti.drop(columns=[c])
        cols_p_join = [c for c in df_p.columns if c not in df_ti.columns or c == "trader_id"]
        if len(cols_p_join) > 1 and "trader_id" in df_ti.columns:
            df_ti = df_ti.merge(df_p[cols_p_join].drop_duplicates("trader_id"), on="trader_id", how="left")

    # 3. Enrich with news headline & news publication date from events_real_master.csv
    events_csv_path = os.path.join(BASE_DIR, "data", "events_real_master.csv")
    if os.path.exists(events_csv_path):
        try:
            df_ev = pd.read_csv(events_csv_path)
            if "pub_ist" in df_ev.columns and "title" in df_ev.columns:
                df_ev["_pub_utc"] = pd.to_datetime(df_ev["pub_ist"], utc=True, errors="coerce")
                ev_by_isin = {isin_v: grp for isin_v, grp in df_ev.groupby("isin")}

                trade_ts_series = pd.to_datetime(df_ti["timestamp"], utc=True, errors="coerce") if "timestamp" in df_ti.columns else None

                headlines = []
                pub_dates = []

                for idx, row in df_ti.iterrows():
                    isin_v = row.get("isin")
                    grp = ev_by_isin.get(isin_v)
                    if grp is not None and not grp.empty and trade_ts_series is not None:
                        ts = trade_ts_series.loc[idx]
                        if pd.notna(ts):
                            diffs = (grp["_pub_utc"] - ts).abs()
                            best_i = diffs.idxmin()
                            best_row = grp.loc[best_i]
                            headlines.append(str(best_row.get("title", "N/A")))
                            pub_dates.append(str(best_row.get("pub_ist", "N/A")))
                        else:
                            best_row = grp.iloc[0]
                            headlines.append(str(best_row.get("title", "N/A")))
                            pub_dates.append(str(best_row.get("pub_ist", "N/A")))
                    elif grp is not None and not grp.empty:
                        best_row = grp.iloc[0]
                        headlines.append(str(best_row.get("title", "N/A")))
                        pub_dates.append(str(best_row.get("pub_ist", "N/A")))
                    else:
                        headlines.append("Corporate Earnings & Strategic Financial Announcement")
                        pub_dates.append("N/A")

                df_ti["news_headline"] = headlines
                df_ti["news_pub_date"] = pub_dates
        except Exception as e:
            print(f"News enrichment note: {e}")

    return _safe_records(df_ti)



@app.get("/api/explainability/trades")
async def get_explainability_trades():
    """Returns only trades that are truly insider (is_insider=1) AND sentiment-flagged (sentiment_flag=1)."""
    trades = _get_truly_insider_flagged_trades()
    return JSONResponse(content={
        "total_truly_insider_flagged": len(trades),
        "trades": trades
    })


@app.get("/api/explainability/trade/{trade_id}")
async def get_trade_explainability(trade_id: str):
    """Get regulatory explainability and violated SEBI sections for a specific trade."""
    trades = _get_truly_insider_flagged_trades()
    target = next((t for t in trades if str(t.get("trade_id")) == str(trade_id)), None)

    if not target:
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found among truly insider sentiment-flagged trades.")

    from regulatory_reasoning import analyze_trade_regulations
    violations = analyze_trade_regulations(target)

    return JSONResponse(content={
        "trade": target,
        "violations": violations
    })


@app.get("/api/explainability/pdf/{trade_id}")
async def export_trade_pdf(trade_id: str):
    """Generate and download PDF compliance report for a specific trade."""
    trades = _get_truly_insider_flagged_trades()
    target = next((t for t in trades if str(t.get("trade_id")) == str(trade_id)), None)

    if not target:
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found among truly insider sentiment-flagged trades.")

    from regulatory_reasoning import analyze_trade_regulations, generate_trade_pdf
    violations = analyze_trade_regulations(target)
    pdf_bytes = generate_trade_pdf(target, violations)

    filename = f"Insider_Trading_Report_{trade_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


# ──────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

