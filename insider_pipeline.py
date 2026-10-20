#!/usr/bin/env python3
"""
Insider-trading detection pipeline: Logistic Regression + Histogram Gradient Boosting (Soft Voting).
Matches insider_detection_FINAL.ipynb exact model architecture and features.

Run in VS Code / terminal:
    pip install pandas numpy scikit-learn joblib
    python insider_pipeline.py --trade trade.csv --profile profile.csv --out-dir outputs

Outputs (in --out-dir):
    insider_model.joblib  trained model bundle (model, features, threshold)
    metrics.json          held-out test metrics + overfitting check
    scored_trades.csv     EVERY trade: out-of-fold probability, flag, risk level, top reasons
    flagged_trades.csv    only the flagged trades -> input to the sentiment / reasoning layer

Import in backend / FastAPI:
    from insider_pipeline import score_trades
    scored = score_trades(new_trades_df, profile_df, "outputs/insider_model.joblib")
"""
import argparse, json, os, time, warnings
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
warnings.filterwarnings("ignore")

SEED, BETA, TEST_SIZE, N_FOLDS = 42, 2.0, 0.20, 5      # BETA>1 favours recall; 1.0 = balanced precision/recall
HIGH_RISK_PROB = 0.90
REQ_TRADE = ["trade_id", "trader_id", "isin", "timestamp", "trade_type", "quantity", "total_value", "volume_zscore",
             "trade_frequency_7d", "order_size_percentile", "time_to_next_event",
             "pre_trade_return_1d", "pre_trade_return_3d", "pre_trade_return_5d"]
REQ_PROFILE = ["trader_id", "role_class", "preclear_status", "associated_pan", "avg_trade_quantity", "avg_trade_value"]
ID_COLS = ["trade_id", "trader_id", "company_name", "isin", "timestamp", "trade_type", "quantity", "role_class"]


# ---------------------------------------------------------------- features (from FINAL notebook)
def build_features(trade, profile, use_event_timing=True, use_linked_accounts=True):
    """Point-in-time / retrospective features only: no labels, no forward-looking prices."""
    trade = trade.copy()
    trade["timestamp"] = pd.to_datetime(trade["timestamp"])
    miss = [c for c in REQ_TRADE if c not in trade.columns] + [c for c in REQ_PROFILE if c not in profile.columns]
    if miss:
        raise ValueError(f"missing columns: {miss}")

    df = trade.merge(profile[REQ_PROFILE], on="trader_id", how="left", validate="many_to_one").reset_index(drop=True)
    if df["role_class"].isna().any():
        raise ValueError("some trades belong to traders missing from the profile file")

    df["log_qty_ratio"]  = np.log(df.quantity / df.avg_trade_quantity).clip(-3, 4)      # size vs trader's own baseline
    df["log_val_ratio"]  = np.log(df.total_value / df.avg_trade_value).clip(-4, 5)
    df["vol_zscore"]     = df.volume_zscore.clip(-4, 6)
    df["order_size_pct"] = df.order_size_percentile
    df["freq_7d"]        = df.trade_frequency_7d
    df["preclear_req"]   = (df.preclear_status == "Required").astype(int)

    for w in (1, 3, 5):                                                                  # momentum going INTO the trade
        df[f"pre_ret_{w}d"] = df[f"pre_trade_return_{w}d"].clip(-15, 15)

    role_list = ["Connected Person", "Designated Person", "Director", "KMP", "Promoter"]
    for r in role_list:
        df[f"role_{r}"] = (df.role_class == r).astype(int)

    feats = (["log_qty_ratio", "log_val_ratio", "vol_zscore", "order_size_pct", "freq_7d", "preclear_req",
              "pre_ret_1d", "pre_ret_3d", "pre_ret_5d"] + [f"role_{r}" for r in role_list])

    if use_event_timing:      # non-linear: insiders cluster in the last days/weeks before an announcement
        df["log_hours_to_event"] = np.log1p(df.time_to_next_event.fillna(720).clip(upper=720))
        df["evt_within_3d"]  = (df.time_to_next_event <= 72).astype(int)
        df["evt_within_14d"] = (df.time_to_next_event <= 14 * 24).astype(int)
        feats += ["log_hours_to_event", "evt_within_3d", "evt_within_14d"]

    if use_linked_accounts:   # other accounts with the same PAN, same stock, same direction, within +-7 days
        out = np.zeros(len(df))
        shared = df.groupby("associated_pan").trader_id.transform("nunique") > 1
        for _, g in df[shared].groupby(["associated_pan", "isin", "trade_type"]):
            ts, tr, pos = g.timestamp.values, g.trader_id.values, g.index.values
            for i in range(len(g)):
                out[pos[i]] = ((np.abs((ts - ts[i]) / np.timedelta64(1, "D")) <= 7) & (tr != tr[i])).sum()
        df["linked_trades_7d"] = np.log1p(out)
        feats.append("linked_trades_7d")

    return df, feats


# ---------------------------------------------------------------- model (from FINAL notebook)
def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.03, class_weight="balanced", max_iter=5000))

def make_hgb():
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                           min_samples_leaf=40, class_weight="balanced", random_state=SEED)

def make_model():
    return VotingClassifier([("lr", make_lr()), ("hgb", make_hgb())], voting="soft")

def oof_proba(Xa, ya, ga, n_folds=N_FOLDS):
    oof = np.zeros(len(ya))
    for a, b in GroupKFold(n_folds).split(Xa, ya, ga):
        oof[b] = make_model().fit(Xa.iloc[a], ya[a]).predict_proba(Xa.iloc[b])[:, 1]
    return oof

def pick_threshold(yy, pr):    # centre of the plateau where F-beta is within 0.01 of its best (stable, not noise-fitted)
    grid = np.arange(0.05, 0.96, 0.01)
    f = np.array([fbeta_score(yy, (pr >= t).astype(int), beta=BETA, zero_division=0) for t in grid])
    return float(np.median(grid[f >= f.max() - 0.01]))

def metrics(yy, pr, thr):
    pred = (pr >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yy, pred, labels=[0, 1]).ravel()
    return dict(precision=precision_score(yy, pred), recall=recall_score(yy, pred), f1=f1_score(yy, pred),
                accuracy=accuracy_score(yy, pred), roc_auc=roc_auc_score(yy, pr), pr_auc=average_precision_score(yy, pr),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


# ---------------------------------------------------------------- scoring / reasons
def top_reasons(model, X, k=3):
    """Top-k features pushing each trade toward 'insider' (from the explainable Logistic Regression component)."""
    lr = model.named_estimators_["lr"]
    contrib = lr.named_steps["standardscaler"].transform(X) * lr.named_steps["logisticregression"].coef_[0]
    names, idx = np.array(X.columns), np.argsort(-contrib, axis=1)[:, :k]
    return [", ".join(names[i][contrib[r, i] > 0]) for r, i in enumerate(idx)]

def assemble(df, feats, prob, thr, model):
    out = df[[c for c in ID_COLS if c in df.columns]].copy()
    out["prob_insider"] = np.round(prob, 4)
    out["risk_score"] = np.round(100 * prob).astype(int)
    out["flag"] = (prob >= thr).astype(int)
    out["risk_level"] = np.where(prob >= HIGH_RISK_PROB, "High", np.where(prob >= thr, "Medium", "Low"))
    out["top_reasons"] = top_reasons(model, df[feats])
    if "is_insider" in df.columns:
        out["is_insider"] = df["is_insider"].values
    return out

def score_trades(trade, profile, bundle_path):
    """For the backend: score a batch of trades with the saved model bundle."""
    b = joblib.load(bundle_path)
    df, feats = build_features(trade, profile, b.get("use_event_timing", True), b.get("use_linked_accounts", True))
    assert feats == b["features"], "feature mismatch with the trained model"
    return assemble(df, feats, b["model"].predict_proba(df[feats])[:, 1], b["threshold"], b["model"])


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trade", default="data/trade.csv"); ap.add_argument("--profile", default="data/profile.csv")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--no-event-timing", action="store_true", help="exclude time-to-announcement features")
    ap.add_argument("--no-linked-accounts", action="store_true", help="drop linked-account feature (real-time mode)")
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True); t0 = time.time()

    # Check for data files in root or data/
    trade_path = a.trade if os.path.exists(a.trade) else ("trade.csv" if os.path.exists("trade.csv") else a.trade)
    profile_path = a.profile if os.path.exists(a.profile) else ("profile.csv" if os.path.exists("profile.csv") else a.profile)

    trade, profile = pd.read_csv(trade_path, parse_dates=["timestamp"]), pd.read_csv(profile_path)
    if "is_insider" not in trade.columns:
        raise ValueError("trade file needs an is_insider column for training")

    df, feats = build_features(trade, profile, not a.no_event_timing, not a.no_linked_accounts)
    X, y, g = df[feats], df["is_insider"].values, df["trader_id"].values
    print(f"{len(df)} trades | {len(feats)} features | insider share {y.mean():.1%}")

    # 1) honest hold-out estimate (unseen traders)
    tr, te = next(GroupShuffleSplit(1, test_size=TEST_SIZE, random_state=SEED).split(X, y, g))
    assert not set(g[tr]) & set(g[te])
    thr_ho = pick_threshold(y[tr], oof_proba(X.iloc[tr], y[tr], g[tr]))
    m_ho = make_model().fit(X.iloc[tr], y[tr])
    test, train = metrics(y[te], m_ho.predict_proba(X.iloc[te])[:, 1], thr_ho), metrics(y[tr], m_ho.predict_proba(X.iloc[tr])[:, 1], thr_ho)
    gap = train["pr_auc"] - test["pr_auc"]
    print("\nHeld-out TEST (traders never seen in training):")
    print("  precision %.3f | recall %.3f | F1 %.3f | accuracy %.3f | ROC-AUC %.3f | PR-AUC %.3f" %
          tuple(test[k] for k in ["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"]))
    print("  TP %d  FP %d  FN %d  TN %d" % (test["tp"], test["fp"], test["fn"], test["tn"]))
    print("  overfitting check: PR-AUC train %.3f vs test %.3f (gap %.3f -> %s)" %
          (train["pr_auc"], test["pr_auc"], gap, "no material overfitting" if gap < 0.03 else "WARNING"))

    # 2) deployment model: threshold from out-of-fold predictions on ALL trades; model refit on all trades
    oof_all = oof_proba(X, y, g); thr = pick_threshold(y, oof_all)
    final = make_model().fit(X, y)
    joblib.dump({"model": final, "features": feats, "threshold": thr, "use_event_timing": not a.no_event_timing,
                 "use_linked_accounts": not a.no_linked_accounts}, f"{a.out_dir}/insider_model.joblib")

    # 3) every trade scored with OUT-OF-FOLD probabilities (no trade is scored by a model that trained on it)
    scored = assemble(df, feats, oof_all, thr, final)
    scored.to_csv(f"{a.out_dir}/scored_trades.csv", index=False)
    flagged = scored[scored.flag == 1].sort_values("prob_insider", ascending=False)
    flagged.to_csv(f"{a.out_dir}/flagged_trades.csv", index=False)
    with open(f"{a.out_dir}/metrics.json", "w") as f:
        json.dump({"test": test, "train": train, "pr_auc_gap": gap, "threshold_deploy": thr, "features": feats}, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, np.floating) else int(o))
    print(f"\nSaved to {a.out_dir}/: insider_model.joblib, metrics.json, scored_trades.csv ({len(scored)} rows), "
          f"flagged_trades.csv ({len(flagged)} rows -> input for the sentiment layer)")
    print("Risk levels:", scored.risk_level.value_counts().to_dict(), "| done in %.0f s" % (time.time() - t0))

if __name__ == "__main__":
    main()