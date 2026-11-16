#!/usr/bin/env python3
"""
Sentiment / news refinement stage (v3) for insider-trading detection.

What changed vs v2
------------------
v2 de-flagged trades with hand-tuned penalties (-0.30, -0.15 ...). Those rules fire on almost
nothing that is actually a false positive (TN was 11 of 899 FPs), because the "far from event"
signal is already used by the ML stage, so it carries no new information.

v3 replaces the hand-tuned rules with a small LEARNED second-stage model trained ONLY on the
ML-flagged trades. It sees the ML probability plus new news evidence (FinBERT sentiment, sentiment/
trade-direction alignment, lead-lag vs. article times, article counts, event type) and the trade
context. The de-flag threshold is then chosen from out-of-fold predictions so that recall on the
flagged set stays >= RECALL_FLOOR (default 99.9%). Recall is protected by construction; precision
rises by whatever the data can support.

Honest expectation: with recall pinned at 99.9% the gain is modest (about +1 to +2 points of
precision on my reproduction). Raising precision much further needs new information, not new rules.
"""
import os, re, warnings, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
warnings.filterwarnings("ignore")

EVENTS_PATH = os.path.join(os.path.dirname(__file__), "data", "events_real_master.csv")
RECALL_FLOOR = 0.999      # minimum recall (on ML-flagged true insiders) the refinement must keep
SEED, N_FOLDS = 42, 5

# ---------------------------------------------------------------- FinBERT (Base / LoRA Fine-Tuned) + Guardrails
_model_instance = None
_tokenizer_instance = None
_is_lora_loaded = False
LORA_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "finbert_lora")

def anonymize_text(text: str, company_name: str = None, symbol: str = None) -> str:
    """Input Guardrail: Strips company names & symbols to prevent brand-prior bias."""
    if not isinstance(text, str) or not text.strip():
        return "corporate disclosure"
    cleaned = text
    if company_name and isinstance(company_name, str) and len(company_name.strip()) > 2:
        c_clean = company_name.strip()
        base_name = re.sub(r"\b(ltd|limited|corp|corporation|inc|industries)\b", "", c_clean, flags=re.IGNORECASE).strip()
        if len(base_name) > 2:
            cleaned = re.sub(re.escape(base_name), "[COMPANY]", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(re.escape(c_clean), "[COMPANY]", cleaned, flags=re.IGNORECASE)
    if symbol and isinstance(symbol, str) and len(symbol.strip()) > 1:
        cleaned = re.sub(r"\b" + re.escape(symbol.strip()) + r"\b", "[COMPANY]", cleaned, flags=re.IGNORECASE)
    # Clean common corporate suffixes (Industries, Bank, Motors, etc.)
    cleaned = re.sub(r"\[COMPANY\]\s+(Industries|Enterprises|Holdings|Motors|Steel|Power|Bank|Services|Finance|Tech|Technologies)\b", "[COMPANY]", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(\[COMPANY\]\s*)+", "[COMPANY] ", cleaned)
    return cleaned.strip()

def _load_finbert():
    """Loads fine-tuned LoRA FinBERT if available, else standard ProsusAI/finbert."""
    global _model_instance, _tokenizer_instance, _is_lora_loaded
    if _model_instance is not None:
        return _model_instance, _tokenizer_instance, _is_lora_loaded

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    base_name = "ProsusAI/finbert"
    if os.path.exists(os.path.join(LORA_MODEL_DIR, "adapter_model.safetensors")) or \
       os.path.exists(os.path.join(LORA_MODEL_DIR, "adapter_model.bin")):
        try:
            from peft import PeftModel
            _tokenizer_instance = AutoTokenizer.from_pretrained(LORA_MODEL_DIR)
            base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=3)
            _model_instance = PeftModel.from_pretrained(base, LORA_MODEL_DIR)
            _model_instance.eval()
            _is_lora_loaded = True
            print(f"--> [FinBERT] Loaded FINE-TUNED LoRA model from {LORA_MODEL_DIR}")
            return _model_instance, _tokenizer_instance, _is_lora_loaded
        except Exception as e:
            print(f"--> [FinBERT] Warning: Failed loading LoRA weights ({e}), falling back to base.")

    _tokenizer_instance = AutoTokenizer.from_pretrained(base_name)
    _model_instance = AutoModelForSequenceClassification.from_pretrained(base_name)
    _model_instance.eval()
    _is_lora_loaded = False
    print("--> [FinBERT] Loaded BASE ProsusAI/finbert model.")
    return _model_instance, _tokenizer_instance, _is_lora_loaded

def _run_finbert_guardrailed_batch(items, batch_size=32):
    """
    Runs batch inference with 3 Guardrail stages:
      1. Input Guardrail: Entity anonymization (replaces company/symbol with [COMPANY]).
      2. Uncertainty Guardrail: Shannon entropy check (dampens ambiguous predictions).
      3. Calibration Guardrail: Discounts routine filings.
    """
    import torch
    model, tokenizer, is_lora = _load_finbert()
    results = []
    n_b = (len(items) + batch_size - 1) // batch_size

    # id2label mapping for ProsusAI/finbert: {0: 'positive', 1: 'negative', 2: 'neutral'}
    id2label = getattr(model.config, "id2label", {0: "positive", 1: "negative", 2: "neutral"})

    for k in range(0, len(items), batch_size):
        chunk = items[k:k + batch_size]
        cleaned_texts = []
        for it in chunk:
            c_text = anonymize_text(it.get("title", ""), it.get("company_name"), it.get("symbol"))
            cleaned_texts.append(c_text)

        enc = tokenizer(cleaned_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for i, it in enumerate(chunk):
            p = probs[i]
            # Probabilities
            p_pos = float(p[0]) if id2label[0] == "positive" else float(p[2])
            p_neg = float(p[1]) if id2label[1] == "negative" else float(p[0])
            p_neu = float(p[2]) if id2label[2] == "neutral" else float(p[1])

            best_idx = int(np.argmax(p))
            label = id2label[best_idx].lower()
            conf = float(np.max(p))

            # Guardrail 2: Shannon Entropy
            entropy = -float(np.sum(p * np.log(p + 1e-9)))
            # If entropy is high (> 0.95) or confidence < 0.55, neutralize
            if entropy > 0.95 or conf < 0.55:
                raw_val = 0.0
                gated = True
            else:
                raw_val = p_pos - p_neg
                gated = False

            # Guardrail 3: Event-type calibration
            ev_type = str(it.get("primary_event_type", "")).lower()
            routine_keywords = ["general meeting", "postal ballot", "change in address", "compliance", "intimation"]
            calibrated_val = raw_val
            if any(r in ev_type for r in routine_keywords):
                calibrated_val *= 0.25

            results.append({
                "label": label,
                "score": conf,
                "sentiment_value": calibrated_val,
                "entropy": entropy,
                "guardrail_gated": gated,
                "anonymized_title": cleaned_texts[i]
            })
        print(f"--> Guardrailed FinBERT batch {k // batch_size + 1}/{n_b} (LoRA: {is_lora})", flush=True)

    return results, is_lora

def _sent_val(label, score):
    return score if label == "positive" else (-score if label == "negative" else 0.0)

def _alignment(sent, trade_type):
    return sent if trade_type == "BUY" else (-sent if trade_type == "SELL" else 0.0)

# ---------------------------------------------------------------- per-trade news features
def _news_features(flagged, pre_events, all_events):
    """Lead-lag + recency features, vectorised per ISIN. Uses ALL articles (pre and post outcome)
    for timing, and FinBERT sentiment from the pre-event articles."""
    pub = pd.to_datetime(all_events["pub_ist"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    art = all_events.assign(_pub=pub).drop_duplicates("article_id")
    pre = pre_events.assign(_pub=pd.to_datetime(pre_events["pub_ist"], utc=True)
                            .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None))
    t_all = {k: g["_pub"].values for k, g in art.groupby("isin")}
    t_pre = {k: g[["_pub", "sentiment_value"]].reset_index(drop=True) for k, g in pre.groupby("isin")}
    ts = pd.to_datetime(flagged["timestamp"]).dt.tz_localize(None) if pd.to_datetime(flagged["timestamp"]).dt.tz is not None \
        else pd.to_datetime(flagged["timestamp"])
    rows = []
    for isin, trade_ts, tt in zip(flagged["isin"], ts, flagged["trade_type"]):
        a = t_all.get(isin)
        if a is None or len(a) == 0:
            rows.append([0, 0, np.nan, 0, 0, np.nan, np.nan, np.nan]); continue
        d = (a - np.datetime64(trade_ts)) / np.timedelta64(1, "h")
        before, after = d < 0, d >= 0
        p = t_pre.get(isin)
        if p is not None and len(p):
            dp = (p["_pub"].values - np.datetime64(trade_ts)) / np.timedelta64(1, "h")
            s_before = p.loc[dp < 0, "sentiment_value"].mean() if (dp < 0).any() else np.nan
            s_after = p.loc[dp >= 0, "sentiment_value"].mean() if (dp >= 0).any() else np.nan
            s_all = p["sentiment_value"].mean()
        else:
            s_before = s_after = s_all = np.nan
        rows.append([int(before.sum()), int(after.sum()),
                     float(-d[before].max()) if before.any() else np.nan,   # hours since latest earlier article
                     int((after & (d <= 48)).sum()), int((before & (d >= -48)).sum()),
                     _alignment(s_after, tt) if not np.isnan(s_after) else np.nan,   # sentiment of news AFTER trade vs direction
                     _alignment(s_before, tt) if not np.isnan(s_before) else np.nan,
                     _alignment(s_all, tt) if not np.isnan(s_all) else np.nan])
    cols = ["news_before_trade", "news_after_trade", "hours_since_closest_news", "news_next_48h",
            "news_prev_48h", "align_after", "align_before", "align_all"]
    for c, v in zip(cols, zip(*rows)):
        flagged[c] = v
    flagged["lead_lag_category"] = np.select(
        [(flagged.news_before_trade + flagged.news_after_trade) == 0,
         (flagged.news_before_trade > 0) & (flagged.news_after_trade == 0),
         (flagged.news_before_trade == 0) & (flagged.news_after_trade > 0)],
        ["no_articles", "trade_after_news", "trade_before_news"],
        default=np.where(flagged.news_after_trade > flagged.news_before_trade, "mixed_mostly_after", "mixed_mostly_before"))
    return flagged

TRADE_CTX = ["timestamp", "time_to_next_event", "volume_zscore", "order_size_percentile", "trade_frequency_7d",
             "pre_trade_return_1d", "pre_trade_return_3d", "pre_trade_return_5d", "quantity", "total_value"]

def _model_matrix(f):
    X = pd.DataFrame(index=f.index)
    X["prob_insider"] = f["prob_insider"]
    for c in ["time_to_next_event", "volume_zscore", "order_size_percentile", "trade_frequency_7d",
              "pre_trade_return_1d", "pre_trade_return_3d", "pre_trade_return_5d",
              "avg_sentiment", "alignment", "n_articles", "news_before_trade", "news_after_trade",
              "hours_since_closest_news", "news_next_48h", "news_prev_48h",
              "align_after", "align_before", "align_all", "avg_hours_from_outcome"]:
        X[c] = f[c] if c in f.columns else np.nan
    X["is_buy"] = (f["trade_type"] == "BUY").astype(int)
    ts = pd.to_datetime(f["timestamp"])
    X["hour"] = ts.dt.hour + ts.dt.minute / 60
    X["dow"] = ts.dt.dayofweek
    X["lead_lag_code"] = f["lead_lag_category"].astype("category").cat.codes
    X["event_type_code"] = f["dominant_event_type"].astype("category").cat.codes
    return X

def _make_stage2():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)

def _pick_threshold(y, p, floor=RECALL_FLOOR):
    """Highest cut-off that still keeps >= floor of the true insiders. Lowest-risk way to protect recall."""
    pos = np.sort(p[y == 1])
    allowed_miss = int(np.floor((1 - floor) * len(pos)))
    return float(pos[allowed_miss]) if len(pos) else 0.0

def _fallback_rule(f):
    """No labels available: very conservative rule, only de-flag trades with NO event nearby and all news already public."""
    far = f["time_to_next_event"].fillna(1e9) > 336
    pub = (f["lead_lag_category"] == "trade_after_news") & (f["hours_since_closest_news"] < 48)
    return np.where(far & pub & (f["prob_insider"] < 0.8), 0, 1)

# ---------------------------------------------------------------- main
def run_sentiment_analysis(flagged_trades, all_scored_trades, trade_data=None,
                           events_path=EVENTS_PATH, progress_callback=None):
    cb = progress_callback or (lambda *a: None)
    cb("Loading events data", 0.05)
    events = pd.read_csv(events_path)
    flagged_isins = set(flagged_trades["isin"].unique())
    matched = events[events["isin"].isin(flagged_isins)].copy()
    pre_events = matched[matched["phase"].isin(["pre_outcome", "pre_intimation"])] \
        .drop_duplicates("article_id").copy()
    if len(pre_events) == 0:
        return _empty_results(flagged_trades, "No pre-event articles found for flagged ISINs")

    cb(f"Running Guardrailed FinBERT on {len(pre_events)} articles", 0.15)
    records_in = pre_events.to_dict("records")
    sents, is_lora = _run_finbert_guardrailed_batch(records_in)
    pre_events["sentiment_label"] = [s["label"] for s in sents]
    pre_events["sentiment_score"] = [s["score"] for s in sents]
    pre_events["sentiment_value"] = [s["sentiment_value"] for s in sents]
    pre_events["entropy"] = [s["entropy"] for s in sents]
    pre_events["guardrail_gated"] = [s["guardrail_gated"] for s in sents]

    cb("Aggregating sentiment per ISIN", 0.55)
    isin_agg = pre_events.groupby("isin").agg(
        n_articles=("article_id", "nunique"), avg_sentiment=("sentiment_value", "mean"),
        pos_count=("sentiment_label", lambda x: (x == "positive").sum()),
        neg_count=("sentiment_label", lambda x: (x == "negative").sum()),
        neu_count=("sentiment_label", lambda x: (x == "neutral").sum()),
        dominant_event_type=("primary_event_type", lambda x: x.mode().iloc[0] if len(x) else "Unknown"),
        avg_hours_from_outcome=("hours_from_outcome", "mean")).reset_index()
    isin_agg["dominant_sentiment"] = isin_agg.apply(
        lambda r: "positive" if r.pos_count > max(r.neg_count, r.neu_count)
        else ("negative" if r.neg_count > max(r.pos_count, r.neu_count) else "neutral"), axis=1)

    flagged = flagged_trades.copy()
    # bring in trade context (timestamp, time_to_next_event, z-scores ...) if missing
    src = trade_data if trade_data is not None else all_scored_trades
    need = [c for c in TRADE_CTX if c not in flagged.columns and c in src.columns]
    if need and "trade_id" in src.columns:
        flagged = flagged.merge(src[["trade_id"] + need].drop_duplicates("trade_id"), on="trade_id", how="left")
    flagged = flagged.merge(isin_agg[["isin", "avg_sentiment", "dominant_sentiment", "n_articles",
                                      "dominant_event_type", "avg_hours_from_outcome"]], on="isin", how="left")
    flagged["avg_sentiment"] = flagged["avg_sentiment"].fillna(0.0)
    flagged["dominant_sentiment"] = flagged["dominant_sentiment"].fillna("neutral")
    flagged["dominant_event_type"] = flagged["dominant_event_type"].fillna("Unknown")
    flagged["n_articles"] = flagged["n_articles"].fillna(0).astype(int)
    flagged["alignment"] = [_alignment(s, t) for s, t in zip(flagged["avg_sentiment"], flagged["trade_type"])]

    cb("Lead-lag / news timing features", 0.70)
    flagged = _news_features(flagged, pre_events, events)

    # ground truth
    gt_col = "is_insider" if "is_insider" in all_scored_trades.columns else ("label" if "label" in all_scored_trades.columns else None)
    if "is_insider" not in flagged.columns and gt_col and "trade_id" in all_scored_trades.columns:
        gt = all_scored_trades[["trade_id", gt_col]].rename(columns={gt_col: "is_insider"}).drop_duplicates("trade_id")
        flagged = flagged.merge(gt, on="trade_id", how="left")
    flagged = flagged.reset_index(drop=True)
    have_labels = "is_insider" in flagged.columns and flagged["is_insider"].notna().all()

    cb("Second-stage refinement", 0.80)
    X = _model_matrix(flagged)
    if have_labels:
        y = flagged["is_insider"].astype(int).values
        groups = flagged["trader_id"].values
        oof = np.zeros(len(flagged))
        for a, b in GroupKFold(N_FOLDS).split(X, y, groups):     # unseen traders -> honest probabilities
            oof[b] = _make_stage2().fit(X.iloc[a], y[a]).predict_proba(X.iloc[b])[:, 1]
        thr = _pick_threshold(y, oof)
        flagged["sentiment_adj_prob"] = oof
        flagged["sentiment_flag"] = (oof >= thr).astype(int)
        print(f"--> stage-2 threshold {thr:.4f} (keeps >= {RECALL_FLOOR:.1%} of flagged insiders)")
    else:                                                          # production: model trained earlier would be loaded here
        flagged["sentiment_adj_prob"] = flagged["prob_insider"]
        flagged["sentiment_flag"] = _fallback_rule(flagged)
        print("--> no labels: using conservative fallback rule")

    print(f"--> kept {flagged.sentiment_flag.sum()} / {len(flagged)}; removed {(flagged.sentiment_flag == 0).sum()} likely FPs")

    cb("Computing metrics", 0.92)
    metrics = {}
    if have_labels:
        yt, yp = flagged["is_insider"].astype(int).values, flagged["sentiment_flag"].values
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        metrics = dict(accuracy=round(accuracy_score(yt, yp), 4), precision=round(precision_score(yt, yp, zero_division=0), 4),
                       recall=round(recall_score(yt, yp, zero_division=0), 4), f1=round(f1_score(yt, yp, zero_division=0), 4),
                       tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))
        if len(set(yt)) > 1:
            metrics["roc_auc"] = round(float(roc_auc_score(yt, flagged["sentiment_adj_prob"])), 4)
        print("--> refined:", metrics)

    isin_sentiments = [{k: (int(v) if isinstance(v, np.integer) else round(float(v), 4) if isinstance(v, np.floating) else v)
                        for k, v in r.items()} for r in isin_agg.to_dict("records")]
    show = [c for c in ["trade_id", "trader_id", "isin", "trade_type", "quantity", "prob_insider", "risk_level",
                        "avg_sentiment", "dominant_sentiment", "alignment", "lead_lag_category", "news_before_trade",
                        "news_after_trade", "hours_since_closest_news", "sentiment_adj_prob", "sentiment_flag",
                        "is_insider", "n_articles", "dominant_event_type"] if c in flagged.columns]
    refined = flagged[show].sort_values("sentiment_adj_prob", ascending=False)
    records = [{c: (None if pd.isna(v) else int(v) if isinstance(v, np.integer) else round(float(v), 4)
                    if isinstance(v, np.floating) else v) for c, v in r.items()} for r in refined.to_dict("records")]
    summary = dict(model_used="FinBERT (LoRA Fine-Tuned)" if is_lora else "FinBERT (Base ProsusAI + Guardrails)",
                   guardrails_active=True,
                   articles_gated_by_entropy=int(pre_events["guardrail_gated"].sum()) if "guardrail_gated" in pre_events else 0,
                   total_flagged_ml=len(flagged_trades), isins_matched_events=len(flagged_isins & set(matched["isin"].unique())),
                   pre_event_articles_analysed=len(pre_events), trades_kept_after_sentiment=int(flagged.sentiment_flag.sum()),
                   trades_removed_by_sentiment=int((flagged.sentiment_flag == 0).sum()),
                   lead_lag_breakdown={str(k): int(v) for k, v in flagged["lead_lag_category"].value_counts().items()})
    cb("Done", 1.0)
    return {"metrics": metrics, "isin_sentiments": isin_sentiments, "refined_trades": records, "summary": summary}

def _empty_results(flagged_trades, reason):
    return {"metrics": {}, "isin_sentiments": [], "refined_trades": [],
            "summary": {"total_flagged_ml": len(flagged_trades), "isins_matched_events": 0, "pre_event_articles_analysed": 0,
                        "trades_kept_after_sentiment": 0, "trades_removed_by_sentiment": 0, "note": reason}}