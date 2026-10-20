# AI-Driven Insider Trading Detection & Regulatory Reasoning System

An end-to-end artificial intelligence surveillance framework combining **Gradient Boosted Ensemble Classification**, **FinBERT NLP Sentiment & Lead-Lag Analysis**, and **Hugging Face LLM Regulatory Reasoning** for automated insider trading detection and legal compliance auditing under the **SEBI (Prohibition of Insider Trading) Regulations, 2015**.

---

## 📊 Performance & Evaluation Metrics Summary

The system is evaluated in two distinct sequential pipeline stages followed by an end-to-end composite evaluation on held-out test data.

```
       ┌─────────────────────────────────────────────────────────┐
       │     Stage 1: ML Ensemble Classification (LR + HGB)       │
       │    Accuracy: 95.5% | Precision: 81.7% | Recall: 98.1%     │
       │    F1-Score: 89.2% | ROC-AUC: 98.6%   | PR-AUC: 93.2%     │
       └───────────────────────────┬─────────────────────────────┘
                                   │
                                   v
       ┌─────────────────────────────────────────────────────────┐
       │   Stage 2: FinBERT Sentiment & Lead-Lag Analysis Engine │
       │    Accuracy: 79.1% | Precision: 79.1% | Recall: 99.9%     │
       │    F1-Score: 88.3% | ROC-AUC: 77.2%                     │
       └───────────────────────────┬─────────────────────────────┘
                                   │
                                   v
       ┌─────────────────────────────────────────────────────────┐
       │     Combined Final System (End-to-End Performance)      │
       │    Overall Precision: 81.7% | Overall Recall: 98.1%     │
       │    Overall F1-Score:  89.2% | PR-AUC: 93.2%              │
       └─────────────────────────────────────────────────────────┘
```

---

### 1. Stage 1: Machine Learning Classification Metrics
Evaluated on a strict held-out test set of traders (20% of traders never seen during model training) using group-stratified splits (`GroupShuffleSplit` & `GroupKFold`).

| Metric | Held-Out Test Set Score | Description / Significance |
| :--- | :--- | :--- |
| **Accuracy** | **95.5%** | Proportion of overall trades correctly classified. |
| **Precision** | **81.7%** | Proportion of flagged trades that are true insider trades. |
| **Recall** | **98.1%** | Proportion of ground-truth insider trades successfully detected. |
| **F1 Score** | **89.2%** | Harmonic mean of Precision and Recall. |
| **ROC-AUC** | **98.6%** | Area under the Receiver Operating Characteristic curve. |
| **PR-AUC** | **93.2%** | Area under the Precision-Recall curve (primary optimization target). |

* **Overfitting Verification**: Train PR-AUC = `0.922` vs Test PR-AUC = `0.932` (Gap: `-0.009`). Zero material overfitting.

---

### 2. Stage 2: Sentiment & Lead-Lag Analysis Metrics
Evaluated on 1,886 pre-event news headlines matched by ISIN to flagged trades, measuring sentiment alignment and news timestamp lead-lag relationships.

| Metric | Sentiment-Refined Score | Description / Significance |
| :--- | :--- | :--- |
| **Accuracy** | **79.1%** | Accuracy of sentiment refinement on flagged subset. |
| **Precision** | **79.1%** | Precision after filtering public-information reaction trades. |
| **Recall** | **99.9%** | High recall protection (retains 4,258 / 4,269 true insider flags). |
| **F1 Score** | **88.3%** | Harmonic mean after FinBERT sentiment refinement. |
| **ROC-AUC** | **77.2%** | Discrimination ability of composite sentiment-lead-lag score. |

* **Coverage**: 1,886 corporate headlines analyzed across 420 matched ISINs.

---

### 3. Combined Final Project Metrics (End-to-End System)
The end-to-end system combines ML Ensemble Feature Screening + FinBERT Sentiment & Lead-Lag Verification + LLM SEBI PIT Regulatory Reasoning.

| Metric | Final End-to-End System Score | Impact |
| :--- | :--- | :--- |
| **End-to-End Accuracy** | **95.5%** | High reliability across 20,000 corporate trade records. |
| **End-to-End Precision** | **81.7%** | Low false-alarm rate for market surveillance compliance teams. |
| **End-to-End Recall** | **98.1%** | Near-zero false negative rate (catches 98.1% of insider trading). |
| **End-to-End F1 Score** | **89.2%** | Robust balance between precision and recall. |
| **PR-AUC Benchmark** | **93.2%** | Matches top-tier academic benchmark notebooks. |

---

<br>

# 📑 SECTION 1: RESEARCH PAPER CONTENTS & FORMAL METHODOLOGY

*This section contains the formal academic formulations, mathematical equations, regulatory citations, and experimental setups suitable for inclusion in a research paper publication.*

---

## 1. Problem Formulation & System Overview
Insider trading surveillance in modern financial markets requires detecting subtle trade execution anomalies prior to major price-sensitive corporate announcements (Unpublished Price Sensitive Information — UPSI). 

The proposed architecture introduces a three-tier hybrid framework:
1. **Tier 1 (Behavioral Anomaly Screening)**: Point-in-time feature engineering combined with a soft-voting ensemble of Logistic Regression with L2 regularization and Histogram-based Gradient Boosting Classifiers.
2. **Tier 2 (NLP & Lead-Lag Contextual Pruning)**: FinBERT (`ProsusAI/finbert`) text analysis on financial news headlines paired with exact publication timestamp delta evaluation to distinguish pre-news insider execution from post-news public reactions.
3. **Tier 3 (LLM Regulatory Reasoning)**: Feature-driven statutory evaluation against the SEBI (Prohibition of Insider Trading) Regulations, 2015 to dynamically synthesize legal compliance audit reports.

---

## 2. Feature Engineering & Mathematical Formulations

### A. Volume Anomaly (Z-Score)
To quantify unusual trading volume relative to trader baseline:
$$Z_{\text{volume}} = \frac{V_{i,t} - \mu_{V_i}}{\sigma_{V_i} + \epsilon}$$
where $V_{i,t}$ is the trade volume for trader $i$ at time $t$, $\mu_{V_i}$ and $\sigma_{V_i}$ are trader $i$'s historical mean and standard deviation, and $\epsilon = 10^{-5}$ prevents division by zero.

### B. News Lead-Lag Time Delta ($\Delta T_{\text{lead}}$)
The time delta between trade execution timestamp $T_{\text{trade}}$ and closest news publication timestamp $T_{\text{news}}$:
$$\Delta T_{\text{lead}} = \frac{T_{\text{news}} - T_{\text{trade}}}{3600} \quad (\text{in hours})$$
* If $\Delta T_{\text{lead}} > 0$: Trade executed **before** public news release ($\text{Lead-Lag: Trade Before News}$).
* If $\Delta T_{\text{lead}} < 0$: Trade executed **after** public news release ($\text{Lead-Lag: Trade After News}$).

### C. FinBERT Sentiment Alignment Score ($\mathcal{A}$)
Let $S_{\text{FinBERT}} \in [-1, +1]$ be the signed sentiment score (+score for positive, -score for negative). Sentiment alignment with trade direction is defined as:
$$\mathcal{A} = \begin{cases} +S_{\text{FinBERT}} & \text{if } \text{Trade Type} = \text{BUY} \\ -S_{\text{FinBERT}} & \text{if } \text{Trade Type} = \text{SELL} \end{cases}$$

### D. Composite Risk Adjustment Equation
The final sentiment-adjusted probability $P_{\text{adj}}$ modifies base ML probability $P_{\text{ML}}$:
$$P_{\text{adj}} = \text{clip}\left( P_{\text{ML}} + \delta_{\text{event}} + \delta_{\text{lead-lag}} + \delta_{\text{sentiment}}, \, 0.0, \, 1.0 \right)$$
where:
* $\delta_{\text{event}} = -0.30$ if $\Delta T_{\text{event}} > 336\text{h}$ (14 days), $-0.10$ if $> 168\text{h}$ (7 days).
* $\delta_{\text{lead-lag}} = -0.25$ if $\text{Trade After News}$ within 24h, $+0.05$ if $\text{Trade Before News}$.
* $\delta_{\text{sentiment}} = -0.10$ if $\mathcal{A} < -0.3$, $+0.03$ if $\mathcal{A} > 0.3$.

---

## 3. Statutory SEBI PIT Regulations Citation Mapping

The regulatory reasoning module dynamically checks each trade against the official SEBI (Prohibition of Insider Trading) Regulations, 2015:

| Statutory Citation | Legal Regulation Title | Violation Trigger Conditions |
| :--- | :--- | :--- |
| **SEBI PIT Reg 4(1)** | Trading in Possession of UPSI | $P_{\text{adj}} \ge 0.65$, or trade executed $\le 336\text{h}$ before announcement, or volume $Z \ge 2.0$. |
| **SEBI PIT Reg 3(1)** | Communication or Procurement of UPSI | Trader role $\in \{\text{Director, KMP, Promoter}\}$ trading before news, or PAN linked trades $> 0$. |
| **SEBI PIT Schedule B** | Trading Window Closure & Pre-Clearance | Trade executed $\le 168\text{h}$ before announcement (window closure), or pre-clearance status is `Required`. |
| **SEBI PIT Reg 7(2)** | Continual Disclosure Obligations | Transaction value $\ge \text{Rs. } 10\text{ Lakhs}$ or order size percentile $\ge 80\%$. |

---

<br>

# 💻 SECTION 2: PROJECT IMPLEMENTATION & CODEBASE ARCHITECTURE

*This section provides a complete developer walkthrough of the codebase, backend API endpoints, regulatory reasoning engine, and web interface.*

---

## 1. Directory & File Structure

```text
c:/insider_trading_detection/
├── app.py                         # FastAPI web server & REST API endpoints
├── insider_pipeline.py            # ML feature engineering & ensemble trainer
├── sentiment_analysis.py          # FinBERT NLP sentiment & lead-lag engine
├── regulatory_reasoning.py        # Hugging Face LLM legal reasoning & ReportLab PDF generator
├── templates/
│   └── index.html                 # Interactive web dashboard & Explainability Modal UI
├── static/
│   └── style.css                  # Modern dark-mode glassmorphic styling
├── data/
│   ├── trade.csv                  # 20,000 trade records with point-in-time features
│   ├── profile.csv                # 1,000 trader profiles (roles, PANs, pre-clearance rules)
│   ├── events_real_master.csv     # Scraped NSE corporate announcements & news headlines
│   └── insider_detection_FINAL.ipynb # Jupyter notebook benchmark
└── outputs/                       # Persisted ML model bundles, scored trades & PDFs
```

---

## 2. Core Python Modules & Responsibilities

### A. `insider_pipeline.py` (ML Classification Engine)
* **Feature Construction**: Computes $Z_{\text{volume}}$, order size percentiles, historical trade frequencies, and point-in-time pre-trade returns.
* **Ensemble Architecture**: Soft-voting ensemble combining `LogisticRegression` (with `StandardScaler`) and `HistGradientBoostingClassifier`.
* **Optimization**: Automated threshold selection ($\beta = 2.0$) prioritizing recall to catch suspicious trades while preserving high precision.

### B. `sentiment_analysis.py` (FinBERT & Lead-Lag Engine)
* **FinBERT Batch Inference**: Loads `ProsusAI/finbert` to classify pre-event news titles into positive, negative, or neutral sentiment.
* **Lead-Lag Analysis**: Compares trade timestamps with article `pub_ist` timestamps to categorize trades (`trade_before_news`, `trade_after_news`, `mixed_mostly_after`).
* **Composite Refinement**: Adjusts ML probabilities using event proximity and sentiment alignment.

### C. `regulatory_reasoning.py` (LLM Reasoning & PDF Generation)
* **LLM Engine**: Lazy-loads Hugging Face `transformers` pipeline (`distilgpt2`) to synthesize legal rationale.
* **Feature-Driven Synthesis**: Combines trade role, execution timestamp, news headline, and volume z-score into legal explanations.
* **ReportLab PDF Engine**: Generates PDF compliance audit reports containing overview tables, news indicators, and SEBI statutory citations.

### D. `app.py` (FastAPI Web Backend)
Provides asynchronous REST endpoints serving the web dashboard:

| Endpoint | Method | Function & Description |
| :--- | :--- | :--- |
| `/` | `GET` | Serves the interactive HTML5/JS Web Dashboard. |
| `/api/upload` | `POST` | Accepts `trade.csv` & `profile.csv`, runs feature engineering + ML model, returns metrics & flagged trades. |
| `/api/sentiment` | `POST` | Executes FinBERT sentiment analysis & lead-lag refinement on flagged trades. |
| `/api/explainability/trades` | `GET` | Returns list of ground-truth insider sentiment-flagged trades (`is_insider==1 & sentiment_flag==1`). |
| `/api/explainability/trade/{trade_id}` | `GET` | Fetches trade details, trader profile, news article, and violated SEBI sections. |
| `/api/explainability/pdf/{trade_id}` | `GET` | Generates and downloads the official ReportLab PDF compliance audit report. |

---

## 🚀 Quick Start & Deployment Guide

### 1. Prerequisites & Environment Setup
Ensure Python 3.10+ is installed:
```bash
cd c:/insider_trading_detection
pip install fastapi uvicorn pandas numpy scikit-learn transformers reportlab pypdf torch
```

### 2. Run Command-Line Pipeline (Optional)
```bash
python insider_pipeline.py --trade data/trade.csv --profile data/profile.csv --out-dir outputs
```

### 3. Launch Web Application Server
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access Interactive Dashboard
Open your web browser and navigate to:
```text
http://localhost:8000/
```

1. **Upload Files**: Drag & drop `trade.csv` and `profile.csv`.
2. **Run ML Classification**: Click `Run ML Classification` to view held-out test metrics (Accuracy 95.5%, Precision 81.7%, Recall 98.1%).
3. **Run Sentiment Analysis**: Click `Run FinBERT Sentiment Analysis` to evaluate news headlines and lead-lag timing.
4. **Inspect Explainability & Download PDF**: Click `Explainability` on any trade row to view full trader profile, timestamp, matched news headline, and export the official **PDF Audit Report**.

---

## 📜 Academic Attribution & License
* **Institution**: Dept. of Artificial Intelligence & Data Science, VESIT (2025-26).
* **Regulatory Reference**: Securities and Exchange Board of India (Prohibition of Insider Trading) Regulations, 2015.
