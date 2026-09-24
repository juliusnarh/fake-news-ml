"""
Hybrid Fake News Detection - Full Pipeline (Distinction Version)
- Train: Kaggle + 20% LIAR
- Test: Kaggle test, LIAR holdout
- External test: FakeNewsNet subsets (BuzzFeed + PolitiFact CSVs)
- Explainability: Feature importance + (optional) SHAP
- Real-time simulation: Inference latency benchmarking
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from transformers import AutoTokenizer, AutoModel
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# ---------------------------
# SETTINGS (EDIT HERE)
# ---------------------------
DATA_DIR = Path("data")
OUT_DIR = Path("results")
FIGURE_OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(exist_ok=True)
FIGURE_OUT_DIR.mkdir(parents=True, exist_ok=True) 

# Kaggle dataset files (already used)
KAGGLE_FAKE = DATA_DIR / "fake-real-news/Fake.csv"
KAGGLE_TRUE = DATA_DIR / "fake-real-news/True.csv"

# LIAR TSV files
LIAR_TRAIN = DATA_DIR / "LIAR/train.tsv"
LIAR_VALID = DATA_DIR / "LIAR/valid.tsv"
LIAR_TEST  = DATA_DIR / "LIAR/test.tsv"

# FakeNewsNet CSVs (YOU UPLOADED THESE)
BUZZ_FAKE = DATA_DIR / "buzzfeed-news/BuzzFeed_fake_news_content.csv"
BUZZ_REAL = DATA_DIR / "buzzfeed-news/BuzzFeed_real_news_content.csv"
POLI_FAKE = DATA_DIR / "buzzfeed-news/PolitiFact_fake_news_content.csv"
POLI_REAL = DATA_DIR / "buzzfeed-news/PolitiFact_real_news_content.csv"

# XLNet settings
MODEL_NAME = "xlnet-base-cased"
BATCH_SIZE = 16       # Mac M4 MPS should handle this
MAX_LEN = 256

# Random Forest settings
N_ESTIMATORS = 300

# Explainability toggles
RUN_SHAP = False      # set True if you install shap
SHAP_SAMPLES = 300    # how many samples for SHAP (keep small)

# Latency benchmark
LATENCY_RUNS = 200

# Evaluation toggle
STOP_AFTER_CACHE = False

# ---------------------------
# DEVICE SETUP
# ---------------------------
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("Using device:", device)

# ---------------------------
# NLTK VADER
# ---------------------------
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()


def get_sentiment(texts: pd.Series) -> np.ndarray:
    feats = []
    for t in texts:
        s = sia.polarity_scores(str(t))
        feats.append([s["neg"], s["neu"], s["pos"], s["compound"]])
    return np.array(feats, dtype=np.float32)


# ---------------------------
# XLNET EMBEDDINGS
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
xlnet = AutoModel.from_pretrained(MODEL_NAME).to(device)
xlnet.eval()


def get_xlnet_embeddings(texts: pd.Series, batch_size=BATCH_SIZE, max_len=MAX_LEN) -> np.ndarray:
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = list(texts.iloc[i:i + batch_size])
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(device)

            out = xlnet(**enc)
            last_hidden = out.last_hidden_state  # (B,T,H)

            mask = enc["attention_mask"].unsqueeze(-1)  # (B,T,1)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts  # (B,H)

            vecs.append(pooled.cpu().numpy())

    return np.vstack(vecs)


# ---------------------------
# HELPERS
# ---------------------------
def eval_pack(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(y_true, y_pred, digits=4, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }


def build_features(df: pd.DataFrame, text_col: str) -> np.ndarray:
    Xx = get_xlnet_embeddings(df[text_col])
    Xs = get_sentiment(df[text_col])
    return np.hstack([Xx, Xs])


def load_fakenewsnet_csv(fake_path: Path, real_path: Path) -> pd.DataFrame:
    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)
    fake["label"] = 0
    real["label"] = 1

    df = pd.concat([fake, real], ignore_index=True)

    # title + text (your preference)
    df["full_text"] = (
        df["title"].astype(str).fillna("") + " " + df["text"].astype(str).fillna("")
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    return df[["full_text", "label"]]


def save_confusion_matrix(y_true, y_pred, labels, filename, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def save_roc_curve(y_true, y_score, filename, title):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def save_semantic_vs_sentiment_bar(semantic, sentiment, filename, title):
    plt.figure(figsize=(6, 5))
    plt.bar(["XLNet semantic (768 dims)", "VADER sentiment (4 dims)"], [semantic, sentiment])
    plt.ylabel("Total feature importance (sum)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
def save_baseline_accuracy_bar(acc_dict, filename, title):
    """
    acc_dict example:
      {"Hybrid (XLNet+VADER)": 0.9888, "XLNet-only": 0.9888, "VADER-only": 0.6867}
    """
    labels = list(acc_dict.keys())
    vals = [acc_dict[k] * 100 for k in labels]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, vals)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title(title)

    # value labels on top
    for i, v in enumerate(vals):
        plt.text(i, v + 1, f"{v:.2f}%", ha="center")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ==========================================================
# 1) LOAD + PREPARE KAGGLE (TRAIN/TEST)
# ==========================================================
print("\n[1] Loading Kaggle...")
fake = pd.read_csv(KAGGLE_FAKE)
true = pd.read_csv(KAGGLE_TRUE)
fake["label"] = 0
true["label"] = 1
kaggle = pd.concat([fake, true], ignore_index=True)

kaggle["full_text"] = (
    kaggle["title"].astype(str).fillna("") + " " + kaggle["text"].astype(str).fillna("")
).str.replace(r"\s+", " ", regex=True).str.strip()

k_train, k_test = train_test_split(
    kaggle[["full_text", "label"]],
    test_size=0.2,
    random_state=42,
    stratify=kaggle["label"]
)

print("Kaggle train/test:", k_train.shape, k_test.shape)

# ==========================================================
# 2) LOAD + PREPARE LIAR (BINARY) + HOLDOUT SPLIT
# ==========================================================
print("\n[2] Loading LIAR...")
liar_train = pd.read_csv(LIAR_TRAIN, sep="\t", header=None)
liar_valid = pd.read_csv(LIAR_VALID, sep="\t", header=None)
liar_test  = pd.read_csv(LIAR_TEST,  sep="\t", header=None)

liar = pd.concat([liar_train, liar_valid, liar_test], ignore_index=True)
liar = liar[[1, 2]]
liar.columns = ["label", "full_text"]

fake_labels = {"pants-fire", "false", "barely-true"}
liar["label"] = liar["label"].apply(lambda x: 0 if x in fake_labels else 1)
liar["full_text"] = liar["full_text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

# 20% LIAR into training, 80% LIAR holdout test
l_train, l_holdout = train_test_split(
    liar,
    test_size=0.8,
    random_state=42,
    stratify=liar["label"]
)

print("LIAR train/holdout:", l_train.shape, l_holdout.shape)

# ==========================================================
# 3) BUILD MULTI-DOMAIN TRAIN SET
# ==========================================================
print("\n[3] Building multi-domain training set...")
train_df = pd.concat([k_train, l_train], ignore_index=True)
print("Final train size:", train_df.shape)

# ==========================================================
# 4) FEATURE EXTRACTION (TRAIN + TESTS)
# ==========================================================
print("\n[4] Extracting features... (this is the heavy part)")

print("-> TRAIN features")
X_train = build_features(train_df, "full_text")
y_train = train_df["label"].values

print("-> Kaggle TEST features")
X_k_test = build_features(k_test, "full_text")
y_k_test = k_test["label"].values

print("-> LIAR HOLDOUT features")
X_l_test = build_features(l_holdout, "full_text")
y_l_test = l_holdout["label"].values

# ---- CACHE FEATURES (add this once) ----
np.save(OUT_DIR / "X_train.npy", X_train)
np.save(OUT_DIR / "y_train.npy", y_train)

np.save(OUT_DIR / "X_k_test.npy", X_k_test)
np.save(OUT_DIR / "y_k_test.npy", y_k_test)
# ---- CACHE FEATURES (add this once) ----

# ==========================================================
# 5) TRAIN RANDOM FOREST
# ==========================================================
print("\n[5] Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ---- SAVE MODEL ----
import joblib
joblib.dump(rf, OUT_DIR / "rf_model.joblib")
# ---- SAVE MODEL ----


if STOP_AFTER_CACHE:
    print("Cached features saved. Exiting early (no external eval, no SHAP).")
    raise SystemExit


# ==========================================================
# 6) EVALUATE ON KAGGLE + LIAR
# ==========================================================
print("\n[6] Evaluating Kaggle + LIAR...")
k_pred = rf.predict(X_k_test)
l_pred = rf.predict(X_l_test)


# ==========================================================
# BASELINES (XLNet-only, VADER-only) vs HYBRID (XLNet+VADER)
# ==========================================================
print("\n[BASELINES] Training + evaluating XLNet-only and VADER-only baselines...")

# Split features
X_train_xlnet = X_train[:, :768]
X_train_vader = X_train[:, 768:]

X_k_test_xlnet = X_k_test[:, :768]
X_k_test_vader = X_k_test[:, 768:]

# Train baseline models (same RF settings for fairness)
rf_xlnet = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1)
rf_vader = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1)

rf_xlnet.fit(X_train_xlnet, y_train)
rf_vader.fit(X_train_vader, y_train)

# Predictions on Kaggle test (to match your Table 13 context)
k_pred_hybrid = rf.predict(X_k_test)
k_pred_xlnet = rf_xlnet.predict(X_k_test_xlnet)
k_pred_vader = rf_vader.predict(X_k_test_vader)

acc_hybrid = accuracy_score(y_k_test, k_pred_hybrid)
acc_xlnet = accuracy_score(y_k_test, k_pred_xlnet)
acc_vader = accuracy_score(y_k_test, k_pred_vader)

print(f"Hybrid accuracy:   {acc_hybrid:.4f}")
print(f"XLNet-only acc:    {acc_xlnet:.4f}")
print(f"VADER-only acc:    {acc_vader:.4f}")

# ---- Figure 1: Baseline Accuracy Bar Chart (supports Table 13) ----
baseline_acc = {
    "Hybrid (XLNet+VADER)": float(acc_hybrid),
    "XLNet-only": float(acc_xlnet),
    "VADER-only": float(acc_vader)
}
save_baseline_accuracy_bar(
    baseline_acc,
    filename=FIGURE_OUT_DIR / "fig_baseline_accuracy_comparison.png",
    title="Baseline Comparison on Kaggle Test Set"
)

# ---- Figure 2: Confusion Matrices (shows error patterns) ----
save_confusion_matrix(
    y_k_test, k_pred_hybrid,
    labels=[0, 1],
    filename=FIGURE_OUT_DIR / "fig_cm_kaggle_hybrid.png",
    title="Confusion Matrix – Kaggle Test (Hybrid)"
)

save_confusion_matrix(
    y_k_test, k_pred_xlnet,
    labels=[0, 1],
    filename=FIGURE_OUT_DIR / "fig_cm_kaggle_xlnet_only.png",
    title="Confusion Matrix – Kaggle Test (XLNet-only)"
)

save_confusion_matrix(
    y_k_test, k_pred_vader,
    labels=[0, 1],
    filename=FIGURE_OUT_DIR / "fig_cm_kaggle_vader_only.png",
    title="Confusion Matrix – Kaggle Test (VADER-only)"
)

# ---- Optional Figure 3: ROC curves (nice extra support) ----
k_prob_hybrid = rf.predict_proba(X_k_test)[:, 1]
k_prob_xlnet = rf_xlnet.predict_proba(X_k_test_xlnet)[:, 1]
k_prob_vader = rf_vader.predict_proba(X_k_test_vader)[:, 1]

save_roc_curve(
    y_k_test, k_prob_hybrid,
    filename=FIGURE_OUT_DIR / "fig_roc_kaggle_hybrid.png",
    title="ROC Curve – Kaggle Test (Hybrid)"
)

save_roc_curve(
    y_k_test, k_prob_xlnet,
    filename=FIGURE_OUT_DIR / "fig_roc_kaggle_xlnet_only.png",
    title="ROC Curve – Kaggle Test (XLNet-only)"
)

save_roc_curve(
    y_k_test, k_prob_vader,
    filename=FIGURE_OUT_DIR / "fig_roc_kaggle_vader_only.png",
    title="ROC Curve – Kaggle Test (VADER-only)"
)

k_res = eval_pack(y_k_test, k_pred)
l_res = eval_pack(y_l_test, l_pred)

print("Kaggle Accuracy:", k_res["accuracy"])
print("LIAR Holdout Accuracy:", l_res["accuracy"])

# ---- FIGURES: Confusion Matrices (Kaggle + LIAR) ----
save_confusion_matrix(
    y_k_test, k_pred,
    labels=[0, 1],
    filename=FIGURE_OUT_DIR / "fig_cm_kaggle.png",
    title="Confusion Matrix – Kaggle Test (Hybrid RF)"
)

save_confusion_matrix(
    y_l_test, l_pred,
    labels=[0, 1],
    filename=FIGURE_OUT_DIR / "fig_cm_liar.png",
    title="Confusion Matrix – LIAR Holdout (Hybrid RF)"
)

# ---- FIGURES: ROC Curves (Kaggle + LIAR) ----
k_prob = rf.predict_proba(X_k_test)[:, 1]
l_prob = rf.predict_proba(X_l_test)[:, 1]

save_roc_curve(
    y_k_test, k_prob,
    filename=FIGURE_OUT_DIR / "fig_roc_kaggle.png",
    title="ROC Curve – Kaggle Test (Hybrid RF)"
)

save_roc_curve(
    y_l_test, l_prob,
    filename=FIGURE_OUT_DIR / "fig_roc_liar.png",
    title="ROC Curve – LIAR Holdout (Hybrid RF)"
)

# ==========================================================
# 7) LOAD + EVALUATE FakeNewsNet (BuzzFeed + PolitiFact)
# ==========================================================
print("\n[7] Evaluating FakeNewsNet subsets (BuzzFeed + PolitiFact)...")

buzz = load_fakenewsnet_csv(BUZZ_FAKE, BUZZ_REAL)
poli = load_fakenewsnet_csv(POLI_FAKE, POLI_REAL)

print("BuzzFeed size:", buzz.shape)
print("PolitiFact size:", poli.shape)

print("-> BuzzFeed features")
X_buzz = build_features(buzz, "full_text")
y_buzz = buzz["label"].values
buzz_pred = rf.predict(X_buzz)
buzz_res = eval_pack(y_buzz, buzz_pred)
print("BuzzFeed Accuracy:", buzz_res["accuracy"])

print("-> PolitiFact features")
X_poli = build_features(poli, "full_text")
y_poli = poli["label"].values
poli_pred = rf.predict(X_poli)
poli_res = eval_pack(y_poli, poli_pred)
print("PolitiFact Accuracy:", poli_res["accuracy"])

# ---- FIGURES: Confusion Matrices (External) ----
save_confusion_matrix(
    y_buzz, buzz_pred,
    labels=[0, 1],
    filename= FIGURE_OUT_DIR / "fig_cm_buzzfeed.png",
    title="Confusion Matrix – BuzzFeed External Test (Hybrid RF)"
)

save_confusion_matrix(
    y_poli, poli_pred,
    labels=[0, 1],
    filename=FIGURE_OUT_DIR / "fig_cm_politifact.png",
    title="Confusion Matrix – PolitiFact External Test (Hybrid RF)"
)

# ---- FIGURES: ROC Curves (External) ----
buzz_prob = rf.predict_proba(X_buzz)[:, 1]
poli_prob = rf.predict_proba(X_poli)[:, 1]

save_roc_curve(
    y_buzz, buzz_prob,
    filename=FIGURE_OUT_DIR / "fig_roc_buzzfeed.png",
    title="ROC Curve – BuzzFeed External Test (Hybrid RF)"
)

save_roc_curve(
    y_poli, poli_prob,
    filename=FIGURE_OUT_DIR / "fig_roc_politifact.png",
    title="ROC Curve – PolitiFact External Test (Hybrid RF)"
)

# ==========================================================
# 8) FEATURE IMPORTANCE (Semantic vs Sentiment)
# ==========================================================
print("\n[8] Feature importance summary...")
imp = rf.feature_importances_
semantic_importance = float(imp[:768].sum())
sentiment_importance = float(imp[768:].sum())

print("Semantic importance:", semantic_importance)
print("Sentiment importance:", sentiment_importance)

save_semantic_vs_sentiment_bar(
    semantic_importance,
    sentiment_importance,
    filename=FIGURE_OUT_DIR / "fig_importance_semantic_vs_sentiment.png",
    title="Semantic vs Sentiment Contribution (Random Forest Importance)"
)

importance_summary = {
    "semantic_importance_sum": semantic_importance,
    "sentiment_importance_sum": sentiment_importance,
    "semantic_percent": float(semantic_importance / (semantic_importance + sentiment_importance)),
    "sentiment_percent": float(sentiment_importance / (semantic_importance + sentiment_importance))
}

# ==========================================================
# 9) REAL-TIME SIMULATION: INFERENCE LATENCY (ms)
# ==========================================================
print("\n[9] Inference latency benchmark...")
sample = X_k_test[0].reshape(1, -1)

times = []
for _ in range(LATENCY_RUNS):
    t0 = time.time()
    rf.predict(sample)
    times.append(time.time() - t0)

latency = {
    "avg_ms": float(np.mean(times) * 1000),
    "p95_ms": float(np.percentile(times, 95) * 1000),
    "runs": LATENCY_RUNS
}

print("Avg latency (ms):", latency["avg_ms"])
print("P95 latency (ms):", latency["p95_ms"])

# ==========================================================
# 10) SHAP (OPTIONAL)
# ==========================================================
shap_notes = {}
if RUN_SHAP:
    print("\n[10] Running SHAP...")
    import shap
    import matplotlib.pyplot as plt

    bg_size = min(200, X_train.shape[0])
    bg_idx = np.random.choice(X_train.shape[0], size=bg_size, replace=False)
    background = X_train[bg_idx]

    X_shap = X_k_test[:SHAP_SAMPLES]

    # Feature names (important for thesis plots)
    feature_names = [f"XLNet_{i}" for i in range(768)] + ["VADER_neg", "VADER_neu", "VADER_pos", "VADER_compound"]

    explainer = shap.Explainer(rf.predict_proba, background)
    # sv = explainer(X_shap)
    sv = explainer(X_shap, max_evals=2000)

    if sv.values.ndim == 3:
        sv_class1 = shap.Explanation(
            values=sv.values[:, :, 1],
            base_values=sv.base_values[:, 1] if getattr(sv.base_values, "ndim", 1) == 2 else sv.base_values,
            data=X_shap,
            feature_names=feature_names
        )
    else:
        sv_class1 = sv

    # Beeswarm
    plt.figure()
    shap.plots.beeswarm(sv_class1, show=False)
    plt.tight_layout()
    plt.savefig(FIGURE_OUT_DIR / "shap_summary_kaggle_class1.png", dpi=300)
    plt.close()

    # Bar
    plt.figure()
    shap.plots.bar(sv_class1, show=False)
    plt.tight_layout()
    plt.savefig(FIGURE_OUT_DIR / "shap_bar_kaggle_class1.png", dpi=300)
    plt.close()

    shap_notes = {
        "enabled": True,
        "shap_version": shap.__version__,
        "background_size": int(bg_size),
        "samples_explained": int(X_shap.shape[0]),
        "class_explained": 1,
        "saved_files": [
            str(FIGURE_OUT_DIR / "shap_summary_kaggle_class1.png"),
            str(FIGURE_OUT_DIR / "shap_bar_kaggle_class1.png"),
        ],
        "note": "Class 1 corresponds to label=1 (Real/True). Feature names label XLNet dims and VADER sentiment."
    }
else:
    shap_notes = {"enabled": False, "note": "Set RUN_SHAP=True after installing shap to enable."}

# ==========================================================
# 11) SAVE ALL RESULTS INTO ONE JSON
# ==========================================================
final = {
    "train_mix": {
        "kaggle_train_size": int(k_train.shape[0]),
        "liar_train_size": int(l_train.shape[0]),
        "liar_holdout_size": int(l_holdout.shape[0]),
        "total_train_size": int(train_df.shape[0])
    },
    "kaggle_test": k_res,
    "liar_holdout_test": l_res,
    "buzzfeed_external_test": buzz_res,
    "politifact_external_test": poli_res,
    "feature_importance_summary": importance_summary,
    "latency_ms": latency,
    "shap": shap_notes,
    "settings": {
        "model_name": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "max_len": MAX_LEN,
        "n_estimators": N_ESTIMATORS
    }
}

out_path = OUT_DIR / "hybrid_full_pipeline_results.json"
out_path.write_text(json.dumps(final, indent=2))
print("\nSaved:", out_path)
print("DONE.")