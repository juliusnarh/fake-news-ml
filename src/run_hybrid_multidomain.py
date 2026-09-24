import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path

from transformers import AutoTokenizer, AutoModel
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

nltk.download("vader_lexicon")

DATA = Path("data")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

# -------- Device (Mac MPS) --------
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("Using device:", device)

# -------- Sentiment --------
sia = SentimentIntensityAnalyzer()

def get_sentiment(texts):
    feats = []
    for t in texts:
        s = sia.polarity_scores(str(t))
        feats.append([s["neg"], s["neu"], s["pos"], s["compound"]])
    return np.array(feats, dtype=np.float32)

# -------- XLNet --------
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
model = AutoModel.from_pretrained("xlnet-base-cased").to(device)
model.eval()

def get_xlnet_embeddings(texts, batch_size=16, max_len=256):
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i+batch_size])
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(device)

            out = model(**enc)
            last_hidden = out.last_hidden_state  # (B,T,H)

            mask = enc["attention_mask"].unsqueeze(-1)  # (B,T,1)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts  # (B,H)

            vecs.append(pooled.cpu().numpy())
    return np.vstack(vecs)

# ============================
# 1) Kaggle: train/test split
# ============================
fake = pd.read_csv(DATA / "fake-real-news/Fake.csv")
true = pd.read_csv(DATA / "fake-real-news/True.csv")
fake["label"] = 0
true["label"] = 1
kaggle_df = pd.concat([fake, true], ignore_index=True)

kaggle_df["text"] = (kaggle_df["title"].astype(str) + " " + kaggle_df["text"].astype(str)).str.replace(r"\s+", " ", regex=True).str.strip()

k_train, k_test = train_test_split(
    kaggle_df, test_size=0.2, random_state=42, stratify=kaggle_df["label"]
)

# ============================
# 2) LIAR: load + binary map
# ============================
liar_train = pd.read_csv(DATA / "LIAR/train.tsv", sep="\t", header=None)
liar_valid = pd.read_csv(DATA / "LIAR/valid.tsv", sep="\t", header=None)
liar_test  = pd.read_csv(DATA / "LIAR/test.tsv",  sep="\t", header=None)
liar_df = pd.concat([liar_train, liar_valid, liar_test], ignore_index=True)

liar_df = liar_df[[1, 2]]
liar_df.columns = ["label", "text"]

fake_labels = {"pants-fire", "false", "barely-true"}
liar_df["label"] = liar_df["label"].apply(lambda x: 0 if x in fake_labels else 1)
liar_df["text"] = liar_df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

# LIAR split: 20% train injection, 80% holdout test
l_train, l_holdout = train_test_split(
    liar_df, test_size=0.8, random_state=42, stratify=liar_df["label"]
)

print("Kaggle train/test:", k_train.shape, k_test.shape)
print("LIAR train/holdout:", l_train.shape, l_holdout.shape)

# ============================
# 3) Multi-domain training set
# ============================
train_df = pd.concat([k_train[["text","label"]], l_train[["text","label"]]], ignore_index=True)

# ============================
# 4) Feature extraction
# ============================
print("Embedding TRAIN (Kaggle+LIAR-train)...")
X_train_x = get_xlnet_embeddings(train_df["text"])
X_train_s = get_sentiment(train_df["text"])
X_train = np.hstack([X_train_x, X_train_s])
y_train = train_df["label"].values

print("Embedding Kaggle TEST...")
X_ktest_x = get_xlnet_embeddings(k_test["text"])
X_ktest_s = get_sentiment(k_test["text"])
X_ktest = np.hstack([X_ktest_x, X_ktest_s])
y_ktest = k_test["label"].values

print("Embedding LIAR HOLDOUT...")
X_ltest_x = get_xlnet_embeddings(l_holdout["text"])
X_ltest_s = get_sentiment(l_holdout["text"])
X_ltest = np.hstack([X_ltest_x, X_ltest_s])
y_ltest = l_holdout["label"].values

# ============================
# 5) Train Random Forest
# ============================
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ============================
# 6) Evaluate
# ============================
def eval_pack(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(y_true, y_pred, digits=4, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

k_pred = rf.predict(X_ktest)
l_pred = rf.predict(X_ltest)

k_res = eval_pack(y_ktest, k_pred)
l_res = eval_pack(y_ltest, l_pred)

print("Kaggle Accuracy (multi-domain):", k_res["accuracy"])
print("LIAR Holdout Accuracy (multi-domain):", l_res["accuracy"])

final = {
    "train_mix": {
        "kaggle_train_size": int(k_train.shape[0]),
        "liar_train_size": int(l_train.shape[0]),
        "liar_holdout_size": int(l_holdout.shape[0]),
    },
    "kaggle_test": k_res,
    "liar_holdout_test": l_res
}

(OUT / "hybrid_multidomain_results.json").write_text(json.dumps(final, indent=2))
print("Saved: results/hybrid_multidomain_results.json")