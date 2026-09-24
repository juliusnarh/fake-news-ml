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

# -----------------------------
# 1️⃣ DEVICE SETUP (Mac M4)
# -----------------------------
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("Using device:", device)

# -----------------------------
# 2️⃣ LOAD KAGGLE DATA
# -----------------------------
fake = pd.read_csv(DATA / "fake-real-news/Fake.csv")
true = pd.read_csv(DATA / "fake-real-news/True.csv")

fake["label"] = 0
true["label"] = 1

kaggle_df = pd.concat([fake, true], ignore_index=True)
kaggle_df["text"] = (kaggle_df["title"].astype(str) + " " + kaggle_df["text"].astype(str))

train_df, test_df = train_test_split(
    kaggle_df,
    test_size=0.2,
    random_state=42,
    stratify=kaggle_df["label"]
)

# -----------------------------
# 3️⃣ VADER SENTIMENT
# -----------------------------
sia = SentimentIntensityAnalyzer()

def get_sentiment(texts):
    features = []
    for t in texts:
        s = sia.polarity_scores(str(t))
        features.append([s["neg"], s["neu"], s["pos"], s["compound"]])
    return np.array(features)

# -----------------------------
# 4️⃣ XLNet EMBEDDINGS
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
model = AutoModel.from_pretrained("xlnet-base-cased").to(device)
model.eval()

def get_xlnet_embeddings(texts, batch_size=16, max_len=256):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(device)

            out = model(**enc)
            last_hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            masked = last_hidden * mask
            summed = masked.sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts

            embeddings.append(pooled.cpu().numpy())

    return np.vstack(embeddings)

print("Extracting XLNet embeddings (Kaggle Train)...")
X_train_xlnet = get_xlnet_embeddings(train_df["text"])

print("Extracting XLNet embeddings (Kaggle Test)...")
X_test_xlnet = get_xlnet_embeddings(test_df["text"])

# -----------------------------
# 5️⃣ COMBINE FEATURES
# -----------------------------
X_train_sent = get_sentiment(train_df["text"])
X_test_sent = get_sentiment(test_df["text"])

X_train = np.hstack([X_train_xlnet, X_train_sent])
X_test = np.hstack([X_test_xlnet, X_test_sent])

y_train = train_df["label"].values
y_test = test_df["label"].values

# -----------------------------
# 6️⃣ RANDOM FOREST
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# -----------------------------
# 7️⃣ KAGGLE EVALUATION
# -----------------------------
pred = rf.predict(X_test)

kaggle_results = {
    "accuracy": accuracy_score(y_test, pred),
    "classification_report": classification_report(y_test, pred, output_dict=True),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist()
}

print("Kaggle Accuracy:", kaggle_results["accuracy"])

# -----------------------------
# 8️⃣ LOAD LIAR (External Test)
# -----------------------------
liar_train = pd.read_csv(DATA / "LIAR/train.tsv", sep="\t", header=None)
liar_valid = pd.read_csv(DATA / "LIAR/valid.tsv", sep="\t", header=None)
liar_test = pd.read_csv(DATA / "LIAR/test.tsv", sep="\t", header=None)

liar_df = pd.concat([liar_train, liar_valid, liar_test], ignore_index=True)
liar_df = liar_df[[1,2]]
liar_df.columns = ["label", "text"]

fake_labels = ["pants-fire", "false", "barely-true"]
liar_df["label"] = liar_df["label"].apply(lambda x: 0 if x in fake_labels else 1)

print("Extracting XLNet embeddings (LIAR)...")
X_liar_xlnet = get_xlnet_embeddings(liar_df["text"])
X_liar_sent = get_sentiment(liar_df["text"])

X_liar = np.hstack([X_liar_xlnet, X_liar_sent])
y_liar = liar_df["label"].values

liar_pred = rf.predict(X_liar)

liar_results = {
    "accuracy": accuracy_score(y_liar, liar_pred),
    "classification_report": classification_report(y_liar, liar_pred, output_dict=True),
    "confusion_matrix": confusion_matrix(y_liar, liar_pred).tolist()
}

print("LIAR External Accuracy:", liar_results["accuracy"])

# -----------------------------
# 9️⃣ SAVE RESULTS
# -----------------------------
final_results = {
    "kaggle": kaggle_results,
    "liar_external": liar_results
}

(OUT / "hybrid_results.json").write_text(json.dumps(final_results, indent=2))

print("Results saved to results/hybrid_results.json")