import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
from pathlib import Path

DATA = Path("data")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

fake = pd.read_csv(DATA / "fake-real-news/Fake.csv")
true = pd.read_csv(DATA / "fake-real-news/True.csv")
fake["label"] = 0
true["label"] = 1
df = pd.concat([fake, true], ignore_index=True)

df["full_text"] = (df["title"].astype(str) + " " + df["text"].astype(str)).str.replace(r"\s+", " ", regex=True).str.strip()

X_train, X_test, y_train, y_test = train_test_split(
    df["full_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

tfidf = TfidfVectorizer(max_features=50000, stop_words="english")
X_train_t = tfidf.fit_transform(X_train)
X_test_t = tfidf.transform(X_test)

rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train_t, y_train)

pred = rf.predict(X_test_t)

acc = accuracy_score(y_test, pred)
report = classification_report(y_test, pred, digits=4, output_dict=True)
cm = confusion_matrix(y_test, pred).tolist()

print("Accuracy:", acc)
print("Confusion matrix:\n", cm)

(OUT / "baseline_metrics.json").write_text(json.dumps({
    "accuracy": acc,
    "classification_report": report,
    "confusion_matrix": cm
}, indent=2))
print("Saved: results/baseline_metrics.json")