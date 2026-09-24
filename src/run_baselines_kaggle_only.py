import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path


# Load cached features
OUT_DIR = Path("results")

X_train = np.load(OUT_DIR/"X_train.npy")
y_train = np.load(OUT_DIR/"y_train.npy")

X_test = np.load(OUT_DIR/"X_k_test.npy")
y_test = np.load(OUT_DIR/"y_k_test.npy")

# Feature layout: 768 XLNet + 4 VADER
Xtr_xlnet = X_train[:, :768]
Xtr_vader = X_train[:, 768:]

Xte_xlnet = X_test[:, :768]
Xte_vader = X_test[:, 768:]

def train_eval(name, Xtr, ytr, Xte, yte):
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(Xtr, ytr)
    preds = rf.predict(Xte)
    acc = accuracy_score(yte, preds)
    print(f"{name:<22} | Accuracy: {acc:.4f}")
    return acc

print("\n=== BASELINE COMPARISON (Kaggle) ===\n")

acc_hybrid = train_eval("Hybrid (XLNet+VADER)", X_train, y_train, X_test, y_test)
acc_xlnet  = train_eval("XLNet-only", Xtr_xlnet, y_train, Xte_xlnet, y_test)
acc_vader  = train_eval("VADER-only", Xtr_vader, y_train, Xte_vader, y_test)