import os
import joblib
import numpy as np
import pandas as pd
import wikipedia

from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# ===============================
# 1. Load Dataset
# ===============================
DATA_PATH = "data/dataset.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("❌ dataset.csv not found inside data/ folder")

df = pd.read_csv(DATA_PATH)

# ===============================
# 2. Clean Data (CRITICAL)
# ===============================
df = df.dropna(subset=["question", "answer", "label"])

df["question"] = df["question"].astype(str)
df["answer"] = df["answer"].astype(str)
df["label"] = df["label"].astype(int)

print("✅ Dataset size after cleaning:", df.shape)

# ===============================
# 3. Prepare Text Input (IMPROVED)
# ===============================
texts = [
    f"Question: {q}\nAnswer: {a}\nIs this answer factually correct?"
    for q, a in zip(df["question"], df["answer"])
]

labels = df["label"].values

# ===============================
# 4. Load SBERT Model
# ===============================
print("🔄 Loading SBERT model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("🔄 Generating embeddings...")
X_embed = embedder.encode(texts, show_progress_bar=True)

# ===============================
# 5. Wikipedia Verification Feature
# ===============================
def get_verification_score(answer, embedder):
    """
    Computes cosine similarity between answer and Wikipedia summary.
    Returns 0.0 if lookup fails.
    """
    try:
        summary = wikipedia.summary(
            answer,
            sentences=2,
            auto_suggest=False,
            redirect=True
        )
        emb_answer = embedder.encode(answer, convert_to_tensor=True)
        emb_summary = embedder.encode(summary, convert_to_tensor=True)
        score = util.cos_sim(emb_answer, emb_summary).item()
        return score
    except Exception:
        return 0.0


print("🔄 Computing verification scores (this may take time)...")
ver_scores = [
    get_verification_score(a, embedder)
    for a in df["answer"]
]

ver_scores = np.array(ver_scores).reshape(-1, 1)

# ===============================
# 6. Combine Features
# ===============================
X = np.hstack([X_embed, ver_scores])

print("✅ Final feature shape:", X.shape)

# ===============================
# 7. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# ===============================
# 8. Train BALANCED Classifier
# ===============================
print("🔄 Training classifier (class balanced)...")

clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)

clf.fit(X_train, y_train)

# ===============================
# 9. Evaluate Model
# ===============================
y_pred = clf.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# ===============================
# 10. Save Model & Embedder
# ===============================
joblib.dump(clf, "model.pkl")
joblib.dump(embedder, "embedder.pkl")

print("\n✅ Model and embedder saved successfully!")
