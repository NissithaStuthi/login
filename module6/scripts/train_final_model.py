import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "module6" / "data" / "classification_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "module6" / "models" / "final_complaint_classifier.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "module6" / "models" / "final_tfidf_vectorizer.pkl"

df = pd.read_csv(DATA_PATH)

df["Complaint_Text"] = df["Complaint_Text"].fillna("").astype(str)

df = df[
    (df["Complaint_Text"].str.strip() != "") &
    (df["Target_Category"].notna())
].copy()

X = df["Complaint_Text"]
y = df["Target_Category"]

print("=" * 60)
print("MODULE 6 - FINAL COMPLAINT CLASSIFIER")
print("=" * 60)

print("\nTraining samples:", len(df))

# Best configuration found during tuning
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 1),
    min_df=2,
    max_features=50000,
    sublinear_tf=True
)

X_tfidf = vectorizer.fit_transform(X)

model = LinearSVC(
    C=2.0,
    class_weight="balanced"
)

print("\nTraining final Linear SVM...")
model.fit(X_tfidf, y)

joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print("\nFinal model saved to:")
print(MODEL_PATH)

print("\nFinal TF-IDF vectorizer saved to:")
print(VECTORIZER_PATH)

print("\n" + "=" * 60)
print("FINAL MODEL CREATED SUCCESSFULLY!")
print("=" * 60)
