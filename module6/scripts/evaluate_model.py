import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "module6" / "data" / "classification_dataset.csv"
REPORT_PATH = PROJECT_ROOT / "module6" / "reports" / "classification_report.txt"
CM_PATH = PROJECT_ROOT / "module6" / "reports" / "confusion_matrix.csv"

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("MODULE 6 - MODEL EVALUATION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

df["Complaint_Text"] = df["Complaint_Text"].fillna("").astype(str)

df = df[
    (df["Complaint_Text"].str.strip() != "") &
    (df["Target_Category"].notna())
].copy()

X = df["Complaint_Text"]
y = df["Target_Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 1),
    min_df=2,
    max_features=50000,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LinearSVC(
    C=2.0,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

cm_df = pd.DataFrame(
    cm,
    index=model.classes_,
    columns=model.classes_
)

cm_df.to_csv(CM_PATH)

full_report = f"""
MODULE 6 - COMPLAINT CLASSIFICATION MODEL EVALUATION
====================================================

Dataset Size: {len(df)}
Training Samples: {len(X_train)}
Testing Samples: {len(X_test)}

Algorithm: Linear Support Vector Machine (LinearSVC)

TF-IDF Configuration:
- N-gram Range: (1,1)
- Minimum Document Frequency: 2
- Maximum Features: 50,000
- Stop Words: English
- Sublinear TF: Enabled

SVM Configuration:
- C: 2.0
- Class Weight: balanced

Accuracy: {accuracy * 100:.2f}%

Classification Report:
{report}
"""

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(full_report)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(report)

print(f"\nReport saved to:")
print(REPORT_PATH)

print(f"\nConfusion matrix saved to:")
print(CM_PATH)

print("\n" + "=" * 60)
print("MODEL EVALUATION COMPLETED")
print("=" * 60)