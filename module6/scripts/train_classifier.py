import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# MODULE 6 - COMPLAINT CLASSIFICATION MODEL
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "module6"
    / "data"
    / "classification_dataset.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "module6"
    / "models"
    / "complaint_classifier.pkl"
)

VECTORIZER_PATH = (
    PROJECT_ROOT
    / "module6"
    / "models"
    / "tfidf_vectorizer.pkl"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "module6"
    / "reports"
)

REPORT_PATH.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

df["Complaint_Text"] = df["Complaint_Text"].fillna("").astype(str)

df = df[
    (df["Complaint_Text"].str.strip() != "") &
    (df["Target_Category"].notna())
].copy()

print("=" * 60)
print("MODULE 6 - MACHINE LEARNING CLASSIFICATION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nCategory Distribution:")
print(df["Target_Category"].value_counts())

# ============================================================
# INPUT AND TARGET
# ============================================================

X = df["Complaint_Text"]
y = df["Target_Category"]

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ============================================================
# TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF Training Shape:")
print(X_train_tfidf.shape)

# ============================================================
# LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

print("\nTraining Logistic Regression model...")

model.fit(X_train_tfidf, y_train)

print("Model training completed!")

# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test_tfidf)

# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print("\nModel saved to:")
print(MODEL_PATH)

print("\nTF-IDF vectorizer saved to:")
print(VECTORIZER_PATH)

# ============================================================
# SAVE PERFORMANCE REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

with open(
    REPORT_PATH / "classification_model_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("MODULE 6 - COMPLAINT CLASSIFICATION MODEL\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total Dataset Samples: {len(df)}\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Testing Samples: {len(X_test)}\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("\nPerformance report saved successfully!")

print("\n" + "=" * 60)
print("MODULE 6 MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)