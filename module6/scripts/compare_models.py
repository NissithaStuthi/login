import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# MODULE 6 - MODEL COMPARISON
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "module6"
    / "data"
    / "classification_dataset.csv"
)

REPORT_PATH = PROJECT_ROOT / "module6" / "reports"
REPORT_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

df["Complaint_Text"] = df["Complaint_Text"].fillna("").astype(str)

df = df[
    (df["Complaint_Text"].str.strip() != "") &
    (df["Target_Category"].notna())
].copy()

print("=" * 60)
print("MODULE 6 - ML MODEL COMPARISON")
print("=" * 60)

print("\nDataset:", df.shape)

X = df["Complaint_Text"]
y = df["Target_Category"]

# Same split for every model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ============================================================
# TF-IDF
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF ready!")

# ============================================================
# MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Naive Bayes": MultinomialNB(),

    "Linear SVM": LinearSVC(
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
}

results = []

# ============================================================
# TRAIN AND COMPARE
# ============================================================

for name, model in models.items():

    print("\n" + "-" * 60)
    print("Training:", name)

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    results.append({
        "Model": name,
        "Accuracy": round(accuracy * 100, 2),
        "F1_Score": round(f1 * 100, 2)
    })

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Weighted F1 Score: {f1 * 100:.2f}%")

# ============================================================
# COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1_Score",
    ascending=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)

print(results_df.to_string(index=False))

# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    REPORT_PATH / "model_comparison.csv",
    index=False
)

print("\nComparison saved to:")
print(REPORT_PATH / "model_comparison.csv")

print("\n" + "=" * 60)
print("MODEL COMPARISON COMPLETED!")
print("=" * 60)