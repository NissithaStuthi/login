import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# MODULE 6 - LINEAR SVM TUNING
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

X = df["Complaint_Text"]
y = df["Target_Category"]

print("=" * 60)
print("MODULE 6 - LINEAR SVM TUNING")
print("=" * 60)

print("\nDataset:", df.shape)

# Same split for fair comparison
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ============================================================
# TRY DIFFERENT TF-IDF SETTINGS
# ============================================================

tfidf_settings = [
    {
        "name": "Unigram",
        "ngram_range": (1, 1),
        "min_df": 2,
        "max_features": 50000
    },
    {
        "name": "Unigram + Bigram",
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_features": 50000
    },
    {
        "name": "Bigram",
        "ngram_range": (1, 2),
        "min_df": 1,
        "max_features": 100000
    }
]

C_values = [0.5, 1.0, 2.0]

results = []

best_score = 0
best_settings = None

for setting in tfidf_settings:

    print("\n" + "-" * 60)
    print("TF-IDF:", setting["name"])

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=setting["ngram_range"],
        min_df=setting["min_df"],
        max_features=setting["max_features"],
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    for C in C_values:

        print(f"Testing Linear SVM with C={C}...")

        model = LinearSVC(
            C=C,
            class_weight="balanced"
        )

        model.fit(X_train_tfidf, y_train)

        predictions = model.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(
            y_test,
            predictions,
            average="weighted"
        )

        results.append({
            "TFIDF": setting["name"],
            "C": C,
            "Accuracy": round(accuracy * 100, 2),
            "F1_Score": round(f1 * 100, 2)
        })

        print(
            f"Accuracy: {accuracy * 100:.2f}% | "
            f"F1: {f1 * 100:.2f}%"
        )

        if f1 > best_score:
            best_score = f1
            best_settings = {
                "TFIDF": setting,
                "C": C
            }

# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1_Score",
    ascending=False
)

print("\n" + "=" * 60)
print("SVM TUNING RESULTS")
print("=" * 60)

print(results_df.to_string(index=False))

print("\n" + "=" * 60)
print("BEST CONFIGURATION")
print("=" * 60)

print(
    f"\nBest TF-IDF: {best_settings['TFIDF']['name']}"
)

print(
    f"Best C value: {best_settings['C']}"
)

print(
    f"Best F1 Score: {best_score * 100:.2f}%"
)

best_row = results_df.iloc[0]

print(
    f"Best Accuracy: {best_row['Accuracy']:.2f}%"
)

# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    REPORT_PATH / "svm_tuning_results.csv",
    index=False
)

print("\nResults saved to:")
print(REPORT_PATH / "svm_tuning_results.csv")

print("\n" + "=" * 60)
print("SVM TUNING COMPLETED!")
print("=" * 60)