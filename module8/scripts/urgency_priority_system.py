import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODULE4_PATH = (
    PROJECT_ROOT
    / "module4"
    / "data"
    / "processed"
    / "processed_customer_complaints.csv"
)

MODULE7_PATH = (
    PROJECT_ROOT
    / "module7"
    / "data"
    / "sentiment_emotion_dataset.csv"
)

DATA_DIR = PROJECT_ROOT / "module8" / "data"
MODEL_DIR = PROJECT_ROOT / "module8" / "models"
REPORT_DIR = PROJECT_ROOT / "module8" / "reports"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MODULE 8 - COMPLAINT URGENCY & PRIORITY DETECTION")
print("=" * 70)

print("\nLoading Module 4 data...")

df = pd.read_csv(MODULE4_PATH)

print("Module 4 dataset:", df.shape)


# ------------------------------------------------------------
# LOAD MODULE 7 SENTIMENT / EMOTION
# ------------------------------------------------------------

print("\nLoading Module 7 sentiment and emotion data...")

try:
    sentiment_df = pd.read_csv(MODULE7_PATH)

    sentiment_columns = [
        "Complaint_ID",
        "Predicted_Sentiment",
        "Predicted_Emotion",
        "Emotion_Intensity"
    ]

    available = [
        col for col in sentiment_columns
        if col in sentiment_df.columns
    ]

    sentiment_df = sentiment_df[available].drop_duplicates(
        subset=["Complaint_ID"]
    )

    df = df.merge(
        sentiment_df,
        on="Complaint_ID",
        how="left"
    )

    print("Module 7 data merged successfully.")

except Exception as error:

    print("Module 7 data could not be merged.")
    print("Reason:", error)

    df["Predicted_Sentiment"] = "Neutral"
    df["Predicted_Emotion"] = "Neutral"
    df["Emotion_Intensity"] = "Neutral"


# ============================================================
# BASIC CLEANING
# ============================================================

df["Complaint_Text"] = (
    df["Complaint_Text"]
    .fillna("")
    .astype(str)
)

df["Category"] = (
    df["Category"]
    .fillna("Unknown")
    .astype(str)
)

df["Predicted_Sentiment"] = (
    df["Predicted_Sentiment"]
    .fillna("Neutral")
    .astype(str)
)

df["Predicted_Emotion"] = (
    df["Predicted_Emotion"]
    .fillna("Neutral")
    .astype(str)
)

df["Emotion_Intensity"] = (
    df["Emotion_Intensity"]
    .fillna("Neutral")
    .astype(str)
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def count_keywords(text, keywords):

    text = text.lower()

    return sum(
        1 for keyword in keywords
        if re.search(
            r"\b" + re.escape(keyword) + r"\b",
            text
        )
    )


# ------------------------------------------------------------
# URGENCY KEYWORDS
# ------------------------------------------------------------

urgency_keywords = [
    "urgent",
    "urgently",
    "immediately",
    "asap",
    "emergency",
    "critical",
    "important",
    "priority",
    "immediate",
    "right away"
]


# ------------------------------------------------------------
# FINANCIAL IMPACT
# ------------------------------------------------------------

financial_keywords = [
    "money",
    "payment",
    "paid",
    "charge",
    "charged",
    "transaction",
    "debit",
    "deducted",
    "refund",
    "fraud",
    "fraudulent",
    "financial",
    "cash",
    "amount",
    "bank",
    "credit",
    "loss",
    "lost money"
]


# ------------------------------------------------------------
# ACCOUNT STATUS
# ------------------------------------------------------------

account_keywords = [
    "account",
    "login",
    "log in",
    "signin",
    "sign in",
    "password",
    "blocked",
    "locked",
    "access",
    "accessed",
    "hacked",
    "compromised",
    "unauthorized"
]


# ------------------------------------------------------------
# SEVERE IMPACT
# ------------------------------------------------------------

severe_keywords = [
    "fraud",
    "fraudulent",
    "unauthorized",
    "stolen",
    "hacked",
    "hack",
    "compromised",
    "identity theft",
    "security breach",
    "scam",
    "scammed",
    "threat",
    "danger"
]


# ------------------------------------------------------------
# CUSTOMER IMPACT
# ------------------------------------------------------------

customer_impact_keywords = [
    "cannot",
    "can't",
    "unable",
    "blocked",
    "failed",
    "failure",
    "not working",
    "doesn't work",
    "does not work",
    "lost",
    "delay",
    "delayed",
    "waiting",
    "problem",
    "issue"
]


df["Urgency_Keyword_Count"] = df["Complaint_Text"].apply(
    lambda text: count_keywords(
        text,
        urgency_keywords
    )
)


df["Financial_Impact"] = df["Complaint_Text"].apply(
    lambda text: int(
        count_keywords(
            text,
            financial_keywords
        ) > 0
    )
)


df["Account_Status"] = df["Complaint_Text"].apply(
    lambda text: (
        "Compromised"
        if count_keywords(text, severe_keywords) > 0
        else (
            "Affected"
            if count_keywords(text, account_keywords) > 0
            else "Normal"
        )
    )
)


df["Customer_Impact"] = df["Complaint_Text"].apply(
    lambda text: (
        "Severe"
        if count_keywords(text, severe_keywords) > 0
        else (
            "High"
            if count_keywords(text, customer_impact_keywords) >= 2
            else (
                "Medium"
                if count_keywords(text, customer_impact_keywords) == 1
                else "Low"
            )
        )
    )
)


# ============================================================
# COMPLAINT HISTORY
# ============================================================

if "Customer_ID" in df.columns:

    df["Customer_ID"] = (
        df["Customer_ID"]
        .fillna("Unknown")
        .astype(str)
    )

    customer_counts = (
        df.groupby("Customer_ID")["Complaint_ID"]
        .transform("count")
    )

    df["Complaint_History_Count"] = customer_counts

else:

    df["Complaint_History_Count"] = 1


# ============================================================
# RULE-BASED URGENCY LABEL
# ============================================================

def determine_urgency(row):

    text = str(row["Complaint_Text"]).lower()

    category = str(row["Category"]).lower()

    sentiment = str(
        row["Predicted_Sentiment"]
    ).lower()

    emotion = str(
        row["Predicted_Emotion"]
    ).lower()

    # --------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------

    critical_patterns = [
        "fraud",
        "fraudulent",
        "unauthorized",
        "unauthorised",
        "hacked",
        "hack",
        "compromised",
        "identity theft",
        "stolen",
        "security breach",
        "scam",
        "scammed",
        "someone accessed my account",
        "someone has accessed my account"
    ]

    if any(
        pattern in text
        for pattern in critical_patterns
    ):
        return "Critical"

    if row["Account_Status"] == "Compromised":
        return "Critical"

    if (
        row["Financial_Impact"] == 1
        and (
            "fraud" in text
            or "unauthorized" in text
            or "unauthorised" in text
            or "stolen" in text
        )
    ):
        return "Critical"


    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------

    high_patterns = [
        "urgent",
        "urgently",
        "immediately",
        "asap",
        "emergency",
        "cannot access",
        "can't access",
        "unable to access",
        "account locked",
        "account blocked",
        "payment failed",
        "payment failure",
        "money deducted",
        "money was deducted",
        "transaction failed",
        "refund pending",
        "refund not received"
    ]

    if any(
        pattern in text
        for pattern in high_patterns
    ):
        return "High"

    if emotion in [
        "anger",
        "frustration"
    ] and sentiment == "negative":
        return "High"

    if row["Customer_Impact"] == "Severe":
        return "High"


    # --------------------------------------------------------
    # MEDIUM
    # --------------------------------------------------------

    if sentiment == "negative":
        return "Medium"

    if row["Customer_Impact"] == "High":
        return "Medium"

    if row["Financial_Impact"] == 1:
        return "Medium"

    if (
        "problem" in text
        or "issue" in text
        or "complaint" in text
        or "error" in text
    ):
        return "Medium"


    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------

    return "Low"


df["Urgency_Label"] = df.apply(
    determine_urgency,
    axis=1
)


# ============================================================
# PRIORITY MAPPING
# ============================================================

priority_mapping = {
    "Critical": "P1",
    "High": "P2",
    "Medium": "P3",
    "Low": "P4"
}

df["Priority"] = df["Urgency_Label"].map(
    priority_mapping
)


# ============================================================
# DEPARTMENT ROUTING
# ============================================================

def determine_department(row):

    text = str(
        row["Complaint_Text"]
    ).lower()

    category = str(
        row["Category"]
    ).lower()

    if any(
        word in text
        for word in [
            "fraud",
            "fraudulent",
            "unauthorized",
            "unauthorised",
            "hacked",
            "scam",
            "security",
            "identity theft",
            "stolen"
        ]
    ):
        return "Security & Fraud"

    if (
        "payment" in category
        or "payment" in text
        or "transaction" in text
        or "charged" in text
        or "refund" in text
    ):
        return "Payments & Refunds"

    if (
        "account" in category
        or "login" in text
        or "password" in text
        or "account" in text
    ):
        return "Account Support"

    if (
        "technical" in category
        or "error" in text
        or "bug" in text
        or "website" in text
        or "application" in text
        or "app" in text
    ):
        return "Technical Support"

    if (
        "delivery" in category
        or "delivery" in text
        or "shipping" in text
        or "courier" in text
    ):
        return "Delivery Support"

    if (
        "product" in category
        or "product" in text
        or "damaged" in text
        or "defective" in text
    ):
        return "Product Support"

    return "Customer Support"


df["Department"] = df.apply(
    determine_department,
    axis=1
)


# ============================================================
# PREPARE ML DATA
# ============================================================

feature_columns = [
    "Category",
    "Predicted_Sentiment",
    "Predicted_Emotion",
    "Emotion_Intensity",
    "Urgency_Keyword_Count",
    "Customer_Impact",
    "Financial_Impact",
    "Account_Status",
    "Complaint_History_Count"
]

X = df[feature_columns].copy()
y = df["Urgency_Label"].copy()


categorical_features = [
    "Category",
    "Predicted_Sentiment",
    "Predicted_Emotion",
    "Emotion_Intensity",
    "Customer_Impact",
    "Account_Status"
]

numeric_features = [
    "Urgency_Keyword_Count",
    "Financial_Impact",
    "Complaint_History_Count"
]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            model
        )
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining Random Forest urgency model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# EVALUATION
# ============================================================

predictions = pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH = (
    MODEL_DIR
    / "urgency_priority_random_forest.pkl"
)

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\nModel saved to:")
print(MODEL_PATH)


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

OUTPUT_COLUMNS = [
    "Complaint_ID",
    "Complaint_Text",
    "Category",
    "Predicted_Sentiment",
    "Predicted_Emotion",
    "Emotion_Intensity",
    "Urgency_Keyword_Count",
    "Customer_Impact",
    "Financial_Impact",
    "Account_Status",
    "Complaint_History_Count",
    "Urgency_Label",
    "Priority",
    "Department"
]

available_output_columns = [
    column
    for column in OUTPUT_COLUMNS
    if column in df.columns
]

OUTPUT_DATA_PATH = (
    DATA_DIR
    / "urgency_priority_dataset.csv"
)

df[available_output_columns].to_csv(
    OUTPUT_DATA_PATH,
    index=False
)

print("\nDataset saved to:")
print(OUTPUT_DATA_PATH)


# ============================================================
# URGENCY DISTRIBUTION
# ============================================================

urgency_counts = (
    df["Urgency_Label"]
    .value_counts()
    .reindex(
        [
            "Critical",
            "High",
            "Medium",
            "Low"
        ],
        fill_value=0
    )
)

plt.figure(figsize=(8, 5))

urgency_counts.plot(
    kind="bar"
)

plt.title(
    "Complaint Urgency Distribution"
)

plt.xlabel("Urgency Level")
plt.ylabel("Number of Complaints")

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    REPORT_DIR
    / "01_urgency_distribution.png"
)

plt.close()


# ============================================================
# PRIORITY DISTRIBUTION
# ============================================================

priority_counts = (
    df["Priority"]
    .value_counts()
    .reindex(
        [
            "P1",
            "P2",
            "P3",
            "P4"
        ],
        fill_value=0
    )
)

plt.figure(figsize=(8, 5))

priority_counts.plot(
    kind="bar"
)

plt.title(
    "Complaint Priority Distribution"
)

plt.xlabel("Priority")
plt.ylabel("Number of Complaints")

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    REPORT_DIR
    / "02_priority_distribution.png"
)

plt.close()


# ============================================================
# DEPARTMENT DISTRIBUTION
# ============================================================

department_counts = (
    df["Department"]
    .value_counts()
)

plt.figure(figsize=(10, 6))

department_counts.plot(
    kind="bar"
)

plt.title(
    "Complaint Department Distribution"
)

plt.xlabel("Department")
plt.ylabel("Number of Complaints")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    REPORT_DIR
    / "03_department_distribution.png"
)

plt.close()


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = [
    "Critical",
    "High",
    "Medium",
    "Low"
]

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title(
    "Urgency Classification Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.xticks(
    range(len(labels)),
    labels
)

plt.yticks(
    range(len(labels)),
    labels
)

for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    REPORT_DIR
    / "04_confusion_matrix.png"
)

plt.close()


# ============================================================
# GENERATE REPORT
# ============================================================

REPORT_PATH = (
    REPORT_DIR
    / "MODULE8_URGENCY_PRIORITY_REPORT.md"
)

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "# MODULE 8 – COMPLAINT URGENCY & PRIORITY DETECTION\n\n"
    )

    report.write(
        "## 1. Objective\n\n"
    )

    report.write(
        "The objective of this module is to determine how urgently "
        "a customer complaint should be handled and assign an "
        "appropriate priority level.\n\n"
    )

    report.write(
        "## 2. Urgency Levels\n\n"
    )

    report.write(
        "- **Critical:** Security, financial loss, account compromise "
        "or severe service disruption.\n"
        "- **High:** Issue requiring quick intervention.\n"
        "- **Medium:** Issue affecting customer experience.\n"
        "- **Low:** General questions or minor issues.\n\n"
    )

    report.write(
        "## 3. Priority Mapping\n\n"
    )

    report.write(
        "| Priority | Urgency | Meaning |\n"
        "|---|---|---|\n"
        "| P1 | Critical | Immediate attention |\n"
        "| P2 | High | Quick intervention |\n"
        "| P3 | Medium | Normal priority |\n"
        "| P4 | Low | Low priority |\n\n"
    )

    report.write(
        "## 4. Input Features\n\n"
    )

    for feature in feature_columns:
        report.write(
            f"- {feature}\n"
        )

    report.write("\n")

    report.write(
        "## 5. AI Methodology\n\n"
    )

    report.write(
        "The system combines rule-based domain knowledge with a "
        "Random Forest machine learning classifier. Sentiment and "
        "emotion information from Module 7 are incorporated into "
        "the urgency prediction features.\n\n"
    )

    report.write(
        "The urgency labels are generated using transparent "
        "domain rules based on security risk, financial impact, "
        "customer impact, sentiment, emotion and urgency keywords. "
        "The Random Forest model learns these patterns and provides "
        "the final ML urgency prediction.\n\n"
    )

    report.write(
        "## 6. Model Performance\n\n"
    )

    report.write(
        f"- Random Forest accuracy: **{accuracy * 100:.2f}%**\n"
        "- Training/test split: 80/20\n"
        "- Number of estimators: 200\n"
        "- Random state: 42\n\n"
    )

    report.write(
        "## 7. Urgency Distribution\n\n"
    )

    for level, count in urgency_counts.items():

        percentage = (
            count
            / len(df)
            * 100
        )

        report.write(
            f"- **{level}:** {count:,} "
            f"({percentage:.2f}%)\n"
        )

    report.write(
        "\n## 8. Priority Distribution\n\n"
    )

    for priority, count in priority_counts.items():

        percentage = (
            count
            / len(df)
            * 100
        )

        report.write(
            f"- **{priority}:** {count:,} "
            f"({percentage:.2f}%)\n"
        )

    report.write(
        "\n## 9. Department Routing\n\n"
    )

    for department, count in department_counts.items():

        report.write(
            f"- **{department}:** {count:,}\n"
        )

    report.write(
        "\n## 10. Example\n\n"
    )

    report.write(
        "**Complaint:**\n\n"
        "> Someone has accessed my account and made an "
        "unauthorized transaction.\n\n"
        "**Expected Urgency:** Critical\n\n"
        "**Expected Priority:** P1\n\n"
        "**Expected Department:** Security & Fraud\n\n"
    )

    report.write(
        "## 11. Output Files\n\n"
        "- Urgency and priority prediction dataset\n"
        "- Random Forest ML model\n"
        "- Urgency distribution chart\n"
        "- Priority distribution chart\n"
        "- Department distribution chart\n"
        "- Confusion matrix\n"
        "- Module 8 report\n"
    )


# ============================================================
# TEST EXAMPLE
# ============================================================

print("\n" + "=" * 70)
print("TEST EXAMPLE")
print("=" * 70)

test_complaint = pd.DataFrame([
    {
        "Category": "Security",
        "Predicted_Sentiment": "Negative",
        "Predicted_Emotion": "Fear",
        "Emotion_Intensity": "Very Negative",
        "Urgency_Keyword_Count": 0,
        "Customer_Impact": "Severe",
        "Financial_Impact": 1,
        "Account_Status": "Compromised",
        "Complaint_History_Count": 1
    }
])

test_prediction = pipeline.predict(
    test_complaint
)[0]

test_priority = priority_mapping[
    test_prediction
]

print(
    "Complaint: Someone has accessed my account "
    "and made an unauthorized transaction."
)

print(
    "Predicted Urgency:",
    test_prediction
)

print(
    "Predicted Priority:",
    test_priority
)

print(
    "Department: Security & Fraud"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("MODULE 8 COMPLETED SUCCESSFULLY!")
print("=" * 70)

print("\nFiles created:")
print("1.", OUTPUT_DATA_PATH)
print("2.", MODEL_PATH)
print("3.", REPORT_PATH)
print("4. Four analysis charts")