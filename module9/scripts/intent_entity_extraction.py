import pandas as pd
import re
import joblib

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "module4"
    / "data"
    / "processed"
    / "processed_customer_complaints.csv"
)

DATA_DIR = PROJECT_ROOT / "module9" / "data"
MODEL_DIR = PROJECT_ROOT / "module9" / "models"
REPORT_DIR = PROJECT_ROOT / "module9" / "reports"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("MODULE 9 - CUSTOMER INTENT & ENTITY EXTRACTION")
print("=" * 75)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading Module 4 processed complaint data...")

df = pd.read_csv(INPUT_PATH)

print("Dataset shape:", df.shape)

df["Complaint_Text"] = (
    df["Complaint_Text"]
    .fillna("")
    .astype(str)
)

df = df[
    df["Complaint_Text"].str.strip() != ""
].copy()

df = df.drop_duplicates(
    subset=["Complaint_ID"]
).copy()

print("Usable complaints:", len(df))


# ============================================================
# CUSTOMER INTENT DEFINITIONS
# ============================================================

INTENTS = [
    "Refund Request",
    "Cancellation Request",
    "Payment Issue",
    "Account Recovery",
    "Delivery Tracking",
    "Product Replacement",
    "Complaint Escalation",
    "Technical Assistance",
    "Information Request"
]


# ============================================================
# SYNTHETIC INTENT TRAINING DATA
# ============================================================

intent_examples = {

    "Refund Request": [
        "I want a refund",
        "Please give me my money back",
        "I need a refund for my order",
        "When will my refund arrive",
        "My refund has not arrived",
        "I am waiting for my refund",
        "I want reimbursement",
        "How can I get my money back",
        "Refund is pending",
        "Please process my refund",
        "I have not received the refund",
        "Can you issue a refund",
        "I need my refund immediately",
        "The refunded amount is missing",
        "Refund request for my purchase"
    ],

    "Cancellation Request": [
        "I want to cancel my order",
        "Please cancel my order",
        "Cancel my purchase",
        "I need to cancel this order",
        "How do I cancel my order",
        "I want cancellation",
        "Please terminate my order",
        "Cancel the subscription",
        "I need to cancel my subscription",
        "Stop my order",
        "I no longer want this order",
        "Please cancel the transaction",
        "Can you cancel this purchase",
        "I want to withdraw my order",
        "Cancellation request"
    ],

    "Payment Issue": [
        "My payment failed",
        "Payment is not working",
        "I was charged twice",
        "Money was deducted",
        "My card payment failed",
        "Payment was declined",
        "I was charged incorrectly",
        "There is a problem with my payment",
        "The payment transaction failed",
        "Why was I charged",
        "My money was deducted",
        "Payment problem",
        "I cannot complete payment",
        "There is an issue with the transaction",
        "Payment has failed"
    ],

    "Account Recovery": [
        "I cannot login to my account",
        "I forgot my password",
        "Help me recover my account",
        "My account is locked",
        "I cannot access my account",
        "I lost access to my account",
        "How do I reset my password",
        "My account has been blocked",
        "I need account recovery",
        "I cannot sign in",
        "Please help me access my account",
        "I forgot my login details",
        "My account access is not working",
        "Recover my account",
        "Account login problem"
    ],

    "Delivery Tracking": [
        "Where is my order",
        "Track my delivery",
        "When will my order arrive",
        "My package has not arrived",
        "Where is my package",
        "I want to track my shipment",
        "Delivery is delayed",
        "Track my shipment",
        "My order is late",
        "When will the delivery arrive",
        "My parcel is missing",
        "Delivery tracking information",
        "What is the status of my order",
        "My shipment has not arrived",
        "Check my delivery status"
    ],

    "Product Replacement": [
        "I want a replacement",
        "My product is damaged",
        "The item is defective",
        "Please replace my product",
        "I received a damaged item",
        "I need a replacement product",
        "Can you replace this item",
        "The product is broken",
        "I received the wrong product",
        "I want to exchange the product",
        "Product replacement request",
        "My item is damaged and needs replacement",
        "The product does not work",
        "Please send another product",
        "I need an exchange"
    ],

    "Complaint Escalation": [
        "I want to escalate my complaint",
        "Please escalate this issue",
        "I want to speak to a manager",
        "This issue needs escalation",
        "I need a supervisor",
        "Please escalate my case",
        "I have complained many times",
        "Nobody has resolved my problem",
        "I want to raise this complaint",
        "Please forward this to management",
        "I need immediate escalation",
        "Escalate my case",
        "I am not satisfied with the response",
        "I want to file a serious complaint",
        "Please contact a senior representative"
    ],

    "Technical Assistance": [
        "The application is not working",
        "I am getting an error",
        "The website is broken",
        "I need technical help",
        "The app keeps crashing",
        "There is a technical problem",
        "I cannot use the application",
        "The system is showing an error",
        "Please fix this technical issue",
        "The website is not working",
        "I need assistance with the app",
        "There is a bug",
        "The application failed",
        "Help me solve this error",
        "Technical assistance required"
    ],

    "Information Request": [
        "I need more information",
        "How does this work",
        "Can you explain this",
        "What are the available options",
        "I have a question",
        "Can you provide information",
        "What is the process",
        "How can I change my address",
        "What are your services",
        "Tell me more about this",
        "I want to know more",
        "Can you give me details",
        "What is the procedure",
        "I need some information",
        "How do I do this"
    ]
}


# ============================================================
# PREPARE INTENT TRAINING DATA
# ============================================================

training_texts = []
training_labels = []

for intent, examples in intent_examples.items():

    for example in examples:

        training_texts.append(example)
        training_labels.append(intent)


training_df = pd.DataFrame({
    "text": training_texts,
    "intent": training_labels
})


print(
    "\nSynthetic intent training examples:",
    len(training_df)
)


# ============================================================
# INTENT MODEL
# ============================================================

intent_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])


# ============================================================
# TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    training_df["text"],
    training_df["intent"],
    test_size=0.20,
    random_state=42,
    stratify=training_df["intent"]
)


print("\nTraining intent classification model...")

intent_model.fit(
    X_train,
    y_train
)

print("Intent model trained.")


# ============================================================
# EVALUATION
# ============================================================

intent_predictions = intent_model.predict(X_test)

intent_accuracy = accuracy_score(
    y_test,
    intent_predictions
)

print("\n" + "=" * 75)
print("INTENT MODEL PERFORMANCE")
print("=" * 75)

print(
    f"Accuracy: {intent_accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        intent_predictions,
        zero_division=0
    )
)


# ============================================================
# SAVE INTENT MODEL
# ============================================================

MODEL_PATH = (
    MODEL_DIR
    / "customer_intent_classifier.pkl"
)

joblib.dump(
    intent_model,
    MODEL_PATH
)

print("\nIntent model saved:")
print(MODEL_PATH)


# ============================================================
# ENTITY EXTRACTION FUNCTIONS
# ============================================================


def extract_customer_name(text):

    patterns = [
        r"\bmy name is ([A-Za-z]+(?:\s+[A-Za-z]+){0,2})\b",
        r"\bi am ([A-Za-z]+(?:\s+[A-Za-z]+){0,2})\b",
        r"\bthis is ([A-Za-z]+(?:\s+[A-Za-z]+){0,2})\b"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            name = match.group(1).strip()

            # Avoid treating common phrases as names
            blocked = {
                "looking for",
                "unable to",
                "having trouble",
                "not happy",
                "waiting for"
            }

            if name.lower() not in blocked:
                return name

    return "Not Found"


# ------------------------------------------------------------
# ORDER ID
# ------------------------------------------------------------

def extract_order_id(text):

    patterns = [
        r"\bORD[- ]?[A-Z0-9]+\b",
        r"\border\s*(?:id|number|no\.?)?\s*[#: -]*([A-Z0-9-]{4,})\b"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            value = match.group(0)

            value = re.sub(
                r"(?i)^order\s*(id|number|no\.?)?\s*[#: -]*",
                "",
                value
            ).strip()

            if value.upper().startswith("ORD"):
                return value.upper()

            return value

    return "Not Found"


# ------------------------------------------------------------
# AMOUNT
# ------------------------------------------------------------

def extract_amount(text):

    patterns = [
        r"(?:₹|rs\.?|inr)\s?[\d,]+(?:\.\d{1,2})?",
        r"\b\d[\d,]*(?:\.\d{1,2})?\s?(?:rupees|rs)\b",
        r"\$\s?[\d,]+(?:\.\d{1,2})?",
        r"\b\d[\d,]*(?:\.\d{1,2})?\s?(?:dollars|usd)\b"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(0).strip()

    return "Not Found"


# ------------------------------------------------------------
# DATE
# ------------------------------------------------------------

def extract_date(text):

    patterns = [

        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b",

        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",

        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",

        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
        r"\.?\s+\d{1,2}(?:,\s*\d{4})?\b"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(0)

    return "Not Found"


# ------------------------------------------------------------
# TRANSACTION ID
# ------------------------------------------------------------

def extract_transaction_id(text):

    patterns = [
        r"\b(?:TXN|TRAN|TRX)[-_]?[A-Z0-9]+\b",
        r"\btransaction\s*(?:id|number|no\.?)?\s*[#: -]*([A-Z0-9-]{4,})\b"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            value = match.group(0)

            if re.match(
                r"(?i)^(txn|tran|trx)",
                value
            ):
                return value.upper()

            return value

    return "Not Found"


# ------------------------------------------------------------
# LOCATION
# ------------------------------------------------------------

def extract_location(text):

    patterns = [
        r"\b(?:in|at|from|near)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\b"
    ]

    blocked = {
        "my account",
        "the order",
        "the application",
        "this issue",
        "the website",
        "my product"
    }

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text
        )

        for match in matches:

            value = match.strip()

            if value.lower() not in blocked:
                return value

    return "Not Found"


# ------------------------------------------------------------
# ACCOUNT TYPE
# ------------------------------------------------------------

def extract_account_type(text):

    account_types = [
        "savings account",
        "current account",
        "business account",
        "student account",
        "personal account",
        "premium account",
        "credit card account",
        "bank account",
        "merchant account"
    ]

    text_lower = text.lower()

    for account_type in account_types:

        if account_type in text_lower:
            return account_type.title()

    return "Not Found"


# ------------------------------------------------------------
# PRODUCT
# ------------------------------------------------------------

def extract_product(text, row):

    # Prefer structured Product field from the dataset
    if "Product" in row.index:

        value = str(row["Product"])

        if (
            value
            and value.lower() not in [
                "nan",
                "unknown",
                "none"
            ]
        ):
            return value

    # Fallback product phrases
    patterns = [
        r"\bproduct\s*(?:is|called|named)?\s*[:#-]?\s*([A-Za-z0-9][A-Za-z0-9 _-]{1,40})",
        r"\bitem\s*(?:is|called|named)?\s*[:#-]?\s*([A-Za-z0-9][A-Za-z0-9 _-]{1,40})"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(1).strip()

    return "Not Found"


# ------------------------------------------------------------
# COMPLAINT CATEGORY
# ------------------------------------------------------------

def extract_category(row):

    # Prefer Module 6 target category if available
    if "Target_Category" in row.index:

        value = str(row["Target_Category"])

        if value.lower() not in [
            "nan",
            "unknown",
            "none"
        ]:
            return value

    if "Category" in row.index:

        value = str(row["Category"])

        if value.lower() not in [
            "nan",
            "unknown",
            "none"
        ]:
            return value

    return "Not Found"


# ============================================================
# INTENT PREDICTION
# ============================================================

print("\nPredicting customer intents...")

df["Predicted_Intent"] = (
    intent_model.predict(
        df["Complaint_Text"]
    )
)


# ============================================================
# ENTITY EXTRACTION
# ============================================================

print("Extracting customer entities...")

df["Customer_Name"] = df["Complaint_Text"].apply(
    extract_customer_name
)

df["Order_ID"] = df["Complaint_Text"].apply(
    extract_order_id
)

df["Amount"] = df["Complaint_Text"].apply(
    extract_amount
)

df["Date_Entity"] = df["Complaint_Text"].apply(
    extract_date
)

df["Transaction_ID"] = df["Complaint_Text"].apply(
    extract_transaction_id
)

df["Location"] = df["Complaint_Text"].apply(
    extract_location
)

df["Account_Type"] = df["Complaint_Text"].apply(
    extract_account_type
)

df["Product_Entity"] = [
    extract_product(
        row["Complaint_Text"],
        row
    )
    for _, row in df.iterrows()
]

df["Complaint_Category_Entity"] = [
    extract_category(row)
    for _, row in df.iterrows()
]


# ============================================================
# ENTITY EXTRACTION SUMMARY
# ============================================================

entity_columns = [
    "Customer_Name",
    "Order_ID",
    "Product_Entity",
    "Amount",
    "Date_Entity",
    "Transaction_ID",
    "Location",
    "Account_Type",
    "Complaint_Category_Entity"
]


entity_summary = []

for column in entity_columns:

    found = (
        df[column]
        .astype(str)
        .str.strip()
        .ne("Not Found")
        .sum()
    )

    percentage = (
        found
        / len(df)
        * 100
    )

    entity_summary.append({
        "Entity": column,
        "Found": found,
        "Percentage": round(
            percentage,
            2
        )
    })


entity_summary_df = pd.DataFrame(
    entity_summary
)


# ============================================================
# SAVE OUTPUT DATASET
# ============================================================

OUTPUT_COLUMNS = [
    "Complaint_ID",
    "Complaint_Text",
    "Predicted_Intent",
    "Customer_Name",
    "Order_ID",
    "Product_Entity",
    "Amount",
    "Date_Entity",
    "Transaction_ID",
    "Location",
    "Account_Type",
    "Complaint_Category_Entity",
    "Source_Dataset"
]

available_columns = [
    column
    for column in OUTPUT_COLUMNS
    if column in df.columns
]

OUTPUT_PATH = (
    DATA_DIR
    / "intent_entity_dataset.csv"
)

df[available_columns].to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nOutput dataset saved:")
print(OUTPUT_PATH)


# ============================================================
# INTENT DISTRIBUTION
# ============================================================

intent_distribution = (
    df["Predicted_Intent"]
    .value_counts()
)


intent_distribution_path = (
    DATA_DIR
    / "intent_distribution.csv"
)

intent_distribution.to_csv(
    intent_distribution_path,
    header=["Count"]
)


# ============================================================
# ENTITY SUMMARY
# ============================================================

ENTITY_SUMMARY_PATH = (
    REPORT_DIR
    / "entity_extraction_summary.csv"
)

entity_summary_df.to_csv(
    ENTITY_SUMMARY_PATH,
    index=False
)


# ============================================================
# TEST EXAMPLE
# ============================================================

print("\n" + "=" * 75)
print("MODULE 9 REQUIRED EXAMPLE TEST")
print("=" * 75)

example = (
    "I want a refund for order #ORD4567. "
    "₹2,500 was charged on August 28."
)

example_intent = intent_model.predict(
    [example]
)[0]

example_order = extract_order_id(
    example
)

example_amount = extract_amount(
    example
)

example_date = extract_date(
    example
)

print("\nInput:")
print(example)

print("\nExtracted Information:")

print(
    "Order ID:",
    example_order
)

print(
    "Amount:",
    example_amount
)

print(
    "Date:",
    example_date
)

print(
    "Intent:",
    example_intent
)


# ============================================================
# SAVE TEST RESULT
# ============================================================

example_result = pd.DataFrame([
    {
        "Input": example,
        "Order_ID": example_order,
        "Amount": example_amount,
        "Date": example_date,
        "Intent": example_intent
    }
])

example_result.to_csv(
    REPORT_DIR
    / "example_test_result.csv",
    index=False
)


# ============================================================
# GENERATE REPORT
# ============================================================

REPORT_PATH = (
    REPORT_DIR
    / "MODULE9_INTENT_ENTITY_REPORT.md"
)

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "# MODULE 9 – CUSTOMER INTENT & ENTITY EXTRACTION\n\n"
    )

    report.write(
        "## 1. Objective\n\n"
    )

    report.write(
        "The objective of this module is to identify what a "
        "customer wants from a complaint and extract important "
        "information from the complaint text.\n\n"
    )

    report.write(
        "## 2. Customer Intent Classes\n\n"
    )

    for intent in INTENTS:
        report.write(
            f"- {intent}\n"
        )

    report.write("\n")

    report.write(
        "## 3. Extracted Entities\n\n"
    )

    for entity in entity_columns:
        report.write(
            f"- {entity}\n"
        )

    report.write("\n")

    report.write(
        "## 4. Methodology\n\n"
    )

    report.write(
        "Customer intent is classified using a TF-IDF text "
        "representation combined with Logistic Regression. "
        "Entity extraction uses NLP-oriented pattern matching "
        "and structured information available in the complaint "
        "dataset.\n\n"
    )

    report.write(
        "## 5. Intent Model Performance\n\n"
    )

    report.write(
        f"Intent classification accuracy on the held-out "
        f"synthetic intent test set: "
        f"**{intent_accuracy * 100:.2f}%**.\n\n"
    )

    report.write(
        "The intent training examples are synthetic domain "
        "examples created from the required intent definitions. "
        "The accuracy therefore measures performance on this "
        "controlled intent dataset and should not be interpreted "
        "as human-annotated real-world benchmark accuracy.\n\n"
    )

    report.write(
        "## 6. Predicted Intent Distribution\n\n"
    )

    report.write(
        "| Intent | Complaints |\n"
        "|---|---:|\n"
    )

    for intent, count in intent_distribution.items():

        report.write(
            f"| {intent} | {count:,} |\n"
        )

    report.write("\n")

    report.write(
        "## 7. Entity Extraction Coverage\n\n"
    )

    report.write(
        "| Entity | Found | Coverage |\n"
        "|---|---:|---:|\n"
    )

    for _, row in entity_summary_df.iterrows():

        report.write(
            f"| {row['Entity']} | "
            f"{row['Found']:,} | "
            f"{row['Percentage']:.2f}% |\n"
        )

    report.write("\n")

    report.write(
        "## 8. Required Example\n\n"
    )

    report.write(
        f"**Input:** {example}\n\n"
    )

    report.write(
        f"- Order ID: **{example_order}**\n"
        f"- Amount: **{example_amount}**\n"
        f"- Date: **{example_date}**\n"
        f"- Intent: **{example_intent}**\n\n"
    )

    report.write(
        "## 9. Lab Activity\n\n"
        "An NLP-based customer intent and entity extraction "
        "system was developed and evaluated using test examples "
        "and a complaint dataset.\n\n"
    )

    report.write(
        "## 10. Deliverables\n\n"
        "- Customer intent classification model\n"
        "- Intent and entity extraction dataset\n"
        "- Entity extraction coverage report\n"
        "- Required example test\n"
        "- Module 9 technical report\n"
    )


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 75)
print("MODULE 9 COMPLETED SUCCESSFULLY!")
print("=" * 75)

print("\nFiles created:")

print(
    "1. Intent model:",
    MODEL_PATH
)

print(
    "2. Output dataset:",
    OUTPUT_PATH
)

print(
    "3. Entity summary:",
    ENTITY_SUMMARY_PATH
)

print(
    "4. Technical report:",
    REPORT_PATH
)

print(
    "\nCustomer intents:",
    len(INTENTS)
)

print(
    "Entities extracted:",
    len(entity_columns)
)

print(
    "\nMODULE 9 READY FOR TESTING!"
)
