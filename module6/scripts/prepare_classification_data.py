import pandas as pd
import re
from pathlib import Path

# ============================================================
# MODULE 6 - PREPARE CLASSIFICATION DATA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "module4"
    / "data"
    / "processed"
    / "processed_customer_complaints.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "module6" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "classification_dataset.csv"

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("MODULE 6 - CLASSIFICATION DATA PREPARATION")
print("=" * 60)

print(f"\nOriginal dataset: {df.shape}")

# ============================================================
# COMBINE TEXT INFORMATION
# ============================================================

df["Complaint_Text"] = (
    df["Complaint_Text"]
    .fillna("")
    .astype(str)
)

df["Lemmatized_Text"] = (
    df["Lemmatized_Text"]
    .fillna("")
    .astype(str)
)

df["Model_Text"] = (
    df["Complaint_Text"]
    + " "
    + df["Lemmatized_Text"]
)

# ============================================================
# CATEGORY MAPPING
# ============================================================

def assign_category(row):

    text = str(row["Model_Text"]).lower()
    original = str(row["Category"]).lower()

    # --------------------------------------------------------
    # SECURITY
    # --------------------------------------------------------

    if re.search(
        r"\b("
        r"fraud|fraudulent|scam|scammed|unauthori[sz]ed|"
        r"stolen|hack|hacked|security|identity theft|"
        r"phishing|suspicious transaction"
        r")\b",
        text
    ):
        return "Security"

    # --------------------------------------------------------
    # REFUND
    # --------------------------------------------------------

    if re.search(
        r"\b("
        r"refund|reimbursement|money back|moneyback|cashback|"
        r"refund pending|refund status"
        r")\b",
        text
    ) or "refund" in original:
        return "Refund"

    # --------------------------------------------------------
    # PAYMENT
    # --------------------------------------------------------

    if re.search(
        r"\b("
        r"payment|paid|pay|transaction|charged|charge|"
        r"debit|deducted|payment failed|payment failure|"
        r"card payment"
        r")\b",
        text
    ):
        return "Payment"

    # --------------------------------------------------------
    # DELIVERY
    # --------------------------------------------------------

    if re.search(
        r"\b("
        r"delivery|delivered|shipping|shipment|courier|"
        r"arrived|dispatch|package|parcel|late delivery|"
        r"delivery delay|not delivered"
        r")\b",
        text
    ):
        return "Delivery"

    # --------------------------------------------------------
    # SUBSCRIPTION
    # --------------------------------------------------------

    if re.search(
        r"\b("
        r"subscription|subscribed|membership|renewal|"
        r"renewed|subscription plan|membership plan"
        r")\b",
        text
    ):
        return "Subscription"

    # --------------------------------------------------------
    # CANCELLATION
    # --------------------------------------------------------

    if re.search(
        r"\b("
        r"cancel|cancelled|cancellation|terminate|"
        r"termination|close account"
        r")\b",
        text
    ) or "cancellation" in original:
        return "Cancellation"

    # --------------------------------------------------------
    # TECHNICAL SUPPORT
    # --------------------------------------------------------

    if re.search(
        r"\b("
        r"error|bug|technical|crash|broken|unable|failure|"
        r"login|log in|sign in|signin|website|app|application|"
        r"system|not working|doesn't work|does not work|"
        r"cannot access|can't access|access problem|"
        r"password problem|technical issue"
        r")\b",
        text
    ) or "technical" in original:
        return "Technical Support"

    # --------------------------------------------------------
    # BILLING
    # --------------------------------------------------------

    if re.search(
        r"\b("
        r"bill|billing|invoice|overcharg|fee|fees|"
        r"statement|billing issue|billing problem"
        r")\b",
        text
    ) or "billing" in original:
        return "Billing"

    # --------------------------------------------------------
    # PRODUCT
    # --------------------------------------------------------

    if re.search(
        r"\b("
        r"product|item|damaged|defective|quality|replacement|"
        r"purchase|purchased|product issue|product problem"
        r")\b",
        text
    ) or "product" in original:
        return "Product"

    # --------------------------------------------------------
    # ACCOUNT
    # --------------------------------------------------------

    if re.search(
        r"\b("
        r"account|profile|username|password|address|"
        r"account access|account problem|account issue"
        r")\b",
        text
    ) or "account" in original:
        return "Account"

    # --------------------------------------------------------
    # GENERAL INQUIRY
    # --------------------------------------------------------

    return "General Inquiry"


# Apply category mapping

df["Target_Category"] = df.apply(
    assign_category,
    axis=1
)

# ============================================================
# REMOVE EMPTY TEXT
# ============================================================

df = df[
    df["Complaint_Text"].str.strip() != ""
].copy()

# ============================================================
# REMOVE DUPLICATE COMPLAINTS
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["Complaint_ID"]
).copy()

after_duplicates = len(df)

print(
    f"\nDuplicate Complaint IDs removed: "
    f"{before_duplicates - after_duplicates}"
)

# ============================================================
# CHECK CATEGORY DISTRIBUTION
# ============================================================

category_counts = (
    df["Target_Category"]
    .value_counts()
)

print("\n" + "=" * 60)
print("MAPPED CATEGORY DISTRIBUTION")
print("=" * 60)

print(category_counts)

print("\nMapped Category Percentages:")

print(
    (
        category_counts
        / len(df)
        * 100
    ).round(2)
)

# ============================================================
# BALANCE THE DATASET
# ============================================================

# Maximum number of records allowed for each category.
# Large categories are reduced while smaller categories
# keep all their available records.

MAX_PER_CATEGORY = 8000

balanced_parts = []

for category in sorted(
    df["Target_Category"].unique()
):

    category_data = df[
        df["Target_Category"] == category
    ].copy()

    original_count = len(category_data)

    if original_count > MAX_PER_CATEGORY:

        category_data = category_data.sample(
            n=MAX_PER_CATEGORY,
            random_state=42
        )

        print(
            f"\n{category}: "
            f"{original_count} -> "
            f"{MAX_PER_CATEGORY}"
        )

    else:

        print(
            f"\n{category}: "
            f"{original_count} -> "
            f"{original_count}"
        )

    balanced_parts.append(
        category_data
    )

# Combine all categories

balanced_df = pd.concat(
    balanced_parts,
    ignore_index=True
)

# ============================================================
# SHUFFLE DATASET
# ============================================================

balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# ============================================================
# FINAL CATEGORY DISTRIBUTION
# ============================================================

final_counts = (
    balanced_df["Target_Category"]
    .value_counts()
)

print("\n" + "=" * 60)
print("FINAL BALANCED CATEGORY DISTRIBUTION")
print("=" * 60)

print(final_counts)

print("\nFinal Category Percentages:")

print(
    (
        final_counts
        / len(balanced_df)
        * 100
    ).round(2)
)

# ============================================================
# SAVE CLASSIFICATION DATASET
# ============================================================

output_columns = [
    "Complaint_ID",
    "Complaint_Text",
    "Lemmatized_Text",
    "Target_Category",
    "Source_Dataset"
]

balanced_df[
    output_columns
].to_csv(
    OUTPUT_PATH,
    index=False
)

# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL CLASSIFICATION DATASET")
print("=" * 60)

print(
    f"\nOriginal dataset size: "
    f"{len(pd.read_csv(INPUT_PATH)):,}"
)

print(
    f"Final dataset size: "
    f"{len(balanced_df):,}"
)

print(
    f"Number of categories: "
    f"{balanced_df['Target_Category'].nunique()}"
)

print("\nCategories:")

for category in sorted(
    balanced_df["Target_Category"].unique()
):
    print(
        f"  - {category}"
    )

print("\nSaved to:")

print(OUTPUT_PATH)

print("\n" + "=" * 60)
print("CLASSIFICATION DATASET PREPARED SUCCESSFULLY!")
print("=" * 60)