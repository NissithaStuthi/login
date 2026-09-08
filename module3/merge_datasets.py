import pandas as pd
from pathlib import Path

# ============================================================
# MODULE 3 - CUSTOMER COMPLAINT DATA COLLECTION
# Merge and standardize Kaggle datasets
# ============================================================

# Paths
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"

CUSTOMER_SUPPORT_FILE = (
    RAW_DIR / "customer_support" / "customer_support_tickets.csv"
)

BANK_COMPLAINT_FILE = (
    RAW_DIR / "bank_complaints" / "bank_account_or_service_complaints.csv"
)

OUTPUT_FILE = RAW_DIR / "unified_customer_complaints_raw.csv"


# ------------------------------------------------------------
# 1. Load Customer Support Dataset
# ------------------------------------------------------------

print("Loading Customer Support dataset...")

support_df = pd.read_csv(CUSTOMER_SUPPORT_FILE)

print(f"Customer Support rows: {len(support_df):,}")


# ------------------------------------------------------------
# 2. Load Banking Complaint Dataset
# ------------------------------------------------------------

print("Loading Banking Complaint dataset...")

bank_df = pd.read_csv(BANK_COMPLAINT_FILE)

print(f"Bank Complaint rows: {len(bank_df):,}")


# ------------------------------------------------------------
# 3. Create common structure for Customer Support
# ------------------------------------------------------------

support_common = pd.DataFrame({
    "Complaint_ID": "CS_" + support_df["Ticket ID"].astype(str),
    "Customer_ID": pd.NA,
    "Complaint_Text": support_df["Ticket Description"],
    "Category": support_df["Ticket Type"],
    "Subcategory": support_df["Ticket Subject"],
    "Date": support_df["Date of Purchase"],
    "Channel": support_df["Ticket Channel"],
    "Product": support_df["Product Purchased"],
    "Sentiment": pd.NA,
    "Urgency": pd.NA,
    "Priority": support_df["Ticket Priority"],
    "Department": pd.NA,
    "Resolution": support_df["Resolution"],
    "Status": support_df["Ticket Status"],
    "Source_Dataset": "Customer Support Tickets"
})


# ------------------------------------------------------------
# 4. Create common structure for Banking Complaints
# ------------------------------------------------------------

bank_common = pd.DataFrame({
    "Complaint_ID": "BANK_" + bank_df["complaint_id"].astype(str),
    "Customer_ID": pd.NA,
    "Complaint_Text": bank_df["consumer_complaint_narrative"],
    "Category": bank_df["product"],
    "Subcategory": bank_df["sub_issue"],
    "Date": bank_df["date_received"],
    "Channel": bank_df["submitted_via"],
    "Product": bank_df["product"],
    "Sentiment": pd.NA,
    "Urgency": pd.NA,
    "Priority": pd.NA,
    "Department": pd.NA,
    "Resolution": bank_df["company_response_to_consumer"],
    "Status": pd.NA,
    "Source_Dataset": "Bank Complaints"
})


# ------------------------------------------------------------
# 5. Combine both datasets
# ------------------------------------------------------------

print("\nCombining datasets...")

unified_df = pd.concat(
    [support_common, bank_common],
    ignore_index=True
)


# ------------------------------------------------------------
# 6. Save unified raw dataset
# ------------------------------------------------------------

unified_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ------------------------------------------------------------
# 7. Display final results
# ------------------------------------------------------------

print("\n============================================")
print("MODULE 3 DATASET CREATION COMPLETE")
print("============================================")

print(f"Customer Support records : {len(support_common):,}")
print(f"Bank Complaint records   : {len(bank_common):,}")
print(f"Total records            : {len(unified_df):,}")

print(f"\nTotal columns            : {len(unified_df.columns)}")

print("\nCommon columns:")
for column in unified_df.columns:
    print(f" - {column}")

print(f"\nOutput file:")
print(OUTPUT_FILE)

print("\nFirst 5 records:")
print(unified_df.head().to_string(index=False))

print("\nDataset successfully created!")