from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "module10"
    / "data"
    / "generated_customer_responses.csv"
)

OUTPUT_DATA_PATH = (
    PROJECT_ROOT
    / "module11"
    / "data"
    / "routing_resolution_dataset.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "module11"
    / "reports"
    / "MODULE11_ROUTING_RESOLUTION_REPORT.md"
)


# ============================================================
# DEPARTMENT ROUTING
# ============================================================

DEPARTMENT_RULES = {
    "Payment": "Payments",
    "Refund": "Finance",
    "Delivery": "Logistics",
    "Product": "Product Support",
    "Technical Support": "Technical Support",
    "Account": "Account Support",
    "Security": "Security",
    "Subscription": "Billing",
    "Billing": "Billing",
    "Cancellation": "Customer Support",
    "General Inquiry": "Customer Support",
}


# ============================================================
# RESOLUTION RECOMMENDATIONS
# ============================================================

RESOLUTION_RULES = {
    "Payment": "Verify the payment transaction and confirm whether the order was successfully created. Initiate a refund if required.",
    
    "Refund": "Verify the refund status and transaction details. Process or escalate the refund according to the payment records.",
    
    "Delivery": "Check the shipment and delivery status. Contact the logistics team and arrange redelivery or appropriate action.",
    
    "Product": "Verify the product issue and arrange replacement, repair, or return according to the product support policy.",
    
    "Technical Support": "Verify the technical problem, guide the customer through troubleshooting, and escalate to technical support when necessary.",
    
    "Account": "Verify the customer account and assist with account-related changes, recovery, or access issues.",
    
    "Security": "Immediately secure the customer account, verify the suspicious activity, and escalate the case to the security team.",
    
    "Subscription": "Verify the subscription status and process the requested subscription change, cancellation, or billing adjustment.",
    
    "Billing": "Verify billing and transaction records and correct any duplicate, incorrect, or pending charges.",
    
    "Cancellation": "Verify the cancellation request and process the cancellation according to the applicable policy.",
    
    "General Inquiry": "Review the customer's request and provide the appropriate information or assistance.",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_category(category):
    """
    Convert different category names into the standard
    categories used by the routing system.
    """

    if pd.isna(category):
        return "General Inquiry"

    category = str(category).strip()

    category_lower = category.lower()

    if "security" in category_lower:
        return "Security"

    if "refund" in category_lower:
        return "Refund"

    if "payment" in category_lower:
        return "Payment"

    if "delivery" in category_lower:
        return "Delivery"

    if "technical" in category_lower or "login" in category_lower:
        return "Technical Support"

    if "account" in category_lower:
        return "Account"

    if "subscription" in category_lower:
        return "Subscription"

    if "billing" in category_lower:
        return "Billing"

    if "cancel" in category_lower:
        return "Cancellation"

    if "product" in category_lower:
        return "Product"

    return "General Inquiry"


def predict_department(category, intent="", complaint_text=""):
    """
    Determine the appropriate department using category,
    intent, and complaint text.
    """

    category = normalize_category(category)

    text = f"{intent} {complaint_text}".lower()

    # High-priority security guardrail
    security_keywords = [
        "unauthorized",
        "hacked",
        "hack",
        "fraud",
        "stolen",
        "suspicious transaction",
        "someone accessed",
        "account compromised",
        "security",
    ]

    if any(keyword in text for keyword in security_keywords):
        return "Security"

    # Refund guardrail
    if "refund" in text:
        return "Finance"

    # Payment guardrail
    if "payment" in text or "charged" in text or "transaction" in text:
        if "refund" not in text:
            return "Payments"

    return DEPARTMENT_RULES.get(
        category,
        "Customer Support"
    )


def recommend_resolution(
    category,
    intent="",
    complaint_text="",
    priority="",
):
    """
    Recommend a suitable resolution based on the complaint.
    """

    category = normalize_category(category)

    text = f"{intent} {complaint_text}".lower()

    # Security complaints
    security_keywords = [
        "unauthorized",
        "fraud",
        "hacked",
        "stolen",
        "suspicious transaction",
        "account compromised",
        "someone accessed",
    ]

    if any(keyword in text for keyword in security_keywords):
        return (
            "Secure the customer account immediately, "
            "verify the suspicious transaction, and escalate "
            "the complaint to the security team."
        )

    # Refund
    if "refund" in text:
        return (
            "Verify the original transaction and refund status, "
            "then process or escalate the refund request."
        )

    # Duplicate charge
    if (
        "charged twice" in text
        or "duplicate charge" in text
        or "charged two times" in text
    ):
        return (
            "Verify the duplicate transaction and refund the "
            "incorrect additional charge."
        )

    # Login / password
    if (
        "login" in text
        or "log in" in text
        or "password" in text
        or "cannot access my account" in text
    ):
        return (
            "Verify the account and assist the customer with "
            "password reset or account recovery."
        )

    # Cancellation
    if "cancel" in text:
        return (
            "Verify the cancellation request and process the "
            "cancellation according to the applicable policy."
        )

    # Delivery
    if (
        "delivery" in text
        or "delivered" in text
        or "shipment" in text
        or "tracking" in text
    ):
        return (
            "Check the shipment status and coordinate with the "
            "logistics team for delivery or redelivery."
        )

    # Product damage / replacement
    if (
        "damaged" in text
        or "broken" in text
        or "replacement" in text
        or "defective" in text
    ):
        return (
            "Verify the product issue and arrange replacement, "
            "return, or repair as appropriate."
        )

    # Payment
    if "payment" in text or "payment failed" in text:
        return (
            "Verify the payment transaction and confirm the order "
            "status. Initiate a refund or retry the transaction "
            "when necessary."
        )

    # Default category resolution
    return RESOLUTION_RULES.get(
        category,
        "Review the complaint and provide the appropriate customer support assistance."
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print("=" * 60)
    print("MODULE 11 - COMPLAINT ROUTING & RESOLUTION")
    print("=" * 60)

    print(f"Input file: {INPUT_PATH}")
    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")

    return df


# ============================================================
# PROCESS DATA
# ============================================================

def process_data(df):

    # Required columns with safe defaults
    for column in [
        "Complaint_Text",
        "Category",
        "Target_Category",
        "Sentiment",
        "Urgency",
        "Priority",
        "Intent",
        "Recommended_Resolution",
    ]:
        if column not in df.columns:
            df[column] = ""

    # Select the best available category
    df["Routing_Category"] = df["Target_Category"].fillna("")

    empty_mask = (
        df["Routing_Category"].astype(str).str.strip() == ""
    )

    df.loc[empty_mask, "Routing_Category"] = (
        df.loc[empty_mask, "Category"]
    )

    df["Routing_Category"] = (
        df["Routing_Category"]
        .apply(normalize_category)
    )

    # Department routing
    df["Recommended_Department"] = df.apply(
        lambda row: predict_department(
            category=row["Routing_Category"],
            intent=row.get("Intent", ""),
            complaint_text=row.get("Complaint_Text", ""),
        ),
        axis=1,
    )

    # Resolution recommendation
    df["Recommended_Resolution"] = df.apply(
        lambda row: recommend_resolution(
            category=row["Routing_Category"],
            intent=row.get("Intent", ""),
            complaint_text=row.get("Complaint_Text", ""),
            priority=row.get("Priority", ""),
        ),
        axis=1,
    )

    return df


# ============================================================
# SAVE DATA
# ============================================================

def save_data(df):

    OUTPUT_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_DATA_PATH,
        index=False
    )

    print(
        f"\nRouting dataset saved to:\n{OUTPUT_DATA_PATH}"
    )


# ============================================================
# CREATE REPORT
# ============================================================

def create_report(df):

    department_counts = (
        df["Recommended_Department"]
        .value_counts()
    )

    category_counts = (
        df["Routing_Category"]
        .value_counts()
    )

    report = f"""# Module 11 – Complaint Routing & Resolution Recommendation

## Objective

Automatically determine the appropriate department for each customer complaint and recommend an appropriate resolution.

## Dataset

- Complaints processed: {len(df):,}
- Routing categories: {df["Routing_Category"].nunique()}
- Recommended departments: {df["Recommended_Department"].nunique()}

## Department Routing Distribution

"""

    for department, count in department_counts.items():
        percentage = count / len(df) * 100

        report += (
            f"- **{department}**: "
            f"{count:,} complaints "
            f"({percentage:.2f}%)\n"
        )

    report += "\n## Category Distribution\n\n"

    for category, count in category_counts.items():
        percentage = count / len(df) * 100

        report += (
            f"- **{category}**: "
            f"{count:,} complaints "
            f"({percentage:.2f}%)\n"
        )

    report += """
## Routing Logic

The system combines complaint category, customer intent, and complaint text to determine the appropriate department.

### Department Mapping

| Complaint Type | Department |
|---|---|
| Payment | Payments |
| Refund | Finance |
| Delivery | Logistics |
| Product Issue | Product Support |
| Login Issue | Technical Support |
| Account Issue | Account Support |
| Security Issue | Security |
| Subscription | Billing |
| General Query | Customer Support |

## Resolution Recommendation

The system generates a recommended action based on the complaint category and important complaint keywords.

Examples include:

- Payment → Verify transaction and order status.
- Refund → Verify refund status and process/escalate the refund.
- Delivery → Check shipment and coordinate with logistics.
- Product → Arrange replacement, return, or repair.
- Security → Secure account and escalate to security.
- Billing → Verify charges and correct billing issues.

## Required Example

**Complaint:**  
"My payment was successful but the order was not created."

Expected routing:

- Category: Payment
- Department: Payments
- Priority: High
- Resolution: Verify transaction and initiate order creation or refund.

## Module 11 Deliverable

Automated complaint routing and resolution recommendation system.
"""

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8"
    )

    print(
        f"Report saved to:\n{REPORT_PATH}"
    )


# ============================================================
# TEST CASES
# ============================================================

def run_tests():

    print("\n" + "=" * 60)
    print("MODULE 11 TEST CASES")
    print("=" * 60)

    test_cases = [
        {
            "complaint": "My payment was successful but the order was not created.",
            "category": "Payment",
            "intent": "Payment Issue",
        },
        {
            "complaint": "I want a refund because I was charged for an order that was cancelled.",
            "category": "Refund",
            "intent": "Refund Request",
        },
        {
            "complaint": "Someone accessed my account and made an unauthorized transaction.",
            "category": "Security",
            "intent": "Payment Issue",
        },
        {
            "complaint": "My order arrived damaged and I want a replacement.",
            "category": "Product",
            "intent": "Product Replacement",
        },
        {
            "complaint": "I cannot login to my account and need help resetting my password.",
            "category": "Technical Support",
            "intent": "Account Recovery",
        },
    ]

    results = []

    for i, case in enumerate(test_cases, start=1):

        department = predict_department(
            category=case["category"],
            intent=case["intent"],
            complaint_text=case["complaint"],
        )

        resolution = recommend_resolution(
            category=case["category"],
            intent=case["intent"],
            complaint_text=case["complaint"],
        )

        print(f"\nTest {i}")
        print(f"Complaint: {case['complaint']}")
        print(f"Department: {department}")
        print(f"Resolution: {resolution}")

        results.append(
            {
                "Test": i,
                "Complaint": case["complaint"],
                "Category": case["category"],
                "Intent": case["intent"],
                "Department": department,
                "Recommended_Resolution": resolution,
            }
        )

    test_path = (
        PROJECT_ROOT
        / "module11"
        / "reports"
        / "module11_test_results.csv"
    )

    pd.DataFrame(results).to_csv(
        test_path,
        index=False
    )

    print(
        f"\nTest results saved to:\n{test_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df = process_data(df)

    save_data(df)

    create_report(df)

    run_tests()

    print("\n" + "=" * 60)
    print("MODULE 11 COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()