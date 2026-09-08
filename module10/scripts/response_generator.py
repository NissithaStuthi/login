from pathlib import Path
import re
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "module9"
    / "data"
    / "intent_entity_dataset.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "module10"
    / "data"
    / "generated_customer_responses.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "module10"
    / "reports"
    / "MODULE10_RESPONSE_GENERATION_REPORT.md"
)


# ============================================================
# RESPONSE GENERATION
# ============================================================

def get_value(row, column, default="Not Found"):
    """Safely get a value from a dataframe row."""
    if column not in row.index:
        return default

    value = row[column]

    if pd.isna(value) or str(value).strip() == "":
        return default

    return str(value).strip()


def clean_value(value):
    """Clean extracted entity values."""
    if value == "Not Found":
        return ""

    return str(value).strip()


def generate_response(
    complaint,
    category="General Inquiry",
    sentiment="Neutral",
    urgency="Low",
    intent="Information Request",
    customer_name="",
    order_id="",
    amount="",
    date="",
    transaction_id="",
    recommended_resolution=""
):
    """
    Generate an intelligent customer response using
    complaint understanding, sentiment, urgency, intent,
    entities and resolution information.
    """

    complaint_lower = complaint.lower()

    # --------------------------------------------------------
    # Identify priority wording
    # --------------------------------------------------------

    if urgency == "Critical":
        urgency_text = (
            "This is a critical issue and has been marked for immediate attention."
        )
    elif urgency == "High":
        urgency_text = (
            "This issue has been marked as high priority for quick assistance."
        )
    elif urgency == "Medium":
        urgency_text = (
            "We understand that this issue is affecting your experience."
        )
    else:
        urgency_text = (
            "We appreciate you bringing this matter to our attention."
        )

    # --------------------------------------------------------
    # Personal greeting
    # --------------------------------------------------------

    if customer_name:
        greeting = f"Dear {customer_name},"
    else:
        greeting = "Dear Customer,"

    # --------------------------------------------------------
    # Entity details
    # --------------------------------------------------------

    reference_details = []

    if order_id:
        reference_details.append(f"order {order_id}")

    if transaction_id:
        reference_details.append(f"transaction {transaction_id}")

    if amount:
        reference_details.append(f"amount {amount}")

    if date:
        reference_details.append(f"date {date}")

    if reference_details:
        reference_text = (
            " We have noted the details related to "
            + ", ".join(reference_details)
            + "."
        )
    else:
        reference_text = ""

    # --------------------------------------------------------
    # Intent-specific response
    # --------------------------------------------------------

    if intent == "Refund Request":
        action = (
            "Your refund request has been identified and will be reviewed "
            "by the payments/refunds team."
        )

    elif intent == "Cancellation Request":
        action = (
            "Your cancellation request has been identified. "
            "The relevant team will verify the request and process it "
            "according to the applicable cancellation policy."
        )

    elif intent == "Payment Issue":
        action = (
            "Your payment issue has been identified. "
            "The payment team will review the transaction and verify "
            "the payment status."
        )

    elif intent == "Account Recovery":
        action = (
            "Your account-access issue has been identified. "
            "The technical support team will assist you with account recovery."
        )

    elif intent == "Delivery Tracking":
        action = (
            "Your delivery-related request has been identified. "
            "The support team will check the latest delivery information "
            "and provide an update."
        )

    elif intent == "Product Replacement":
        action = (
            "Your product replacement request has been identified. "
            "The product support team will review the issue and guide you "
            "through the replacement process."
        )

    elif intent == "Complaint Escalation":
        action = (
            "Your complaint has been marked for escalation. "
            "The appropriate support team will review the matter "
            "and provide an update."
        )

    elif intent == "Technical Assistance":
        action = (
            "Your technical issue has been identified. "
            "Our technical support team will review the problem "
            "and assist you with the next steps."
        )

    else:
        action = (
            "Your request has been identified. "
            "Our customer support team will review the information "
            "and provide the appropriate assistance."
        )

    # --------------------------------------------------------
    # Recommended resolution
    # --------------------------------------------------------

    if recommended_resolution:
        resolution_text = (
            f" Recommended resolution: {recommended_resolution}."
        )
    else:
        resolution_text = ""

    # --------------------------------------------------------
    # Sentiment-aware closing
    # --------------------------------------------------------

    if sentiment == "Negative":
        closing = (
            "We sincerely apologize for the inconvenience and understand "
            "your concern."
        )

    elif sentiment == "Positive":
        closing = (
            "Thank you for your patience and for contacting us."
        )

    else:
        closing = (
            "Thank you for contacting customer support."
        )

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    response = (
        f"{greeting}\n\n"
        f"We understand your concern regarding your complaint. "
        f"{urgency_text}{reference_text}\n\n"
        f"{action}{resolution_text}\n\n"
        f"{closing}\n\n"
        f"Regards,\nCustomer Support Team"
    )

    return response


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset():

    print("=" * 70)
    print("MODULE 10 - AI AUTOMATIC RESPONSE GENERATION")
    print("=" * 70)

    print("\nLoading Module 9 intent and entity dataset...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_PATH}\n\n"
            "Please make sure Module 9 has been completed successfully."
        )

    df = pd.read_csv(INPUT_PATH)

    print(f"Dataset loaded successfully: {df.shape[0]:,} rows")

    responses = []

    for _, row in df.iterrows():

        complaint = get_value(row, "Complaint_Text", "")

        category = get_value(
            row,
            "Target_Category",
            get_value(row, "Category", "General Inquiry")
        )

        sentiment = get_value(
            row,
            "Predicted_Sentiment",
            get_value(row, "Sentiment", "Neutral")
        )

        urgency = get_value(
            row,
            "Urgency",
            "Low"
        )

        intent = get_value(
            row,
            "Predicted_Intent",
            get_value(row, "Intent", "Information Request")
        )

        customer_name = clean_value(
            get_value(row, "Customer_Name")
        )

        order_id = clean_value(
            get_value(row, "Order_ID")
        )

        amount = clean_value(
            get_value(row, "Amount")
        )

        date = clean_value(
            get_value(row, "Date_Entity")
        )

        transaction_id = clean_value(
            get_value(row, "Transaction_ID")
        )

        recommended_resolution = clean_value(
            get_value(row, "Recommended_Resolution")
        )

        response = generate_response(
            complaint=complaint,
            category=category,
            sentiment=sentiment,
            urgency=urgency,
            intent=intent,
            customer_name=customer_name,
            order_id=order_id,
            amount=amount,
            date=date,
            transaction_id=transaction_id,
            recommended_resolution=recommended_resolution
        )

        responses.append(response)

    df["AI_Generated_Response"] = responses

    # Save generated responses
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nGenerated responses successfully.")
    print(f"Output saved to:\n{OUTPUT_PATH}")

    # ========================================================
    # CREATE REPORT
    # ========================================================

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = f"""# MODULE 10 – AI Automatic Response Generation

## Objective

Generate intelligent customer responses using complaint text,
category, sentiment, urgency, customer intent, extracted entities,
and recommended resolution information.

## Input Dataset

Module 9 intent and entity extraction dataset.

**Total complaints processed:** {len(df):,}

## Response Generation Workflow

Customer Complaint
↓
Complaint Understanding
↓
Sentiment Analysis
↓
Intent Detection
↓
Entity Extraction
↓
Urgency & Priority Information
↓
Resolution Information
↓
AI Response Generator
↓
Customer Response

## Features Implemented

- Complaint-aware response generation
- Intent-specific responses
- Sentiment-aware wording
- Urgency-aware wording
- Customer name personalization
- Order ID extraction support
- Amount extraction support
- Date extraction support
- Transaction ID extraction support
- Recommended resolution integration
- Professional customer-support tone

## Output

Generated response dataset:

`module10/data/generated_customer_responses.csv`

## Example

**Complaint:**

My payment was deducted but my order was cancelled. I have been
waiting for my refund for 7 days.

**Intent:**

Refund Request

**Urgency:**

High

**Generated Response:**

Your refund request has been identified and will be reviewed
by the payments/refunds team.

The issue has been marked as high priority for quick assistance.

## Conclusion

Module 10 successfully implements an automatic customer response
generation system that combines complaint intelligence outputs
from previous modules to produce structured and personalized
customer-support responses.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8"
    )

    print(f"\nReport saved to:\n{REPORT_PATH}")

    return df


# ============================================================
# TEST EXAMPLES
# ============================================================

def run_tests():

    print("\n" + "=" * 70)
    print("MODULE 10 RESPONSE GENERATION TESTS")
    print("=" * 70)

    test_cases = [

        {
            "complaint": (
                "My payment was deducted but my order was cancelled. "
                "I have been waiting for my refund for 7 days."
            ),
            "category": "Payment",
            "sentiment": "Negative",
            "urgency": "High",
            "intent": "Refund Request",
            "order_id": "ORD4567",
            "amount": "₹2,500",
            "date": "August 28"
        },

        {
            "complaint": (
                "I cannot login to my account even though my password is correct."
            ),
            "category": "Account",
            "sentiment": "Negative",
            "urgency": "High",
            "intent": "Account Recovery"
        },

        {
            "complaint": (
                "Someone accessed my account and made an unauthorized transaction."
            ),
            "category": "Security",
            "sentiment": "Negative",
            "urgency": "Critical",
            "intent": "Payment Issue",
            "transaction_id": "TXN98765"
        },

        {
            "complaint": (
                "I received my product damaged and would like a replacement."
            ),
            "category": "Product",
            "sentiment": "Negative",
            "urgency": "Medium",
            "intent": "Product Replacement"
        }
    ]

    for number, test in enumerate(test_cases, start=1):

        response = generate_response(
            complaint=test["complaint"],
            category=test.get("category", "General Inquiry"),
            sentiment=test.get("sentiment", "Neutral"),
            urgency=test.get("urgency", "Low"),
            intent=test.get("intent", "Information Request"),
            order_id=test.get("order_id", ""),
            amount=test.get("amount", ""),
            date=test.get("date", ""),
            transaction_id=test.get("transaction_id", "")
        )

        print(f"\nTEST {number}")
        print("-" * 50)
        print("Complaint:")
        print(test["complaint"])
        print("\nGenerated Response:")
        print(response)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    process_dataset()
    run_tests()

    print("\n" + "=" * 70)
    print("MODULE 10 COMPLETED SUCCESSFULLY!")
    print("=" * 70)