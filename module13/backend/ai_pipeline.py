from pathlib import Path
import re
import sys
import joblib


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# OPTIONAL MODULE IMPORTS
# ============================================================

try:
    from module10.scripts.response_generator import generate_response
except Exception:
    generate_response = None


try:
    from module11.scripts.routing_resolution_system import (
        predict_department,
        recommend_resolution,
    )
except Exception:
    predict_department = None
    recommend_resolution = None


# ============================================================
# MODULE 6 - MODEL LOADING
# ============================================================

CLASSIFIER_PATH = (
    PROJECT_ROOT
    / "module6"
    / "models"
    / "final_complaint_classifier.pkl"
)

VECTORIZER_PATH = (
    PROJECT_ROOT
    / "module6"
    / "models"
    / "final_tfidf_vectorizer.pkl"
)

classifier = None
vectorizer = None

try:
    if CLASSIFIER_PATH.exists() and VECTORIZER_PATH.exists():
        classifier = joblib.load(CLASSIFIER_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("✅ Module 6 classifier loaded")
    else:
        print("⚠️ Module 6 model files not found")

except Exception as error:
    print(f"⚠️ Module 6 model loading failed: {error}")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ============================================================
# MODULE 6 - COMPLAINT CLASSIFICATION
# ============================================================

def classify_complaint(text: str) -> str:

    text_lower = text.lower().strip()

    # Security
    if any(
        word in text_lower
        for word in [
            "unauthorized",
            "fraud",
            "hacked",
            "stolen",
            "suspicious transaction",
            "account compromised",
            "security breach",
        ]
    ):
        return "Security"

    # Billing + Refund
    has_refund = "refund" in text_lower

    has_duplicate_charge = any(
        word in text_lower
        for word in [
            "charged twice",
            "duplicate charge",
            "double charged",
            "charged two times",
        ]
    )

    if has_refund and has_duplicate_charge:
        return "Billing / Refund"

    if has_refund:
        return "Refund"

    if has_duplicate_charge:
        return "Billing"

    # Delivery
    if any(
        word in text_lower
        for word in [
            "delivery",
            "delivered",
            "shipment",
            "tracking",
            "not delivered",
            "late delivery",
        ]
    ):
        return "Delivery"

    # Product
    if any(
        word in text_lower
        for word in [
            "damaged",
            "broken",
            "defective",
            "replacement",
        ]
    ):
        return "Product"

    # Technical Support
    if any(
        word in text_lower
        for word in [
            "login",
            "log in",
            "password",
            "cannot access",
            "account access",
            "not working",
            "error",
        ]
    ):
        return "Technical Support"

    # Subscription
    if "subscription" in text_lower:
        return "Subscription"

    # ML fallback
    if classifier is not None and vectorizer is not None:
        try:
            complaint_vector = vectorizer.transform([text])
            prediction = classifier.predict(complaint_vector)[0]
            return str(prediction)
        except Exception:
            pass

    return "General Inquiry"


# ============================================================
# MODULE 7 - SENTIMENT
# ============================================================

def analyze_sentiment(text: str) -> str:

    text_lower = text.lower()

    positive_words = [
        "good",
        "great",
        "excellent",
        "happy",
        "satisfied",
        "thank",
        "thanks",
        "resolved",
    ]

    negative_words = [
        "bad",
        "angry",
        "frustrated",
        "terrible",
        "worst",
        "failed",
        "problem",
        "issue",
        "wrong",
        "charged",
        "damaged",
        "refund",
        "delay",
        "unable",
        "complaint",
    ]

    positive_score = sum(
        word in text_lower
        for word in positive_words
    )

    negative_score = sum(
        word in text_lower
        for word in negative_words
    )

    if negative_score > positive_score:
        return "Negative"

    if positive_score > negative_score:
        return "Positive"

    return "Neutral"


# ============================================================
# MODULE 7 - EMOTION
# ============================================================

def analyze_emotion(text: str) -> tuple[str, str]:

    text_lower = text.lower()

    emotion_rules = {
        "Anger": [
            "angry",
            "furious",
            "unacceptable",
            "ridiculous",
            "worst",
        ],
        "Frustration": [
            "frustrated",
            "frustrating",
            "again",
            "still",
            "unable",
            "not working",
            "charged twice",
        ],
        "Disappointment": [
            "disappointed",
            "expected better",
            "let down",
        ],
        "Fear": [
            "scared",
            "afraid",
            "fear",
            "unauthorized",
            "hacked",
            "fraud",
        ],
        "Sadness": [
            "sad",
            "upset",
            "lost",
            "sorry",
        ],
        "Satisfaction": [
            "happy",
            "satisfied",
            "excellent",
            "great",
            "thank you",
        ],
    }

    for emotion, keywords in emotion_rules.items():

        if any(
            keyword in text_lower
            for keyword in keywords
        ):

            intensity = "High"

            if len(text_lower) < 60:
                intensity = "Medium"

            return emotion, intensity

    return "Neutral", "Low"


# ============================================================
# MODULE 8 - URGENCY & PRIORITY
# ============================================================

def detect_urgency(text: str) -> tuple[str, str]:

    text_lower = text.lower()

    critical_keywords = [
        "unauthorized",
        "fraud",
        "hacked",
        "stolen",
        "account compromised",
        "security breach",
        "someone accessed",
    ]

    high_keywords = [
        "urgent",
        "immediately",
        "as soon as possible",
        "payment failed",
        "refund",
        "charged twice",
        "cannot login",
        "cannot log in",
        "account locked",
    ]

    medium_keywords = [
        "problem",
        "issue",
        "delay",
        "damaged",
        "not working",
    ]

    if any(
        word in text_lower
        for word in critical_keywords
    ):
        return "Critical", "P1"

    if any(
        word in text_lower
        for word in high_keywords
    ):
        return "High", "P2"

    if any(
        word in text_lower
        for word in medium_keywords
    ):
        return "Medium", "P3"

    return "Low", "P4"


# ============================================================
# MODULE 9 - INTENT
# ============================================================

def detect_intent(text: str) -> str:

    text_lower = text.lower()

    if "refund" in text_lower:
        return "Refund Request"

    if "cancel" in text_lower:
        return "Cancellation Request"

    if any(
        word in text_lower
        for word in [
            "payment",
            "charged",
            "transaction",
        ]
    ):
        return "Payment Issue"

    if any(
        word in text_lower
        for word in [
            "login",
            "log in",
            "password",
            "recover",
            "access my account",
        ]
    ):
        return "Account Recovery"

    if any(
        word in text_lower
        for word in [
            "tracking",
            "where is my order",
            "delivery status",
            "shipment",
        ]
    ):
        return "Delivery Tracking"

    if any(
        word in text_lower
        for word in [
            "replacement",
            "replace",
            "damaged",
            "defective",
        ]
    ):
        return "Product Replacement"

    if any(
        word in text_lower
        for word in [
            "escalate",
            "manager",
            "complaint",
        ]
    ):
        return "Complaint Escalation"

    if any(
        word in text_lower
        for word in [
            "technical",
            "error",
            "not working",
            "bug",
        ]
    ):
        return "Technical Assistance"

    return "Information Request"


# ============================================================
# MODULE 9 - ENTITY EXTRACTION
# ============================================================

def extract_entities(text: str) -> dict:

    result = {
        "customer_name": None,
        "order_id": None,
        "amount": None,
        "date": None,
        "transaction_id": None,
        "location": None,
        "account_type": None,
    }

    # Customer name
    name_match = re.search(
        r"\b(?:my name is|i am|this is)\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        text,
        re.IGNORECASE,
    )

    if name_match:
        result["customer_name"] = (
            name_match.group(1).strip()
        )

    # Order ID
    order_match = re.search(
        r"\b(?:ORD|ORDER)[-_]?[A-Z0-9]+\b",
        text,
        re.IGNORECASE,
    )

    if order_match:
        result["order_id"] = order_match.group(0)

    # Amount
    amount_match = re.search(
        r"(₹\s?[\d,]+(?:\.\d+)?|"
        r"\bRs\.?\s?[\d,]+(?:\.\d+)?|"
        r"\bINR\s?[\d,]+(?:\.\d+)?|"
        r"\$\s?[\d,]+(?:\.\d+)?|"
        r"\bUSD\s?[\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )

    if amount_match:
        result["amount"] = amount_match.group(0)

    # Date
    date_match = re.search(
        r"\b(?:January|February|March|April|May|June|July|"
        r"August|September|October|November|December)"
        r"\s+\d{1,2}(?:,\s*\d{4})?\b",
        text,
        re.IGNORECASE,
    )

    if date_match:
        result["date"] = date_match.group(0)

    # Transaction ID
    transaction_match = re.search(
        r"\b(?:TXN|TRAN|TRX)[-_]?[A-Z0-9]+\b",
        text,
        re.IGNORECASE,
    )

    if transaction_match:
        result["transaction_id"] = (
            transaction_match.group(0)
        )

    # Account type
    account_types = [
        "savings",
        "current",
        "business",
        "student",
        "personal",
        "premium",
        "credit card",
        "bank account",
        "merchant",
    ]

    for account_type in account_types:

        if account_type in text.lower():
            result["account_type"] = account_type
            break

    # Location
    location_match = re.search(
        r"\b(?:in|at|from|near)\s+"
        r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})",
        text,
    )

    if location_match:
        result["location"] = (
            location_match.group(1).strip()
        )

    return result


# ============================================================
# MODULE 11 - ROUTING & RESOLUTION
# ============================================================

def route_and_resolve(
    category: str,
    intent: str,
    complaint_text: str,
) -> tuple[str, str]:

    # Department
    if predict_department is not None:

        try:
            department = predict_department(
                category=category,
                intent=intent,
                complaint_text=complaint_text,
            )
        except Exception:
            department = "Customer Support"

    else:
        department = "Customer Support"

    # Resolution
    if recommend_resolution is not None:

        try:
            resolution = recommend_resolution(
                category=category,
                intent=intent,
                complaint_text=complaint_text,
            )
        except Exception:
            resolution = (
                "Review the complaint and provide "
                "appropriate customer support."
            )

    else:
        resolution = (
            "Review the complaint and provide "
            "appropriate customer support."
        )

    return str(department), str(resolution)


# ============================================================
# MODULE 10 - AI RESPONSE
# ============================================================

def create_response(
    complaint_text: str,
    category: str,
    sentiment: str,
    urgency: str,
    intent: str,
    entities: dict,
    resolution: str,
) -> str:

    if generate_response is not None:

        try:

            return generate_response(
                complaint=complaint_text,
                category=category,
                sentiment=sentiment,
                urgency=urgency,
                intent=intent,
                customer_name=entities.get(
                    "customer_name"
                ),
                order_id=entities.get(
                    "order_id"
                ),
                amount=entities.get(
                    "amount"
                ),
                date=entities.get(
                    "date"
                ),
                transaction_id=entities.get(
                    "transaction_id"
                ),
                recommended_resolution=resolution,
            )

        except Exception:
            pass

    return (
        "We understand your concern. "
        f"Your complaint has been identified as "
        f"a {category} issue with {urgency.lower()} urgency. "
        "Our support team will review the issue and "
        "take the appropriate action. "
        f"Recommended action: {resolution}"
    )


# ============================================================
# MODULE 12 - ANOMALY & RECURRING ISSUE
# ============================================================

def detect_anomaly_and_recurring(
    text: str,
) -> tuple[bool, str | None]:

    text_lower = text.lower()

    anomaly_keywords = [
        "sudden spike",
        "system failure",
        "massive",
        "many customers",
        "outage",
    ]

    recurring_patterns = {

        "Payment Status Synchronization": [
            "payment completed",
            "order status pending",
            "order not created",
        ],

        "Refund Delay": [
            "refund pending",
            "refund not received",
            "waiting for refund",
        ],

        "Login Failure": [
            "cannot login",
            "cannot log in",
            "password not working",
        ],

        "Duplicate Billing": [
            "charged twice",
            "duplicate charge",
            "double charged",
        ],
    }

    is_anomaly = any(
        keyword in text_lower
        for keyword in anomaly_keywords
    )

    for issue_name, keywords in recurring_patterns.items():

        if any(
            keyword in text_lower
            for keyword in keywords
        ):
            return is_anomaly, issue_name

    return is_anomaly, None


# ============================================================
# COMPLETE AI PIPELINE
# ============================================================

def analyze_complaint(
    complaint_text: str,
) -> dict:

    complaint_text = clean_text(
        complaint_text
    )

    if not complaint_text:
        raise ValueError(
            "Complaint text cannot be empty."
        )

    # Module 6
    category = classify_complaint(
        complaint_text
    )

    # Module 7
    sentiment = analyze_sentiment(
        complaint_text
    )

    emotion, emotion_intensity = analyze_emotion(
        complaint_text
    )

    # Module 8
    urgency, priority = detect_urgency(
        complaint_text
    )

    # Module 9
    intent = detect_intent(
        complaint_text
    )

    entities = extract_entities(
        complaint_text
    )

    # Module 11
    department, resolution = route_and_resolve(
        category=category,
        intent=intent,
        complaint_text=complaint_text,
    )

    # Module 10
    ai_response = create_response(
        complaint_text=complaint_text,
        category=category,
        sentiment=sentiment,
        urgency=urgency,
        intent=intent,
        entities=entities,
        resolution=resolution,
    )

    # Module 12
    is_anomaly, recurring_issue = (
        detect_anomaly_and_recurring(
            complaint_text
        )
    )

    return {
        "complaint_text": complaint_text,
        "category": category,
        "sentiment": sentiment,
        "emotion": emotion,
        "emotion_intensity": emotion_intensity,
        "urgency": urgency,
        "priority": priority,
        "intent": intent,
        "entities": entities,
        "recommended_department": department,
        "recommended_resolution": resolution,
        "ai_response": ai_response,
        "is_anomaly": is_anomaly,
        "recurring_issue": recurring_issue,
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    test_complaint = (
        "I was charged twice and still haven't "
        "received my refund."
    )

    result = analyze_complaint(
        test_complaint
    )

    print("\n" + "=" * 60)
    print("RESOLVEIQ AI PIPELINE TEST")
    print("=" * 60)

    for key, value in result.items():
        print(f"\n{key}: {value}")

    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETED")
    print("=" * 60)