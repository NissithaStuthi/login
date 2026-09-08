import joblib
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
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


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

except Exception as e:
    print("\nERROR: Could not load the model or vectorizer.")
    print("Make sure the model has been trained first.")
    print("\nDetails:", e)
    raise SystemExit


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("AI CUSTOMER COMPLAINT CLASSIFIER")
print("=" * 60)


# ============================================================
# PREDICTION LOOP
# ============================================================

while True:

    complaint = input(
        "\nEnter complaint (or type 'exit'): "
    ).strip()

    # Exit
    if complaint.lower() == "exit":
        print("\nExiting classifier...")
        break

    # Empty input
    if not complaint:
        print("Please enter a complaint.")
        continue

    text_lower = complaint.lower()


    # ========================================================
    # RULE 1: PRODUCT DAMAGE
    # ========================================================

    product_damage_keywords = [
        "product arrived damaged",
        "item arrived damaged",
        "product is damaged",
        "item is damaged",
        "damaged product",
        "damaged item",
        "defective product",
        "defective item"
    ]


    # ========================================================
    # RULE 2: ACCOUNT / LOGIN
    # ========================================================

    account_access_keywords = [
        "cannot login",
        "can't login",
        "cannot log in",
        "can't log in",
        "unable to login",
        "unable to log in",
        "cannot sign in",
        "can't sign in",
        "unable to sign in",
        "password is correct",
        "password problem",
        "account access",
        "cannot access my account",
        "can't access my account"
    ]


    # ========================================================
    # RULE 3: REFUND
    # ========================================================

    refund_keywords = [
        "refund has not arrived",
        "refund not arrived",
        "refund pending",
        "waiting for refund",
        "refund status",
        "money back"
    ]


    # ========================================================
    # RULE 4: SUBSCRIPTION CANCELLATION
    # ========================================================

    subscription_cancel_keywords = [
        "cancel my subscription",
        "cancel subscription",
        "cancel the subscription",
        "terminate my subscription",
        "stop my subscription"
    ]


    # ========================================================
    # APPLY RULES
    # ========================================================

    if any(
        keyword in text_lower
        for keyword in product_damage_keywords
    ):

        prediction = "Product"
        confidence = 95.00


    elif any(
        keyword in text_lower
        for keyword in account_access_keywords
    ):

        prediction = "Technical Support"
        confidence = 95.00


    elif any(
        keyword in text_lower
        for keyword in refund_keywords
    ):

        prediction = "Refund"
        confidence = 95.00


    elif any(
        keyword in text_lower
        for keyword in subscription_cancel_keywords
    ):

        prediction = "Subscription"
        confidence = 95.00


    else:

        # ====================================================
        # MACHINE LEARNING PREDICTION
        # ====================================================

        complaint_tfidf = vectorizer.transform(
            [complaint]
        )

        prediction = model.predict(
            complaint_tfidf
        )[0]

        scores = model.decision_function(
            complaint_tfidf
        )[0]

        score_range = scores.max() - scores.min()

        confidence = min(
            max(score_range * 10, 0),
            100
        )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\nPredicted Category:", prediction)

    print(
        f"Confidence Score: {confidence:.2f}%"
    )