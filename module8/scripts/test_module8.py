import joblib
import pandas as pd
from pathlib import Path


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "module8"
    / "models"
    / "urgency_priority_random_forest.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

print("=" * 70)
print("MODULE 8 SYSTEM TEST")
print("=" * 70)


# ============================================================
# TEST CASES
# ============================================================

test_cases = [

    {
        "name": "LOW - General Question",
        "Category": "General Inquiry",
        "Predicted_Sentiment": "Neutral",
        "Predicted_Emotion": "Neutral",
        "Emotion_Intensity": "Neutral",
        "Urgency_Keyword_Count": 0,
        "Customer_Impact": "Low",
        "Financial_Impact": 0,
        "Account_Status": "Normal",
        "Complaint_History_Count": 1,
        "expected": "Low"
    },

    {
        "name": "MEDIUM - Customer Experience Issue",
        "Category": "Product",
        "Predicted_Sentiment": "Negative",
        "Predicted_Emotion": "Disappointment",
        "Emotion_Intensity": "Negative",
        "Urgency_Keyword_Count": 0,
        "Customer_Impact": "Medium",
        "Financial_Impact": 0,
        "Account_Status": "Normal",
        "Complaint_History_Count": 1,
        "expected": "Medium"
    },

    {
        "name": "HIGH - Urgent Payment Problem",
        "Category": "Payment",
        "Predicted_Sentiment": "Negative",
        "Predicted_Emotion": "Frustration",
        "Emotion_Intensity": "Very Negative",
        "Urgency_Keyword_Count": 1,
        "Customer_Impact": "High",
        "Financial_Impact": 1,
        "Account_Status": "Affected",
        "Complaint_History_Count": 1,
        "expected": "High"
    },

    {
        "name": "CRITICAL - Unauthorized Transaction",
        "Category": "Security",
        "Predicted_Sentiment": "Negative",
        "Predicted_Emotion": "Fear",
        "Emotion_Intensity": "Very Negative",
        "Urgency_Keyword_Count": 0,
        "Customer_Impact": "Severe",
        "Financial_Impact": 1,
        "Account_Status": "Compromised",
        "Complaint_History_Count": 1,
        "expected": "Critical"
    }
]


# ============================================================
# RUN TESTS
# ============================================================

passed = 0
failed = 0

priority_mapping = {
    "Critical": "P1",
    "High": "P2",
    "Medium": "P3",
    "Low": "P4"
}


for case in test_cases:

    input_data = pd.DataFrame([{
        "Category": case["Category"],
        "Predicted_Sentiment": case["Predicted_Sentiment"],
        "Predicted_Emotion": case["Predicted_Emotion"],
        "Emotion_Intensity": case["Emotion_Intensity"],
        "Urgency_Keyword_Count": case["Urgency_Keyword_Count"],
        "Customer_Impact": case["Customer_Impact"],
        "Financial_Impact": case["Financial_Impact"],
        "Account_Status": case["Account_Status"],
        "Complaint_History_Count": case["Complaint_History_Count"]
    }])

    prediction = model.predict(input_data)[0]

    priority = priority_mapping[prediction]

    expected = case["expected"]

    print("\nTest:", case["name"])
    print("Expected Urgency:", expected)
    print("Predicted Urgency:", prediction)
    print("Priority:", priority)

    if prediction == expected:
        print("RESULT: PASS")
        passed += 1
    else:
        print("RESULT: FAIL")
        failed += 1


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

print("Tests Passed:", passed)
print("Tests Failed:", failed)

if failed == 0:
    print("\nALL MODULE 8 TESTS PASSED!")
    print("MODULE 8 IS READY.")
else:
    print("\nSOME TESTS FAILED.")
    print("Module 8 needs adjustment.")