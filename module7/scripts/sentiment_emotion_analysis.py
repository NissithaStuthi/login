import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "module4"
    / "data"
    / "processed"
    / "processed_customer_complaints.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "module7" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "sentiment_emotion_dataset.csv"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("Dataset loaded:", df.shape)

df["Complaint_Text"] = (
    df["Complaint_Text"]
    .fillna("")
    .astype(str)
)


# --------------------------------------------------
# VADER SENTIMENT ANALYZER
# --------------------------------------------------

vader = SentimentIntensityAnalyzer()


def analyze_sentiment(text):

    vader_result = vader.polarity_scores(text)

    compound = vader_result["compound"]

    textblob_score = TextBlob(text).sentiment.polarity

    # Combine VADER and TextBlob
    combined_score = (compound + textblob_score) / 2

    if combined_score >= 0.05:
        sentiment = "Positive"

    elif combined_score <= -0.05:
        sentiment = "Negative"

    else:
        sentiment = "Neutral"

    return sentiment, compound, textblob_score, combined_score


results = df["Complaint_Text"].apply(analyze_sentiment)

df[
    [
        "Predicted_Sentiment",
        "VADER_Score",
        "TextBlob_Score",
        "Combined_Sentiment_Score"
    ]
] = pd.DataFrame(
    results.tolist(),
    index=df.index
)


# --------------------------------------------------
# EMOTION DETECTION
# --------------------------------------------------

emotion_keywords = {

    "Anger": [
        "angry",
        "furious",
        "rage",
        "ridiculous",
        "unacceptable",
        "mad",
        "outrage"
    ],

    "Frustration": [
        "frustrated",
        "frustration",
        "failed",
        "failure",
        "unable",
        "cannot",
        "can't",
        "problem",
        "issue",
        "error",
        "not working"
    ],

    "Disappointment": [
        "disappointed",
        "disappointing",
        "poor",
        "worst",
        "expected",
        "expectation"
    ],

    "Fear": [
        "scared",
        "fear",
        "worried",
        "worry",
        "danger",
        "unsafe",
        "security",
        "threat"
    ],

    "Sadness": [
        "sad",
        "unhappy",
        "upset",
        "hurt",
        "lost",
        "cry"
    ],

    "Satisfaction": [
        "happy",
        "satisfied",
        "great",
        "excellent",
        "good",
        "thank",
        "thanks",
        "resolved",
        "helpful",
        "amazing",
        "perfect"
    ]
}


def detect_emotion(text):

    text = text.lower()

    scores = {}

    for emotion, keywords in emotion_keywords.items():

        score = 0

        for keyword in keywords:

            if re.search(
                r"\b" + re.escape(keyword) + r"\b",
                text
            ):
                score += 1

        scores[emotion] = score

    best_emotion = max(
        scores,
        key=scores.get
    )

    if scores[best_emotion] == 0:
        return "Neutral"

    return best_emotion


df["Predicted_Emotion"] = (
    df["Complaint_Text"]
    .apply(detect_emotion)
)


# --------------------------------------------------
# EMOTION INTENSITY
# --------------------------------------------------

def calculate_intensity(score):

    if score <= -0.60:
        return "Very Negative"

    elif score <= -0.05:
        return "Negative"

    elif score >= 0.60:
        return "Very Positive"

    elif score >= 0.05:
        return "Positive"

    else:
        return "Neutral"


df["Emotion_Intensity"] = (
    df["Combined_Sentiment_Score"]
    .apply(calculate_intensity)
)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

output_columns = [
    "Complaint_ID",
    "Complaint_Text",
    "Predicted_Sentiment",
    "Predicted_Emotion",
    "Emotion_Intensity",
    "VADER_Score",
    "TextBlob_Score",
    "Combined_Sentiment_Score",
    "Source_Dataset"
]

available_columns = [
    column
    for column in output_columns
    if column in df.columns
]

df[available_columns].to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 50)
print("SENTIMENT DISTRIBUTION")
print("=" * 50)

print(
    df["Predicted_Sentiment"]
    .value_counts()
)


print("\n" + "=" * 50)
print("EMOTION DISTRIBUTION")
print("=" * 50)

print(
    df["Predicted_Emotion"]
    .value_counts()
)


print("\n" + "=" * 50)
print("EMOTION INTENSITY")
print("=" * 50)

print(
    df["Emotion_Intensity"]
    .value_counts()
)


print("\n" + "=" * 50)
print("SAMPLE PREDICTIONS")
print("=" * 50)

print(
    df[
        [
            "Complaint_Text",
            "Predicted_Sentiment",
            "Predicted_Emotion",
            "Emotion_Intensity"
        ]
    ].head(10).to_string(index=False)
)


print("\nOutput saved to:")
print(OUTPUT_PATH)

print("\nMODULE 7 NLP ANALYSIS COMPLETED SUCCESSFULLY!")