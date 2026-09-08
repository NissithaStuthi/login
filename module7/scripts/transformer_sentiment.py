import pandas as pd
from pathlib import Path
from transformers import pipeline

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "module7"
    / "data"
    / "sentiment_emotion_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "module7" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "transformer_sentiment_results.csv"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

df["Complaint_Text"] = (
    df["Complaint_Text"]
    .fillna("")
    .astype(str)
)

# Use a sample for fast Transformer analysis
SAMPLE_SIZE = min(100, len(df))

sample_df = df.sample(
    n=SAMPLE_SIZE,
    random_state=42
).copy()

print("Total complaints:", len(df))
print("Transformer sample:", len(sample_df))


# --------------------------------------------------
# LOAD DISTILBERT MODEL
# --------------------------------------------------

print("\nLoading DistilBERT sentiment model...")

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

print("DistilBERT model loaded successfully.")


# --------------------------------------------------
# TRANSFORMER PREDICTION
# --------------------------------------------------

texts = sample_df["Complaint_Text"].tolist()

predictions = classifier(
    texts,
    truncation=True,
    max_length=256,
    batch_size=8
)


# --------------------------------------------------
# CONVERT LABELS
# --------------------------------------------------

sample_df["Transformer_Label"] = [
    prediction["label"]
    for prediction in predictions
]

sample_df["Transformer_Confidence"] = [
    round(prediction["score"], 4)
    for prediction in predictions
]


def convert_label(label):
    if label == "POSITIVE":
        return "Positive"

    return "Negative"


sample_df["Transformer_Sentiment"] = (
    sample_df["Transformer_Label"]
    .apply(convert_label)
)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

output_columns = [
    "Complaint_ID",
    "Complaint_Text",
    "Predicted_Sentiment",
    "Transformer_Sentiment",
    "Transformer_Confidence",
    "Predicted_Emotion",
    "Emotion_Intensity"
]

sample_df[output_columns].to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 50)
print("TRANSFORMER SENTIMENT DISTRIBUTION")
print("=" * 50)

print(
    sample_df["Transformer_Sentiment"]
    .value_counts()
)


print("\n" + "=" * 50)
print("TRANSFORMER CONFIDENCE")
print("=" * 50)

print(
    sample_df["Transformer_Confidence"]
    .describe()
)


print("\n" + "=" * 50)
print("SAMPLE TRANSFORMER RESULTS")
print("=" * 50)

print(
    sample_df[
        [
            "Complaint_Text",
            "Transformer_Sentiment",
            "Transformer_Confidence"
        ]
    ].head(10).to_string(index=False)
)


print("\nResults saved to:")
print(OUTPUT_PATH)

print("\nMODULE 7 TRANSFORMER ANALYSIS COMPLETED!")