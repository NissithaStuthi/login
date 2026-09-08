import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "module7"
    / "data"
    / "sentiment_emotion_dataset.csv"
)

REPORT_DIR = PROJECT_ROOT / "module7" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

print("Dataset:", df.shape)


# -----------------------------
# SENTIMENT CHART
# -----------------------------

sentiment_counts = df["Predicted_Sentiment"].value_counts()

plt.figure(figsize=(8, 5))
sentiment_counts.plot(kind="bar")
plt.title("Customer Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Complaints")
plt.tight_layout()

plt.savefig(REPORT_DIR / "01_sentiment_distribution.png")
plt.close()


# -----------------------------
# EMOTION CHART
# -----------------------------

emotion_counts = df["Predicted_Emotion"].value_counts()

plt.figure(figsize=(10, 6))
emotion_counts.plot(kind="bar")
plt.title("Customer Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(REPORT_DIR / "02_emotion_distribution.png")
plt.close()


# -----------------------------
# INTENSITY CHART
# -----------------------------

intensity_counts = df["Emotion_Intensity"].value_counts()

plt.figure(figsize=(8, 5))
intensity_counts.plot(kind="bar")
plt.title("Emotion Intensity Distribution")
plt.xlabel("Intensity")
plt.ylabel("Number of Complaints")
plt.tight_layout()

plt.savefig(REPORT_DIR / "03_emotion_intensity.png")
plt.close()


# -----------------------------
# SENTIMENT + EMOTION TABLE
# -----------------------------

cross_table = pd.crosstab(
    df["Predicted_Sentiment"],
    df["Predicted_Emotion"]
)

cross_table.to_csv(
    REPORT_DIR / "sentiment_emotion_cross_table.csv"
)


# -----------------------------
# GENERATE REPORT
# -----------------------------

report_path = REPORT_DIR / "MODULE7_SENTIMENT_EMOTION_REPORT.md"

with open(report_path, "w", encoding="utf-8") as file:

    file.write("# MODULE 7 – SENTIMENT & EMOTION ANALYSIS\n\n")

    file.write("## 1. Objective\n\n")
    file.write(
        "The objective of this module is to analyze customer complaints "
        "and determine customer sentiment and emotional state using "
        "NLP-based keyword analysis.\n\n"
    )

    file.write("## 2. Dataset\n\n")
    file.write(
        f"- Total complaints analyzed: {len(df):,}\n"
        f"- Sentiment classes: Positive, Neutral, Negative\n"
        f"- Emotion classes: Anger, Frustration, Disappointment, Fear, "
        f"Sadness, Satisfaction, Neutral\n\n"
    )

    file.write("## 3. Sentiment Distribution\n\n")

    for label, count in sentiment_counts.items():
        percentage = count / len(df) * 100
        file.write(
            f"- **{label}:** {count:,} ({percentage:.2f}%)\n"
        )

    file.write("\n## 4. Emotion Distribution\n\n")

    for label, count in emotion_counts.items():
        percentage = count / len(df) * 100
        file.write(
            f"- **{label}:** {count:,} ({percentage:.2f}%)\n"
        )

    file.write("\n## 5. Emotion Intensity\n\n")

    for label, count in intensity_counts.items():
        percentage = count / len(df) * 100
        file.write(
            f"- **{label}:** {count:,} ({percentage:.2f}%)\n"
        )

    file.write("\n## 6. Methodology\n\n")
    file.write(
        "The complaint text was analyzed using predefined positive, "
        "negative, and emotion-related keyword patterns. Each complaint "
        "was assigned a sentiment, emotion, and emotion intensity level.\n\n"
    )

    file.write("## 7. Output Files\n\n")
    file.write(
        "- Sentiment distribution chart\n"
        "- Emotion distribution chart\n"
        "- Emotion intensity chart\n"
        "- Sentiment-emotion cross table\n"
        "- Module 7 analysis report\n"
    )


print("\nSENTIMENT DISTRIBUTION")
print(sentiment_counts)

print("\nEMOTION DISTRIBUTION")
print(emotion_counts)

print("\nEMOTION INTENSITY")
print(intensity_counts)

print("\nReports saved in:")
print(REPORT_DIR)

print("\nMODULE 7 COMPLETED SUCCESSFULLY!")