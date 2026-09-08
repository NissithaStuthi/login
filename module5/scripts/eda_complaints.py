import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# MODULE 5 - EXPLORATORY COMPLAINT DATA ANALYSIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "module4"
    / "data"
    / "processed"
    / "processed_customer_complaints.csv"
)

REPORT_PATH = PROJECT_ROOT / "module5" / "reports"
REPORT_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("MODULE 5 - EXPLORATORY COMPLAINT DATA ANALYSIS")
print("=" * 60)

print(f"\nDataset Shape: {df.shape}")

# ============================================================
# 1. CATEGORY ANALYSIS
# ============================================================

category_counts = df["Category"].value_counts()

print("\n1. COMPLAINT CATEGORY ANALYSIS")
print(category_counts)

print("\nCategory Percentages:")
print((category_counts / len(df) * 100).round(2))

plt.figure(figsize=(10, 6))
category_counts.plot(kind="bar")
plt.title("Customer Complaints by Category")
plt.xlabel("Complaint Category")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(REPORT_PATH / "01_category_analysis.png", dpi=300)
plt.close()

# ============================================================
# 2. SENTIMENT ANALYSIS
# ============================================================

sentiment_counts = df["Sentiment"].value_counts()

print("\n2. SENTIMENT DISTRIBUTION")
print(sentiment_counts)

print("\nSentiment Percentages:")
print((sentiment_counts / len(df) * 100).round(2))

plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind="bar")
plt.title("Customer Complaint Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(REPORT_PATH / "02_sentiment_distribution.png", dpi=300)
plt.close()

# ============================================================
# 3. URGENCY ANALYSIS
# ============================================================

urgency_counts = df["Urgency"].value_counts()

print("\n3. URGENCY DISTRIBUTION")
print(urgency_counts)

print("\nUrgency Percentages:")
print((urgency_counts / len(df) * 100).round(2))

plt.figure(figsize=(8, 6))
urgency_counts.plot(kind="bar")
plt.title("Complaint Urgency Distribution")
plt.xlabel("Urgency")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(REPORT_PATH / "03_urgency_distribution.png", dpi=300)
plt.close()

# ============================================================
# 4. PRIORITY ANALYSIS
# ============================================================

priority_counts = df["Priority"].value_counts()

print("\n4. PRIORITY ANALYSIS")
print(priority_counts)

print("\nPriority Percentages:")
print((priority_counts / len(df) * 100).round(2))

plt.figure(figsize=(8, 6))
priority_counts.plot(kind="bar")
plt.title("Customer Complaint Priority Distribution")
plt.xlabel("Priority")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(REPORT_PATH / "04_priority_distribution.png", dpi=300)
plt.close()

# ============================================================
# 5. DEPARTMENT ANALYSIS
# ============================================================

department_counts = df["Department"].value_counts().head(15)

print("\n5. DEPARTMENT-WISE COMPLAINTS")
print(department_counts)

plt.figure(figsize=(10, 6))
department_counts.plot(kind="bar")
plt.title("Department-wise Customer Complaints")
plt.xlabel("Department")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(REPORT_PATH / "05_department_analysis.png", dpi=300)
plt.close()

# ============================================================
# 6. PRODUCT ANALYSIS
# ============================================================

product_counts = df["Product"].value_counts().head(15)

print("\n6. PRODUCT-WISE COMPLAINTS")
print(product_counts)

plt.figure(figsize=(10, 6))
product_counts.plot(kind="bar")
plt.title("Product-wise Customer Complaints")
plt.xlabel("Product")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(REPORT_PATH / "06_product_analysis.png", dpi=300)
plt.close()

# ============================================================
# 7. CHANNEL ANALYSIS
# ============================================================

channel_counts = df["Channel"].value_counts()

print("\n7. CHANNEL-WISE COMPLAINTS")
print(channel_counts)

plt.figure(figsize=(9, 6))
channel_counts.plot(kind="bar")
plt.title("Channel-wise Customer Complaints")
plt.xlabel("Channel")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(REPORT_PATH / "07_channel_analysis.png", dpi=300)
plt.close()

# ============================================================
# 8. DAILY COMPLAINT TRENDS
# ============================================================

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

daily_complaints = df.dropna(subset=["Date"]).groupby(
    df["Date"].dt.date
).size()

print("\n8. DAILY COMPLAINT TREND")
print(daily_complaints.head(10))

plt.figure(figsize=(12, 6))
daily_complaints.plot()
plt.title("Daily Complaint Trend")
plt.xlabel("Date")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(REPORT_PATH / "08_daily_trend.png", dpi=300)
plt.close()

# ============================================================
# 9. MONTHLY COMPLAINT TRENDS
# ============================================================

monthly_complaints = df.dropna(subset=["Date"]).groupby(
    df["Date"].dt.to_period("M")
).size()

print("\n9. MONTHLY COMPLAINT TREND")
print(monthly_complaints)

plt.figure(figsize=(12, 6))
monthly_complaints.plot(marker="o")
plt.title("Monthly Complaint Trend")
plt.xlabel("Month")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(REPORT_PATH / "09_monthly_trend.png", dpi=300)
plt.close()

# ============================================================
# 10. KEYWORD ANALYSIS
# ============================================================

from collections import Counter

all_words = []

for text in df["Lemmatized_Text"].dropna():
    all_words.extend(str(text).split())

word_frequency = Counter(all_words)

# Remove very common generic words
common_words = {
    "customer",
    "please",
    "would",
    "could",
    "want",
    "need",
    "help",
    "get",
    "make",
    "tell"
}

filtered_words = {
    word: count
    for word, count in word_frequency.items()
    if word not in common_words and len(word) > 2
}

top_words = pd.Series(filtered_words).sort_values(
    ascending=False
).head(20)

print("\n10. TOP COMPLAINT KEYWORDS")
print(top_words)

plt.figure(figsize=(10, 7))
top_words.sort_values().plot(kind="barh")
plt.title("Top 20 Complaint Keywords")
plt.xlabel("Frequency")
plt.ylabel("Keyword")
plt.tight_layout()
plt.savefig(REPORT_PATH / "10_keyword_frequency.png", dpi=300)
plt.close()

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MODULE 5 EDA COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(f"\nCharts saved in:")
print(REPORT_PATH)

print("\nGenerated reports:")

for file in sorted(REPORT_PATH.glob("*.png")):
    print(f"✓ {file.name}")

print("\nTotal complaints analyzed:", len(df))
print("Total features analyzed:", len(df.columns))
# ============================================================
# FINAL STEP - GENERATE EDA REPORT
# ============================================================

# Recurring complaint analysis
recurring_complaints = (
    df[df["Complaint_Text"].notna()]
    .assign(Complaint_Text=df["Complaint_Text"].astype(str).str.strip())
    .query("Complaint_Text != ''")
    ["Complaint_Text"]
    .value_counts()
)

recurring_complaints = recurring_complaints[
    recurring_complaints > 1
].head(20)

# Save recurring complaints
recurring_complaints.to_csv(
    REPORT_PATH / "recurring_complaints.csv",
    header=["Frequency"]
)

# Helper function
def top_value(series):
    if len(series) == 0:
        return "No data"
    return f"{series.index[0]} ({series.iloc[0]:,})"

# Create Markdown report
report = f"""# Customer Complaint EDA Report

## 1. Dataset Overview

- Total complaints analyzed: **{len(df):,}**
- Total features analyzed: **{len(df.columns)}**
- Dataset source: Module 4 processed complaint dataset

## 2. Complaint Category Analysis

The most common complaint category is:

**{top_value(category_counts)}**

### Category Distribution

{category_counts.to_string()}

## 3. Sentiment Analysis

The most common sentiment is:

**{top_value(sentiment_counts)}**

### Sentiment Distribution

{sentiment_counts.to_string()}

## 4. Urgency Analysis

The most common urgency level is:

**{top_value(urgency_counts)}**

### Urgency Distribution

{urgency_counts.to_string()}

## 5. Priority Analysis

The most common priority level is:

**{top_value(priority_counts)}**

### Priority Distribution

{priority_counts.to_string()}

## 6. Department-wise Complaint Analysis

The department receiving the highest number of complaints is:

**{top_value(department_counts)}**

## 7. Product-wise Complaint Analysis

The product with the highest number of complaints is:

**{top_value(product_counts)}**

## 8. Channel-wise Complaint Analysis

The channel with the highest number of complaints is:

**{top_value(channel_counts)}**

## 9. Daily Complaint Trends

Daily complaint volumes were analyzed to identify changes in complaint activity over time.

The daily trend visualization is available in:

`08_daily_trend.png`

## 10. Monthly Complaint Trends

Monthly complaint volumes were analyzed to identify long-term complaint patterns.

The monthly trend visualization is available in:

`09_monthly_trend.png`

## 11. Recurring Complaint Analysis

Repeated complaint texts were identified to detect recurring customer issues.

Top recurring complaints:

{recurring_complaints.to_string()}

## 12. Keyword Analysis

The most frequently occurring complaint keywords were identified from the processed complaint text.

The visualization is available in:

`10_keyword_frequency.png`

## 13. EDA Visualizations

The following charts were generated:

1. Complaint Category Analysis
2. Sentiment Distribution
3. Urgency Distribution
4. Priority Distribution
5. Department-wise Complaints
6. Product-wise Complaints
7. Channel-wise Complaints
8. Daily Complaint Trend
9. Monthly Complaint Trend
10. Complaint Keyword Frequency

## 14. Conclusion

Exploratory Data Analysis was successfully performed on the customer complaint dataset.

The analysis identifies major complaint categories, sentiment patterns, urgency levels, priorities, departments, products, communication channels, complaint trends, recurring issues, and frequently used keywords.

These findings will be used as a foundation for the next AI/NLP stages of the project.
"""

with open(REPORT_PATH / "MODULE5_EDA_REPORT.md", "w", encoding="utf-8") as f:
    f.write(report)

print("\nEDA report generated successfully!")
print(f"Report saved to: {REPORT_PATH / 'MODULE5_EDA_REPORT.md'}")
print(f"Recurring complaints saved to: {REPORT_PATH / 'recurring_complaints.csv'}")

print("\n" + "=" * 60)
print("MODULE 5 COMPLETED SUCCESSFULLY!")
print("=" * 60)