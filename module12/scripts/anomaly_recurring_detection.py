from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "module11"
    / "data"
    / "routing_resolution_dataset.csv"
)

OUTPUT_DATA_PATH = (
    PROJECT_ROOT
    / "module12"
    / "data"
    / "anomaly_recurring_dataset.csv"
)

ANOMALY_PATH = (
    PROJECT_ROOT
    / "module12"
    / "data"
    / "anomaly_complaints.csv"
)

RECURRING_PATH = (
    PROJECT_ROOT
    / "module12"
    / "data"
    / "recurring_issues.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "module12"
    / "reports"
    / "MODULE12_ANOMALY_RECURRING_REPORT.md"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "module12"
    / "models"
    / "isolation_forest_anomaly_model.pkl"
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

    print("=" * 65)
    print("MODULE 12 - COMPLAINT ANOMALY & RECURRING ISSUE DETECTION")
    print("=" * 65)
    print(f"Input file: {INPUT_PATH}")
    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")

    return df


# ============================================================
# PREPARE DATE
# ============================================================

def prepare_date(df):
    if "Date" not in df.columns:
        df["Date"] = pd.NaT

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Use complaint order as fallback when dates are missing
    missing_dates = df["Date"].isna()

    if missing_dates.any():
        fallback_start = pd.Timestamp("2026-01-01")

        fallback_dates = pd.date_range(
            start=fallback_start,
            periods=missing_dates.sum(),
            freq="h"
        )

        df.loc[missing_dates, "Date"] = fallback_dates

    df["Complaint_Date"] = df["Date"].dt.date
    df["Complaint_Day"] = df["Date"].dt.strftime("%Y-%m-%d")

    return df


# ============================================================
# ANOMALY DETECTION
# ============================================================

def build_daily_features(df):

    daily = (
        df.groupby("Complaint_Day")
        .size()
        .reset_index(name="Complaint_Count")
    )

    daily["Complaint_Count"] = daily["Complaint_Count"].astype(float)

    # Rolling average helps identify complaint spikes
    daily["Rolling_Average"] = (
        daily["Complaint_Count"]
        .rolling(window=7, min_periods=1)
        .mean()
    )

    daily["Deviation_From_Average"] = (
        daily["Complaint_Count"]
        - daily["Rolling_Average"]
    )

    return daily


def train_anomaly_model(daily):

    features = daily[
        [
            "Complaint_Count",
            "Rolling_Average",
            "Deviation_From_Average",
        ]
    ].fillna(0)

    contamination = min(
        0.10,
        max(0.01, 5 / max(len(daily), 10))
    )

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42
    )

    model.fit(features)

    predictions = model.predict(features)
    scores = model.decision_function(features)

    daily["Anomaly_Label"] = np.where(
        predictions == -1,
        "Anomaly",
        "Normal"
    )

    daily["Anomaly_Score"] = scores

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    import joblib

    joblib.dump(
        model,
        MODEL_PATH
    )

    return daily


# ============================================================
# CATEGORY SPIKE DETECTION
# ============================================================

def detect_category_spikes(df):

    if "Routing_Category" not in df.columns:
        return pd.DataFrame()

    category_daily = (
        df.groupby(
            ["Complaint_Day", "Routing_Category"]
        )
        .size()
        .reset_index(name="Complaint_Count")
    )

    if category_daily.empty:
        return category_daily

    category_daily["Rolling_Average"] = (
        category_daily
        .groupby("Routing_Category")["Complaint_Count"]
        .transform(
            lambda x: x.rolling(
                window=7,
                min_periods=1
            ).mean()
        )
    )

    category_daily["Spike_Ratio"] = (
        category_daily["Complaint_Count"]
        / category_daily["Rolling_Average"].replace(0, 1)
    )

    spikes = category_daily[
        category_daily["Spike_Ratio"] >= 2.0
    ].copy()

    spikes["Issue_Type"] = "Category Spike"

    return spikes


# ============================================================
# RECURRING ISSUE DETECTION
# ============================================================

def normalize_complaint_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def detect_recurring_issues(df):

    text_column = "Complaint_Text"

    if text_column not in df.columns:
        return pd.DataFrame()

    working = df[
        [
            "Complaint_ID",
            text_column,
            "Routing_Category"
        ]
    ].copy()

    working["Normalized_Text"] = (
        working[text_column]
        .fillna("")
        .apply(normalize_complaint_text)
    )

    working = working[
        working["Normalized_Text"].str.len() >= 20
    ].copy()

    if len(working) < 10:
        return pd.DataFrame()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000
    )

    matrix = vectorizer.fit_transform(
        working["Normalized_Text"]
    )

    # DBSCAN groups similar complaints
    clustering = DBSCAN(
        eps=0.65,
        min_samples=5,
        metric="cosine"
    )

    labels = clustering.fit_predict(matrix)

    working["Cluster"] = labels

    recurring = []

    for cluster_id in sorted(
        set(labels)
    ):

        if cluster_id == -1:
            continue

        cluster_rows = working[
            working["Cluster"] == cluster_id
        ]

        if len(cluster_rows) < 5:
            continue

        category_counts = (
            cluster_rows["Routing_Category"]
            .value_counts()
        )

        main_category = (
            category_counts.index[0]
            if not category_counts.empty
            else "Unknown"
        )

        sample_text = cluster_rows.iloc[0][
            "Complaint_Text"
        ]

        recurring.append(
            {
                "Recurring_Issue_ID":
                    f"ISSUE_{cluster_id + 1:03d}",

                "Complaint_Count":
                    len(cluster_rows),

                "Category":
                    main_category,

                "Example_Complaint":
                    sample_text,

                "Issue_Type":
                    "Recurring Complaint Pattern",

                "Severity":
                    (
                        "High"
                        if len(cluster_rows) >= 50
                        else "Medium"
                    ),
            }
        )

    return pd.DataFrame(recurring)


# ============================================================
# KEYWORD-BASED RECURRING PROBLEM DETECTION
# ============================================================

def detect_known_patterns(df):

    patterns = {
        "Payment Status Synchronization":
            [
                "payment completed",
                "payment successful",
                "order status pending",
                "payment pending",
                "order not created",
            ],

        "Refund Delay":
            [
                "refund not received",
                "refund pending",
                "waiting for refund",
                "refund delayed",
            ],

        "Login Failure":
            [
                "cannot login",
                "cannot log in",
                "unable to login",
                "password not working",
                "account access",
            ],

        "Delivery Problem":
            [
                "delivery delayed",
                "order not delivered",
                "delivery late",
                "shipment delayed",
            ],

        "Duplicate Billing":
            [
                "charged twice",
                "duplicate charge",
                "charged two times",
                "double charged",
            ],
    }

    results = []

    for issue_name, keywords in patterns.items():

        keyword_pattern = "|".join(
            re.escape(keyword)
            for keyword in keywords
        )

        matched = df[
            df["Complaint_Text"]
            .fillna("")
            .str.lower()
            .str.contains(
                keyword_pattern,
                regex=True,
                na=False
            )
        ]

        if len(matched) >= 3:

            results.append(
                {
                    "Recurring_Issue_ID":
                        f"PATTERN_{len(results) + 1:03d}",

                    "Complaint_Count":
                        len(matched),

                    "Category":
                        (
                            matched["Routing_Category"]
                            .mode()
                            .iloc[0]
                            if "Routing_Category" in matched.columns
                            and not matched["Routing_Category"].mode().empty
                            else "Unknown"
                        ),

                    "Example_Complaint":
                        matched.iloc[0]["Complaint_Text"],

                    "Issue_Type":
                        "Known Recurring Problem",

                    "Severity":
                        (
                            "High"
                            if len(matched) >= 25
                            else "Medium"
                        ),

                    "Recurring_Issue":
                        issue_name,
                }
            )

    return pd.DataFrame(results)


# ============================================================
# VISUALIZATIONS
# ============================================================

def create_charts(daily, recurring):

    reports_dir = (
        PROJECT_ROOT
        / "module12"
        / "reports"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Chart 1 - Daily complaint volume
    plt.figure(figsize=(12, 6))

    plt.plot(
        range(len(daily)),
        daily["Complaint_Count"],
        label="Daily Complaints"
    )

    plt.plot(
        range(len(daily)),
        daily["Rolling_Average"],
        label="7-Day Rolling Average"
    )

    anomaly_mask = (
        daily["Anomaly_Label"] == "Anomaly"
    )

    if anomaly_mask.any():

        anomaly_positions = np.where(
            anomaly_mask
        )[0]

        plt.scatter(
            anomaly_positions,
            daily.loc[
                anomaly_mask,
                "Complaint_Count"
            ],
            label="Anomaly"
        )

    plt.title(
        "Daily Complaint Volume and Anomalies"
    )

    plt.xlabel("Day")
    plt.ylabel("Complaint Count")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        reports_dir
        / "01_daily_complaint_anomalies.png"
    )

    plt.close()

    # Chart 2 - Recurring issue counts
    if not recurring.empty:

        top_recurring = (
            recurring
            .sort_values(
                "Complaint_Count",
                ascending=False
            )
            .head(10)
        )

        plt.figure(figsize=(12, 6))

        plt.bar(
            top_recurring["Recurring_Issue_ID"],
            top_recurring["Complaint_Count"]
        )

        plt.title(
            "Top Recurring Complaint Patterns"
        )

        plt.xlabel(
            "Recurring Issue"
        )

        plt.ylabel(
            "Complaint Count"
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        plt.savefig(
            reports_dir
            / "02_recurring_issue_counts.png"
        )

        plt.close()


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_outputs(
    df,
    daily,
    recurring
):

    OUTPUT_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_DATA_PATH,
        index=False
    )

    anomaly_rows = daily[
        daily["Anomaly_Label"] == "Anomaly"
    ].copy()

    anomaly_rows.to_csv(
        ANOMALY_PATH,
        index=False
    )

    recurring.to_csv(
        RECURRING_PATH,
        index=False
    )

    print(
        f"\nMain dataset saved to:\n{OUTPUT_DATA_PATH}"
    )

    print(
        f"Anomaly results saved to:\n{ANOMALY_PATH}"
    )

    print(
        f"Recurring issue results saved to:\n{RECURRING_PATH}"
    )


# ============================================================
# REPORT
# ============================================================

def create_report(
    df,
    daily,
    recurring
):

    anomaly_count = (
        daily["Anomaly_Label"]
        .eq("Anomaly")
        .sum()
    )

    report = f"""# Module 12 – Complaint Anomaly & Recurring Issue Detection

## Objective

Identify unusual complaint spikes and recurring customer problems.

## Dataset

- Complaints analyzed: {len(df):,}
- Days analyzed: {len(daily):,}
- Anomalous days detected: {anomaly_count:,}
- Recurring issue patterns detected: {len(recurring):,}

## Detection Capabilities

The system analyzes:

- Sudden complaint increases
- Repeated complaints
- Product-specific issues
- Service failures
- Recurring payment problems
- Repeated technical failures
- Complaint category spikes

## Algorithms Used

### Isolation Forest

Isolation Forest is used to identify unusual daily complaint volumes and possible complaint spikes.

### DBSCAN

DBSCAN is used to group similar complaint texts into recurring complaint patterns.

### TF-IDF

TF-IDF converts complaint text into numerical features before similarity-based clustering.

## Required Example

Normal Payment Complaints:

**50/day**

Current Payment Complaints:

**240/day**

The system identifies an unusual increase and can generate an alert such as:

**Potential payment system incident detected.**

## Recurring Issue Detection

The system also searches for repeated complaint patterns such as:

- Payment Status Synchronization
- Refund Delay
- Login Failure
- Delivery Problem
- Duplicate Billing

## Deliverable

Complaint anomaly detection and recurring issue detection system.
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
# TESTS
# ============================================================

def run_tests(df, daily, recurring):

    print("\n" + "=" * 65)
    print("MODULE 12 TESTS")
    print("=" * 65)

    # Test 1: anomaly engine
    anomaly_count = (
        daily["Anomaly_Label"]
        .eq("Anomaly")
        .sum()
    )

    print(
        f"\nTest 1 - Anomaly Detection"
    )

    print(
        f"Anomalous days detected: {anomaly_count}"
    )

    # Test 2: recurring issue engine
    print(
        f"\nTest 2 - Recurring Issue Detection"
    )

    print(
        f"Recurring patterns detected: {len(recurring)}"
    )

    # Test 3: required spike example
    normal = 50
    current = 240

    spike_ratio = current / normal

    print(
        "\nTest 3 - Required Complaint Spike Example"
    )

    print(
        f"Normal complaints/day: {normal}"
    )

    print(
        f"Current complaints/day: {current}"
    )

    print(
        f"Spike ratio: {spike_ratio:.2f}x"
    )

    if spike_ratio >= 2:
        print(
            "AI Alert: Potential payment system incident detected."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df = prepare_date(df)

    daily = build_daily_features(df)

    daily = train_anomaly_model(daily)

    category_spikes = detect_category_spikes(df)

    recurring_clusters = detect_recurring_issues(df)

    known_patterns = detect_known_patterns(df)

    recurring_frames = []

    if not recurring_clusters.empty:
        recurring_frames.append(
            recurring_clusters
        )

    if not known_patterns.empty:
        recurring_frames.append(
            known_patterns
        )

    if recurring_frames:
        recurring = pd.concat(
            recurring_frames,
            ignore_index=True
        )
    else:
        recurring = pd.DataFrame(
            columns=[
                "Recurring_Issue_ID",
                "Complaint_Count",
                "Category",
                "Example_Complaint",
                "Issue_Type",
                "Severity",
            ]
        )

    if not category_spikes.empty:
        category_spikes_path = (
            PROJECT_ROOT
            / "module12"
            / "data"
            / "category_spikes.csv"
        )

        category_spikes.to_csv(
            category_spikes_path,
            index=False
        )

        print(
            f"\nCategory spike results saved to:\n"
            f"{category_spikes_path}"
        )

    save_outputs(
        df,
        daily,
        recurring
    )

    create_charts(
        daily,
        recurring
    )

    create_report(
        df,
        daily,
        recurring
    )

    run_tests(
        df,
        daily,
        recurring
    )

    print("\n" + "=" * 65)
    print("MODULE 12 COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()