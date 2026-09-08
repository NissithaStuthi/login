import pandas as pd
import re
from pathlib import Path
# ==========================================
# MODULE 4 - CUSTOMER COMPLAINT PREPROCESSING
# ==========================================

# Path to Module 3 raw dataset
# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Path to Module 3 raw dataset
RAW_DATA_PATH = PROJECT_ROOT / "module3" / "data" / "raw" / "unified_customer_complaints_raw.csv"

# Load dataset
df = pd.read_csv(RAW_DATA_PATH)

# Display basic information
print("=" * 60)
print("MODULE 4 - RAW DATA INSPECTION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Records:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Complaint IDs:")
print(df["Complaint_ID"].duplicated().sum())

print("\nData Types:")
print(df.dtypes)

print("\nRaw dataset loaded successfully!")
# ==========================================
# STEP 2 - MISSING VALUES & TEXT CLEANING
# ==========================================

print("\n" + "=" * 60)
print("STEP 2 - MISSING VALUE HANDLING & TEXT CLEANING")
print("=" * 60)

# ------------------------------------------
# 1. Handle missing complaint text
# ------------------------------------------

df["Complaint_Text"] = df["Complaint_Text"].fillna("").astype(str)

# Create a flag to identify records that originally had text
df["Text_Available"] = df["Complaint_Text"].str.strip().ne("")

# ------------------------------------------
# 2. Handle missing categorical values
# ------------------------------------------

categorical_columns = [
    "Customer_ID",
    "Category",
    "Subcategory",
    "Channel",
    "Product",
    "Sentiment",
    "Urgency",
    "Priority",
    "Department",
    "Resolution",
    "Status"
]

for column in categorical_columns:
    if column in df.columns:
        df[column] = df[column].fillna("Unknown")

# ------------------------------------------
# 3. Basic text cleaning function
# ------------------------------------------

def clean_text(text):
    text = str(text)
    text = re.sub(r"\{[^}]+\}", " ", text)
    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Apply cleaning
df["Cleaned_Complaint_Text"] = df["Complaint_Text"].apply(clean_text)

# ------------------------------------------
# 4. Remove duplicate complaint records
# ------------------------------------------

before_duplicates = len(df)

# Remove duplicate Complaint IDs
df = df.drop_duplicates(subset=["Complaint_ID"])

after_duplicates = len(df)

print(f"\nDuplicate records removed: {before_duplicates - after_duplicates}")

# ------------------------------------------
# 5. Create basic text features
# ------------------------------------------

df["Word_Count"] = df["Cleaned_Complaint_Text"].apply(
    lambda x: len(x.split())
)

df["Character_Count"] = df["Cleaned_Complaint_Text"].apply(
    len
)

df["Question_Detected"] = df["Complaint_Text"].str.contains(
    r"\?",
    regex=True,
    na=False
)

# ------------------------------------------
# 6. Display results
# ------------------------------------------

print("\nMissing values after handling:")
print(df.isnull().sum())

print("\nText cleaning examples:")

sample = df[df["Text_Available"] == True].head(5)

for _, row in sample.iterrows():
    print("\nOriginal:")
    print(row["Complaint_Text"])

    print("Cleaned:")
    print(row["Cleaned_Complaint_Text"])

print("\nText features created:")
print("✓ Word_Count")
print("✓ Character_Count")
print("✓ Question_Detected")

print("\nSTEP 2 COMPLETED SUCCESSFULLY!")
# ==========================================
# STEP 3 - NLP TEXT PREPROCESSING
# ==========================================

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

print("\n" + "=" * 60)
print("STEP 3 - NLP PREPROCESSING")
print("=" * 60)

# Download required NLTK resources
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Initialize NLP tools
stop_words = set(stopwords.words("english"))

# Keep negation words because they can change the meaning
negation_words = {
    "no",
    "not",
    "nor",
    "never",
    "neither",
    "n't"
}

stop_words = stop_words - negation_words
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


# ------------------------------------------
# 1. Tokenization
# ------------------------------------------

def tokenize_text(text):
    if not text.strip():
        return []
    
    return word_tokenize(text)


df["Tokens"] = df["Cleaned_Complaint_Text"].apply(tokenize_text)


# ------------------------------------------
# 2. Stopword Removal
# ------------------------------------------

def remove_stopwords(tokens):
    return [
        word for word in tokens
        if word.lower() not in stop_words
    ]


df["Tokens_No_Stopwords"] = df["Tokens"].apply(remove_stopwords)


# ------------------------------------------
# 3. Lemmatization
# ------------------------------------------

def lemmatize_tokens(tokens):
    return [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]


df["Lemmatized_Text"] = df["Tokens_No_Stopwords"].apply(
    lambda tokens: " ".join(lemmatize_tokens(tokens))
)


# ------------------------------------------
# 4. Stemming
# ------------------------------------------

def stem_tokens(tokens):
    return [
        stemmer.stem(word)
        for word in tokens
    ]


df["Stemmed_Text"] = df["Tokens_No_Stopwords"].apply(
    lambda tokens: " ".join(stem_tokens(tokens))
)


# ------------------------------------------
# 5. Token count feature
# ------------------------------------------

df["Token_Count"] = df["Tokens_No_Stopwords"].apply(len)


# ------------------------------------------
# Display NLP results
# ------------------------------------------

print("\nNLP preprocessing examples:")

sample = df[df["Text_Available"] == True].head(5)

for _, row in sample.iterrows():

    print("\nOriginal:")
    print(row["Complaint_Text"])

    print("\nTokens:")
    print(row["Tokens"])

    print("\nAfter Stopword Removal:")
    print(row["Tokens_No_Stopwords"])

    print("\nLemmatized:")
    print(row["Lemmatized_Text"])

    print("\nStemmed:")
    print(row["Stemmed_Text"])

    print("-" * 60)


print("\nNLP features created:")
print("✓ Tokens")
print("✓ Tokens_No_Stopwords")
print("✓ Lemmatized_Text")
print("✓ Stemmed_Text")
print("✓ Token_Count")

print("\nSTEP 3 COMPLETED SUCCESSFULLY!")
print("\n" + "=" * 60)
print("MODULE 4 - CURRENT PROGRESS CHECK")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nCurrent Columns:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

print("\nSample Processed Text:")
print(
    df[
        [
            "Complaint_Text",
            "Cleaned_Complaint_Text",
            "Lemmatized_Text",
            "Stemmed_Text"
        ]
    ].head(10).to_string(index=False)
)

print("\nText Availability:")
print(df["Text_Available"].value_counts(dropna=False))

print("\nQuestion Detection:")
print(df["Question_Detected"].value_counts(dropna=False))

print("\nMissing Values in Current Dataset:")
print(df.isnull().sum())

print("\n" + "=" * 60)
print("CURRENT MODULE 4 CHECK COMPLETE")
print("=" * 60)
# ============================================================
# STEP 4 - FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("STEP 4 - FEATURE ENGINEERING")
print("=" * 60)

# ------------------------------------------------------------
# 1. Complaint Length Features
# ------------------------------------------------------------

df["Complaint_Length"] = df["Cleaned_Complaint_Text"].str.len()

df["Sentence_Count"] = (
    df["Complaint_Text"]
    .fillna("")
    .astype(str)
    .str.count(r"[.!?]+")
)

# ------------------------------------------------------------
# 2. Keyword Lists
# ------------------------------------------------------------

positive_words = {
    "good", "great", "excellent", "happy", "satisfied",
    "helpful", "thanks", "thank", "resolved", "perfect"
}

negative_words = {
    "bad", "poor", "worst", "terrible", "angry",
    "disappointed", "failed", "failure", "problem",
    "issue", "error", "wrong", "broken", "unable",
    "cancelled", "cancel", "delay", "damaged"
}

emotional_words = {
    "angry", "upset", "frustrated", "disappointed",
    "worried", "sad", "annoyed", "hate", "furious"
}

urgency_words = {
    "urgent", "urgently", "immediately", "asap",
    "emergency", "critical", "quickly", "today",
    "now"
}

# ------------------------------------------------------------
# 3. Keyword Counting Function
# ------------------------------------------------------------

def count_keywords(text, keywords):
    words = set(text.split())
    return sum(1 for word in keywords if word in words)


df["Positive_Word_Count"] = df["Lemmatized_Text"].apply(
    lambda text: count_keywords(text, positive_words)
)

df["Negative_Word_Count"] = df["Lemmatized_Text"].apply(
    lambda text: count_keywords(text, negative_words)
)

df["Emotional_Word_Count"] = df["Lemmatized_Text"].apply(
    lambda text: count_keywords(text, emotional_words)
)

df["Urgency_Keyword_Count"] = df["Lemmatized_Text"].apply(
    lambda text: count_keywords(text, urgency_words)
)

# ------------------------------------------------------------
# 4. Sentiment Score
# ------------------------------------------------------------

df["Sentiment_Score"] = (
    df["Positive_Word_Count"]
    - df["Negative_Word_Count"]
)

# ------------------------------------------------------------
# 5. Complaint Category Indicators
# ------------------------------------------------------------

df["Payment_Keyword"] = df["Lemmatized_Text"].str.contains(
    r"\b(payment|paid|pay|transaction|charge|charged)\b",
    regex=True,
    na=False
).astype(int)

df["Refund_Keyword"] = df["Lemmatized_Text"].str.contains(
    r"\b(refund|moneyback|reimbursement)\b",
    regex=True,
    na=False
).astype(int)

df["Delivery_Keyword"] = df["Lemmatized_Text"].str.contains(
    r"\b(delivery|delivered|shipping|shipment|courier|arrived)\b",
    regex=True,
    na=False
).astype(int)

df["Account_Keyword"] = df["Lemmatized_Text"].str.contains(
    r"\b(account|login|password|username|profile)\b",
    regex=True,
    na=False
).astype(int)

df["Technical_Issue_Keyword"] = df["Lemmatized_Text"].str.contains(
    r"\b(error|bug|technical|crash|broken|unable|failure|issue)\b",
    regex=True,
    na=False
).astype(int)

# ------------------------------------------------------------
# 6. Display Created Features
# ------------------------------------------------------------

feature_columns = [
    "Complaint_Length",
    "Sentence_Count",
    "Positive_Word_Count",
    "Negative_Word_Count",
    "Emotional_Word_Count",
    "Urgency_Keyword_Count",
    "Sentiment_Score",
    "Payment_Keyword",
    "Refund_Keyword",
    "Delivery_Keyword",
    "Account_Keyword",
    "Technical_Issue_Keyword"
]

print("\nCreated Feature Columns:")

for column in feature_columns:
    print(f"✓ {column}")

# ------------------------------------------------------------
# 7. Feature Summary
# ------------------------------------------------------------

print("\nFeature Summary:")
print(df[feature_columns].describe())

print("\n" + "=" * 60)
print("STEP 4 FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
print("=" * 60)
# ============================================================
# STEP 5 - SAVE PROCESSED DATASET
# ============================================================

OUTPUT_PATH = PROJECT_ROOT / "module4" / "data" / "processed" / "processed_customer_complaints.csv"

df.to_csv(OUTPUT_PATH, index=False)

print("\nProcessed dataset saved successfully!")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Final dataset shape: {df.shape}")