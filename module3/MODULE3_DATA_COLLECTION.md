# Module 3 – Customer Complaint Data Collection

## 1. Objective

The objective of Module 3 is to collect and organize customer complaint datasets
from multiple sources and prepare a unified raw customer complaint dataset for
further AI/NLP processing.

## 2. Data Sources

Two Kaggle datasets were selected for this project:

### Dataset 1 – Customer Support Tickets

Source:
Kaggle – Customer Support Ticket Dataset

Records collected:
8,469

Purpose:
General customer support and service-related complaints.

### Dataset 2 – Bank Account or Service Complaints

Source:
Kaggle – Bank Account or Service Complaints Dataset

Records collected:
84,811

Purpose:
Banking and financial service-related customer complaints.

## 3. Dataset Integration

The two datasets originally contained different column names and structures.

To create a common dataset structure, relevant fields from both datasets were
mapped to standardized complaint-intelligence fields.

The datasets were then combined vertically, meaning the rows from both datasets
were appended into one unified dataset.

## 4. Common Dataset Structure

The unified dataset contains the following fields:

1. Complaint_ID
2. Customer_ID
3. Complaint_Text
4. Category
5. Subcategory
6. Date
7. Channel
8. Product
9. Sentiment
10. Urgency
11. Priority
12. Department
13. Resolution
14. Status
15. Source_Dataset

## 5. Column Mapping

### Customer Support Dataset

| Original Field | Unified Field |
|---|---|
| Ticket ID | Complaint_ID |
| Ticket Description | Complaint_Text |
| Ticket Type | Category |
| Ticket Subject | Subcategory |
| Date of Purchase | Date |
| Ticket Channel | Channel |
| Product Purchased | Product |
| Ticket Priority | Priority |
| Resolution | Resolution |
| Ticket Status | Status |

### Banking Complaint Dataset

| Original Field | Unified Field |
|---|---|
| complaint_id | Complaint_ID |
| consumer_complaint_narrative | Complaint_Text |
| product | Category |
| sub_issue | Subcategory |
| date_received | Date |
| submitted_via | Channel |
| product | Product |
| company_response_to_consumer | Resolution |

Fields that were not available in the original datasets were left empty
instead of creating artificial values.

## 6. Final Dataset

The two datasets were combined as follows:

Customer Support Tickets: 8,469 records

Bank Complaints: 84,811 records

Total Unified Records: 93,280 records

Total Columns: 15

## 7. Output

The final raw dataset was saved as:

`data/raw/unified_customer_complaints_raw.csv`

The original Kaggle datasets were preserved separately in the raw-data folders.

## 8. Data Processing Status

At this stage, the dataset is considered RAW data.

No text cleaning, duplicate removal, missing-value treatment, tokenization,
lemmatization, or other NLP preprocessing has been performed.

These activities will be handled in the next preprocessing module.

## 9. Module 3 Deliverable

The final deliverable is the unified raw customer complaint dataset containing
93,280 complaint records collected from two different Kaggle data sources.