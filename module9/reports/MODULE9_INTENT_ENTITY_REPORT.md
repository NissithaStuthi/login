# MODULE 9 – CUSTOMER INTENT & ENTITY EXTRACTION

## 1. Objective

The objective of this module is to identify what a customer wants from a complaint and extract important information from the complaint text.

## 2. Customer Intent Classes

- Refund Request
- Cancellation Request
- Payment Issue
- Account Recovery
- Delivery Tracking
- Product Replacement
- Complaint Escalation
- Technical Assistance
- Information Request

## 3. Extracted Entities

- Customer_Name
- Order_ID
- Product_Entity
- Amount
- Date_Entity
- Transaction_ID
- Location
- Account_Type
- Complaint_Category_Entity

## 4. Methodology

Customer intent is classified using a TF-IDF text representation combined with Logistic Regression. Entity extraction uses NLP-oriented pattern matching and structured information available in the complaint dataset.

## 5. Intent Model Performance

Intent classification accuracy on the held-out synthetic intent test set: **85.19%**.

The intent training examples are synthetic domain examples created from the required intent definitions. The accuracy therefore measures performance on this controlled intent dataset and should not be interpreted as human-annotated real-world benchmark accuracy.

## 6. Predicted Intent Distribution

| Intent | Complaints |
|---|---:|
| Account Recovery | 9,219 |
| Complaint Escalation | 4,179 |
| Technical Assistance | 2,176 |
| Payment Issue | 1,719 |
| Product Replacement | 1,668 |
| Information Request | 1,513 |
| Refund Request | 1,499 |
| Cancellation Request | 187 |
| Delivery Tracking | 169 |

## 7. Entity Extraction Coverage

| Entity | Found | Coverage |
|---|---:|---:|
| Customer_Name | 6,388 | 28.61% |
| Order_ID | 1,855 | 8.31% |
| Product_Entity | 22,329 | 100.00% |
| Amount | 7,956 | 35.63% |
| Date_Entity | 170 | 0.76% |
| Transaction_ID | 4,467 | 20.01% |
| Location | 6,869 | 30.76% |
| Account_Type | 3,243 | 14.52% |
| Complaint_Category_Entity | 22,329 | 100.00% |

## 8. Required Example

**Input:** I want a refund for order #ORD4567. ₹2,500 was charged on August 28.

- Order ID: ****
- Amount: **₹2,500**
- Date: **August 28**
- Intent: **Refund Request**

## 9. Lab Activity

An NLP-based customer intent and entity extraction system was developed and evaluated using test examples and a complaint dataset.

## 10. Deliverables

- Customer intent classification model
- Intent and entity extraction dataset
- Entity extraction coverage report
- Required example test
- Module 9 technical report
