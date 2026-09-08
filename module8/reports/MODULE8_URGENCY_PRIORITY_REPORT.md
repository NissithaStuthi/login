# MODULE 8 – COMPLAINT URGENCY & PRIORITY DETECTION

## 1. Objective

The objective of this module is to determine how urgently a customer complaint should be handled and assign an appropriate priority level.

## 2. Urgency Levels

- **Critical:** Security, financial loss, account compromise or severe service disruption.
- **High:** Issue requiring quick intervention.
- **Medium:** Issue affecting customer experience.
- **Low:** General questions or minor issues.

## 3. Priority Mapping

| Priority | Urgency | Meaning |
|---|---|---|
| P1 | Critical | Immediate attention |
| P2 | High | Quick intervention |
| P3 | Medium | Normal priority |
| P4 | Low | Low priority |

## 4. Input Features

- Category
- Predicted_Sentiment
- Predicted_Emotion
- Emotion_Intensity
- Urgency_Keyword_Count
- Customer_Impact
- Financial_Impact
- Account_Status
- Complaint_History_Count

## 5. AI Methodology

The system combines rule-based domain knowledge with a Random Forest machine learning classifier. Sentiment and emotion information from Module 7 are incorporated into the urgency prediction features.

The urgency labels are generated using transparent domain rules based on security risk, financial impact, customer impact, sentiment, emotion and urgency keywords. The Random Forest model learns these patterns and provides the final ML urgency prediction.

## 6. Model Performance

- Random Forest accuracy: **99.18%**
- Training/test split: 80/20
- Number of estimators: 200
- Random state: 42

## 7. Urgency Distribution

- **Critical:** 3,842 (4.12%)
- **High:** 4,468 (4.79%)
- **Medium:** 13,527 (14.50%)
- **Low:** 71,443 (76.59%)

## 8. Priority Distribution

- **P1:** 3,842 (4.12%)
- **P2:** 4,468 (4.79%)
- **P3:** 13,527 (14.50%)
- **P4:** 71,443 (76.59%)

## 9. Department Routing

- **Account Support:** 76,510
- **Payments & Refunds:** 5,016
- **Security & Fraud:** 4,718
- **Product Support:** 3,655
- **Technical Support:** 3,323
- **Delivery Support:** 58

## 10. Example

**Complaint:**

> Someone has accessed my account and made an unauthorized transaction.

**Expected Urgency:** Critical

**Expected Priority:** P1

**Expected Department:** Security & Fraud

## 11. Output Files

- Urgency and priority prediction dataset
- Random Forest ML model
- Urgency distribution chart
- Priority distribution chart
- Department distribution chart
- Confusion matrix
- Module 8 report
