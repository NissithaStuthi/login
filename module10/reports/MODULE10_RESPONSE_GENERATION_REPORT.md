# MODULE 10 – AI Automatic Response Generation

## Objective

Generate intelligent customer responses using complaint text,
category, sentiment, urgency, customer intent, extracted entities,
and recommended resolution information.

## Input Dataset

Module 9 intent and entity extraction dataset.

**Total complaints processed:** 22,329

## Response Generation Workflow

Customer Complaint
↓
Complaint Understanding
↓
Sentiment Analysis
↓
Intent Detection
↓
Entity Extraction
↓
Urgency & Priority Information
↓
Resolution Information
↓
AI Response Generator
↓
Customer Response

## Features Implemented

- Complaint-aware response generation
- Intent-specific responses
- Sentiment-aware wording
- Urgency-aware wording
- Customer name personalization
- Order ID extraction support
- Amount extraction support
- Date extraction support
- Transaction ID extraction support
- Recommended resolution integration
- Professional customer-support tone

## Output

Generated response dataset:

`module10/data/generated_customer_responses.csv`

## Example

**Complaint:**

My payment was deducted but my order was cancelled. I have been
waiting for my refund for 7 days.

**Intent:**

Refund Request

**Urgency:**

High

**Generated Response:**

Your refund request has been identified and will be reviewed
by the payments/refunds team.

The issue has been marked as high priority for quick assistance.

## Conclusion

Module 10 successfully implements an automatic customer response
generation system that combines complaint intelligence outputs
from previous modules to produce structured and personalized
customer-support responses.
