# Module 12 – Complaint Anomaly & Recurring Issue Detection

## Objective

Identify unusual complaint spikes and recurring customer problems.

## Dataset

- Complaints analyzed: 22,329
- Days analyzed: 931
- Anomalous days detected: 1
- Recurring issue patterns detected: 5

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
