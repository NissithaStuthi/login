# Module 11 – Complaint Routing & Resolution Recommendation

## Objective

Automatically determine the appropriate department for each customer complaint and recommend an appropriate resolution.

## Dataset

- Complaints processed: 22,329
- Routing categories: 1
- Recommended departments: 4

## Department Routing Distribution

- **Customer Support**: 12,741 complaints (57.06%)
- **Security**: 4,441 complaints (19.89%)
- **Payments**: 4,020 complaints (18.00%)
- **Finance**: 1,127 complaints (5.05%)

## Category Distribution

- **General Inquiry**: 22,329 complaints (100.00%)

## Routing Logic

The system combines complaint category, customer intent, and complaint text to determine the appropriate department.

### Department Mapping

| Complaint Type | Department |
|---|---|
| Payment | Payments |
| Refund | Finance |
| Delivery | Logistics |
| Product Issue | Product Support |
| Login Issue | Technical Support |
| Account Issue | Account Support |
| Security Issue | Security |
| Subscription | Billing |
| General Query | Customer Support |

## Resolution Recommendation

The system generates a recommended action based on the complaint category and important complaint keywords.

Examples include:

- Payment → Verify transaction and order status.
- Refund → Verify refund status and process/escalate the refund.
- Delivery → Check shipment and coordinate with logistics.
- Product → Arrange replacement, return, or repair.
- Security → Secure account and escalate to security.
- Billing → Verify charges and correct billing issues.

## Required Example

**Complaint:**  
"My payment was successful but the order was not created."

Expected routing:

- Category: Payment
- Department: Payments
- Priority: High
- Resolution: Verify transaction and initiate order creation or refund.

## Module 11 Deliverable

Automated complaint routing and resolution recommendation system.
