from typing import Optional

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from module13.database.database import engine, get_db
from module13.backend.ai_pipeline import analyze_complaint


app = FastAPI(
    title="CCAA - AI Customer Complaint Intelligence API",
    description="Backend API for the AI Customer Complaint Intelligence platform",
    version="1.0.0",
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ComplaintRequest(BaseModel):
    complaint_id: str
    customer_id: Optional[str] = None
    complaint_text: str


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "CCAA AI Customer Complaint Intelligence API is running",
        "status": "success",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return {
            "status": "healthy",
            "database": "connected",
        }

    except Exception as error:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(error),
        }


# ============================================================
# COMPLAINT API
# ============================================================

@app.post("/complaints")
def submit_complaint(
    complaint: ComplaintRequest,
    db: Session = Depends(get_db),
):
    try:

        query = text("""
            INSERT INTO complaints (
                complaint_id,
                customer_id,
                complaint_text,
                status
            )
            VALUES (
                :complaint_id,
                :customer_id,
                :complaint_text,
                'Open'
            )
            RETURNING id, complaint_id, customer_id,
                      complaint_text, status, created_at;
        """)

        result = db.execute(
            query,
            {
                "complaint_id": complaint.complaint_id,
                "customer_id": complaint.customer_id,
                "complaint_text": complaint.complaint_text,
            },
        )

        db.commit()

        row = result.fetchone()

        return {
            "message": "Complaint submitted successfully",
            "complaint": {
                "id": row.id,
                "complaint_id": row.complaint_id,
                "customer_id": row.customer_id,
                "complaint_text": row.complaint_text,
                "status": row.status,
                "created_at": row.created_at,
            },
        }

    except Exception as error:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Unable to save complaint: {str(error)}",
        )
        # ============================================================
# GET ALL COMPLAINTS
# ============================================================

@app.get("/complaints")
def get_complaints(db: Session = Depends(get_db)):
    try:
        query = text("""
            SELECT
                id,
                complaint_id,
                customer_id,
                complaint_text,
                category,
                sentiment,
                emotion,
                urgency,
                priority,
                intent,
                recommended_department,
                recommended_resolution,
                ai_response,
                status,
                created_at
            FROM complaints
            ORDER BY created_at DESC;
        """)

        result = db.execute(query)

        complaints = [
            dict(row._mapping)
            for row in result
        ]

        return {
            "count": len(complaints),
            "complaints": complaints
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to retrieve complaints: {str(error)}"
        )
        # ============================================================
# AI COMPLAINT ANALYSIS API
# ============================================================

@app.post("/analyze")
def analyze_customer_complaint(
    complaint: ComplaintRequest,
    db: Session = Depends(get_db),
):
    try:
        # Run AI pipeline
        result = analyze_complaint(
            complaint.complaint_text
        )

        entities = result.get("entities", {})

        # Save complete AI analysis to PostgreSQL
        query = text("""
            INSERT INTO complaints (
                complaint_id,
                customer_id,
                complaint_text,
                category,
                sentiment,
                emotion,
                emotion_intensity,
                urgency,
                priority,
                intent,
                customer_name,
                order_id,
                amount,
                complaint_date,
                transaction_id,
                location,
                account_type,
                recommended_department,
                recommended_resolution,
                ai_response,
                is_anomaly,
                recurring_issue,
                issue_type,
                status
            )
            VALUES (
                :complaint_id,
                :customer_id,
                :complaint_text,
                :category,
                :sentiment,
                :emotion,
                :emotion_intensity,
                :urgency,
                :priority,
                :intent,
                :customer_name,
                :order_id,
                :amount,
                :complaint_date,
                :transaction_id,
                :location,
                :account_type,
                :recommended_department,
                :recommended_resolution,
                :ai_response,
                :is_anomaly,
                :recurring_issue,
                :issue_type,
                'Open'
            )
            RETURNING id, complaint_id, created_at;
        """)

        result_db = db.execute(
            query,
            {
                "complaint_id": complaint.complaint_id,
                "customer_id": complaint.customer_id,
                "complaint_text": complaint.complaint_text,

                "category": result.get("category"),
                "sentiment": result.get("sentiment"),
                "emotion": result.get("emotion"),
                "emotion_intensity": result.get("emotion_intensity"),

                "urgency": result.get("urgency"),
                "priority": result.get("priority"),

                "intent": result.get("intent"),

                "customer_name": entities.get("customer_name"),
                "order_id": entities.get("order_id"),
                "amount": entities.get("amount"),
                "complaint_date": entities.get("date"),
                "transaction_id": entities.get("transaction_id"),
                "location": entities.get("location"),
                "account_type": entities.get("account_type"),

                "recommended_department":
                    result.get("recommended_department"),

                "recommended_resolution":
                    result.get("recommended_resolution"),

                "ai_response":
                    result.get("ai_response"),

                "is_anomaly":
                    result.get("is_anomaly", False),

                "recurring_issue":
                    result.get("recurring_issue"),

                "issue_type":
                    (
                        "Recurring Issue"
                        if result.get("recurring_issue")
                        else None
                    ),
            },
        )

        db.commit()

        saved = result_db.fetchone()

        return {
            "message": "Complaint analyzed and saved successfully",
            "database_record": {
                "id": saved.id,
                "complaint_id": saved.complaint_id,
                "created_at": saved.created_at,
            },
            "analysis": result,
        }

    except Exception as error:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Analysis and database save failed: {str(error)}",
        )
        # ============================================================
# MODULE 13 - INDIVIDUAL AI APIs
# ============================================================

from module13.backend.ai_pipeline import (
    classify_complaint,
    analyze_sentiment,
    analyze_emotion,
    detect_urgency,
    detect_intent,
    extract_entities,
    route_and_resolve,
    create_response,
    detect_anomaly_and_recurring,
)


# ============================================================
# CLASSIFICATION API
# ============================================================

@app.post("/classification")
def classification_api(complaint: ComplaintRequest):
    category = classify_complaint(
        complaint.complaint_text
    )

    return {
        "complaint_id": complaint.complaint_id,
        "category": category,
    }


# ============================================================
# SENTIMENT API
# ============================================================

@app.post("/sentiment")
def sentiment_api(complaint: ComplaintRequest):
    sentiment = analyze_sentiment(
        complaint.complaint_text
    )

    emotion, intensity = analyze_emotion(
        complaint.complaint_text
    )

    return {
        "complaint_id": complaint.complaint_id,
        "sentiment": sentiment,
        "emotion": emotion,
        "emotion_intensity": intensity,
    }


# ============================================================
# URGENCY API
# ============================================================

@app.post("/urgency")
def urgency_api(complaint: ComplaintRequest):
    urgency, priority = detect_urgency(
        complaint.complaint_text
    )

    return {
        "complaint_id": complaint.complaint_id,
        "urgency": urgency,
        "priority": priority,
    }


# ============================================================
# PRIORITY API
# ============================================================

@app.post("/priority")
def priority_api(complaint: ComplaintRequest):
    urgency, priority = detect_urgency(
        complaint.complaint_text
    )

    return {
        "complaint_id": complaint.complaint_id,
        "priority": priority,
        "urgency": urgency,
    }


# ============================================================
# INTENT API
# ============================================================

@app.post("/intent")
def intent_api(complaint: ComplaintRequest):
    intent = detect_intent(
        complaint.complaint_text
    )

    return {
        "complaint_id": complaint.complaint_id,
        "intent": intent,
    }


# ============================================================
# ENTITY API
# ============================================================

@app.post("/entities")
def entity_api(complaint: ComplaintRequest):
    entities = extract_entities(
        complaint.complaint_text
    )

    return {
        "complaint_id": complaint.complaint_id,
        "entities": entities,
    }


# ============================================================
# RESPONSE API
# ============================================================

@app.post("/response")
def response_api(complaint: ComplaintRequest):

    category = classify_complaint(
        complaint.complaint_text
    )

    sentiment = analyze_sentiment(
        complaint.complaint_text
    )

    urgency, _ = detect_urgency(
        complaint.complaint_text
    )

    intent = detect_intent(
        complaint.complaint_text
    )

    entities = extract_entities(
        complaint.complaint_text
    )

    department, resolution = route_and_resolve(
        category=category,
        intent=intent,
        complaint_text=complaint.complaint_text,
    )

    response = create_response(
        complaint_text=complaint.complaint_text,
        category=category,
        sentiment=sentiment,
        urgency=urgency,
        intent=intent,
        entities=entities,
        resolution=resolution,
    )

    return {
        "complaint_id": complaint.complaint_id,
        "ai_response": response,
    }


# ============================================================
# ROUTING API
# ============================================================

@app.post("/routing")
def routing_api(complaint: ComplaintRequest):

    category = classify_complaint(
        complaint.complaint_text
    )

    intent = detect_intent(
        complaint.complaint_text
    )

    department, _ = route_and_resolve(
        category=category,
        intent=intent,
        complaint_text=complaint.complaint_text,
    )

    return {
        "complaint_id": complaint.complaint_id,
        "category": category,
        "intent": intent,
        "recommended_department": department,
    }


# ============================================================
# RESOLUTION API
# ============================================================

@app.post("/resolution")
def resolution_api(complaint: ComplaintRequest):

    category = classify_complaint(
        complaint.complaint_text
    )

    intent = detect_intent(
        complaint.complaint_text
    )

    department, resolution = route_and_resolve(
        category=category,
        intent=intent,
        complaint_text=complaint.complaint_text,
    )

    return {
        "complaint_id": complaint.complaint_id,
        "recommended_department": department,
        "recommended_resolution": resolution,
    }


# ============================================================
# ANOMALY API
# ============================================================

@app.post("/anomaly")
def anomaly_api(complaint: ComplaintRequest):

    is_anomaly, recurring_issue = (
        detect_anomaly_and_recurring(
            complaint.complaint_text
        )
    )

    return {
        "complaint_id": complaint.complaint_id,
        "is_anomaly": is_anomaly,
        "recurring_issue": recurring_issue,
    }


# ============================================================
# INSIGHT API
# ============================================================

@app.get("/insights")
def insights_api(db: Session = Depends(get_db)):

    try:

        query = text("""
            SELECT
                COUNT(*) AS total_complaints,
                COUNT(*) FILTER (
                    WHERE priority = 'P1'
                ) AS critical_complaints,
                COUNT(*) FILTER (
                    WHERE priority = 'P2'
                ) AS high_priority_complaints,
                COUNT(*) FILTER (
                    WHERE status = 'Open'
                ) AS open_complaints
            FROM complaints;
        """)

        result = db.execute(query).fetchone()

        return {
            "total_complaints": result.total_complaints,
            "critical_complaints": result.critical_complaints,
            "high_priority_complaints": result.high_priority_complaints,
            "open_complaints": result.open_complaints,
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to generate insights: {str(error)}",
        )