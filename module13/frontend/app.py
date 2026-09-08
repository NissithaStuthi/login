```python
import requests
import streamlit as st
import pandas as pd
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ResolveIQ",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "https://login-j0hk.onrender.com"


# ============================================================
# GLOBAL STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       HIDE STREAMLIT DEFAULT HEADER / TOOLBAR
       ======================================================== */

    header[data-testid="stHeader"] {
        display: none !important;
    }

    [data-testid="stToolbar"] {
        display: none !important;
    }

    [data-testid="stDecoration"] {
        display: none !important;
    }

    /* Remove top white space */
    .block-container {
        max-width: 1180px;
        padding-top: 18px !important;
        padding-bottom: 50px !important;
    }


    /* ========================================================
       PAGE
       ======================================================== */

    .stApp {
        background: #f7f5f0;
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .brand {
        font-size: 38px;
        font-weight: 800;
        color: #173f3a;
        letter-spacing: -1.5px;
        line-height: 1.1;
    }

    .tagline {
        color: #727b78;
        font-size: 14px;
        margin-top: 2px;
    }

    .online {
        background: #e7f4ee;
        color: #28735f;
        padding: 7px 13px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }


    /* ========================================================
       TITLES
       ======================================================== */

    .page-title {
        color: #173f3a;
        font-size: 31px;
        font-weight: 800;
        margin-top: 25px;
        margin-bottom: 4px;
    }

    .page-description {
        color: #747d7a;
        font-size: 15px;
        margin-bottom: 24px;
    }


    /* ========================================================
       CARDS
       ======================================================== */

    .card {
        background: white;
        border: 1px solid #e8e5de;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 16px rgba(30, 45, 40, 0.04);
        margin-bottom: 18px;
    }

    .card-title {
        color: #173f3a;
        font-size: 17px;
        font-weight: 750;
        margin-bottom: 13px;
    }


    /* ========================================================
       METRICS
       ======================================================== */

    .metric {
        background: white;
        border: 1px solid #e8e5de;
        border-radius: 15px;
        padding: 18px;
        min-height: 105px;
        box-shadow: 0 4px 16px rgba(30, 45, 40, 0.04);
    }

    .metric-label {
        color: #7d8582;
        font-size: 12px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #173f3a;
        font-size: 27px;
        font-weight: 800;
    }


    /* ========================================================
       RESULT BOXES
       ======================================================== */

    .result-box {
        background: #f1f7f4;
        border: 1px solid #dbeae4;
        border-radius: 13px;
        padding: 18px;
        line-height: 1.6;
        color: #34423e;
    }

    .resolution-box {
        background: #faf8f3;
        border: 1px solid #e9e3d8;
        border-radius: 13px;
        padding: 18px;
        line-height: 1.6;
        color: #3f4946;
    }


    /* ========================================================
       PRIORITY
       ======================================================== */

    .priority-high {
        background: #fff3df;
        color: #9b6200;
        padding: 6px 11px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }

    .priority-critical {
        background: #ffe7e7;
        color: #a12d2d;
        padding: 6px 11px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }


    /* ========================================================
       TABS
       ======================================================== */

    button[data-baseweb="tab"] {
        font-weight: 600;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;
        color: #969d9a;
        font-size: 12px;
        padding-top: 35px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API HELPERS
# ============================================================

def check_backend():

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        if response.status_code == 200:
            return response.json()

    except requests.RequestException:
        return None

    return None


def fetch_complaints():

    try:

        response = requests.get(
            f"{API_URL}/complaints",
            timeout=10,
        )

        if response.status_code == 200:
            return response.json().get(
                "complaints",
                [],
            )

    except requests.RequestException:
        return []

    return []


def safe_text(value, default="—"):

    if value is None:
        return default

    if pd.isna(value):
        return default

    text = str(value).strip()

    return text if text else default


def metric(label, value):

    st.markdown(
        f"""
        <div class="metric">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================

left_header, right_header = st.columns(
    [5, 1],
    vertical_alignment="center",
)


with left_header:

    st.markdown(
        '<div class="brand">ResolveIQ</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="tagline">'
        'AI Customer Complaint Intelligence & Resolution'
        '</div>',
        unsafe_allow_html=True,
    )


with right_header:

    health = check_backend()

    if health and health.get("database") == "connected":

        st.markdown(
            '<div style="text-align:right;">'
            '<span class="online">● System Online</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            '<div style="text-align:right;">'
            '<span class="online">● Offline</span>'
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# NAVIGATION
# ============================================================

tab_overview, tab_analyze, tab_complaints, tab_insights = st.tabs(
    [
        "Overview",
        "Analyze Complaint",
        "Complaints",
        "Insights",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:

    st.markdown(
        '<div class="page-title">Complaint Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-description">'
        'A simple view of your customer complaint operations.'
        '</div>',
        unsafe_allow_html=True,
    )

    complaints = fetch_complaints()

    df = pd.DataFrame(complaints)

    total = len(df)

    if not df.empty and "priority" in df.columns:

        critical = int(
            (df["priority"] == "P1").sum()
        )

        high = int(
            (df["priority"] == "P2").sum()
        )

    else:

        critical = 0
        high = 0


    if not df.empty and "status" in df.columns:

        open_cases = int(
            (df["status"] == "Open").sum()
        )

    else:

        open_cases = 0


    m1, m2, m3, m4 = st.columns(4)


    with m1:
        metric(
            "Total Complaints",
            total,
        )


    with m2:
        metric(
            "Critical Cases",
            critical,
        )


    with m3:
        metric(
            "High Priority",
            high,
        )


    with m4:
        metric(
            "Open Cases",
            open_cases,
        )


    st.write("")


    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="card-title">Recent Complaints</div>',
        unsafe_allow_html=True,
    )


    if not df.empty:

        columns = [
            "complaint_id",
            "customer_id",
            "category",
            "priority",
            "recommended_department",
            "status",
        ]

        available = [
            col
            for col in columns
            if col in df.columns
        ]

        st.dataframe(
            df[available].head(8),
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "No complaints have been submitted yet."
        )


    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# ANALYZE COMPLAINT
# ============================================================

with tab_analyze:

    st.markdown(
        '<div class="page-title">'
        'Analyze Customer Complaint'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-description">'
        'Enter a complaint and let ResolveIQ understand, prioritize, '
        'route, and recommend a resolution.'
        '</div>',
        unsafe_allow_html=True,
    )


    col1, col2 = st.columns(2)


    with col1:

        complaint_id = st.text_input(
            "Complaint ID",
            value="UI-TEST-001",
        )


    with col2:

        customer_id = st.text_input(
            "Customer ID",
            value="CUS-001",
        )


    complaint_text = st.text_area(
        "Customer Complaint",
        height=160,
        placeholder=(
            "Example: I was charged twice and still haven't "
            "received my refund."
        ),
    )


    analyze = st.button(
        "Analyze Complaint",
        type="primary",
        width="stretch",
    )


    if analyze:

        if not complaint_text.strip():

            st.warning(
                "Please enter a complaint."
            )

        else:

            payload = {
                "complaint_id": complaint_id,
                "customer_id": customer_id,
                "complaint_text": complaint_text,
            }


            try:

                with st.spinner(
                    "Analyzing complaint..."
                ):

                    response = requests.post(
                        f"{API_URL}/analyze",
                        json=payload,
                        timeout=120,
                    )


                if response.status_code == 200:

                    data = response.json()

                    analysis = data.get(
                        "analysis",
                        {},
                    )


                    st.success(
                        "Complaint analyzed and saved successfully."
                    )


                    # ====================================================
                    # AI ANALYSIS
                    # ====================================================

                    st.markdown(
                        '<div class="card">',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        '<div class="card-title">'
                        'AI Analysis'
                        '</div>',
                        unsafe_allow_html=True,
                    )


                    r1, r2, r3, r4 = st.columns(4)


                    with r1:

                        metric(
                            "Category",
                            safe_text(
                                analysis.get("category")
                            ),
                        )


                    with r2:

                        metric(
                            "Sentiment",
                            safe_text(
                                analysis.get("sentiment")
                            ),
                        )


                    with r3:

                        metric(
                            "Urgency",
                            safe_text(
                                analysis.get("urgency")
                            ),
                        )


                    with r4:

                        metric(
                            "Priority",
                            safe_text(
                                analysis.get("priority")
                            ),
                        )


                    r5, r6, r7, r8 = st.columns(4)


                    with r5:

                        metric(
                            "Emotion",
                            safe_text(
                                analysis.get("emotion")
                            ),
                        )


                    with r6:

                        metric(
                            "Intent",
                            safe_text(
                                analysis.get("intent")
                            ),
                        )


                    with r7:

                        metric(
                            "Department",
                            safe_text(
                                analysis.get(
                                    "recommended_department"
                                )
                            ),
                        )


                    with r8:

                        anomaly = (
                            "Detected"
                            if analysis.get("is_anomaly")
                            else "No"
                        )

                        metric(
                            "Anomaly",
                            anomaly,
                        )


                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True,
                    )


                    # ====================================================
                    # EXTRACTED INFORMATION
                    # ====================================================

                    entities = analysis.get(
                        "entities",
                        {},
                    )


                    actual_entities = {
                        key: value
                        for key, value in entities.items()
                        if value not in [
                            None,
                            "",
                            "null",
                            "Not Found",
                        ]
                    }


                    if actual_entities:

                        st.markdown(
                            '<div class="card">',
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            '<div class="card-title">'
                            'Extracted Information'
                            '</div>',
                            unsafe_allow_html=True,
                        )


                        entity_data = pd.DataFrame(
                            [
                                {
                                    "Entity":
                                        key.replace(
                                            "_",
                                            " "
                                        ).title(),

                                    "Value":
                                        value,
                                }

                                for key, value
                                in actual_entities.items()
                            ]
                        )


                        st.dataframe(
                            entity_data,
                            width="stretch",
                            hide_index=True,
                        )


                        st.markdown(
                            "</div>",
                            unsafe_allow_html=True,
                        )


                    else:

                        st.info(
                            "No specific customer, order, payment, "
                            "date, or location details were detected."
                        )


                    # ====================================================
                    # RECOMMENDATION + RESPONSE
                    # ====================================================

                    left, right = st.columns(2)


                    with left:

                        st.markdown(
                            '<div class="card">',
                            unsafe_allow_html=True,
                        )


                        st.markdown(
                            '<div class="card-title">'
                            'Recommended Resolution'
                            '</div>',
                            unsafe_allow_html=True,
                        )


                        resolution = safe_text(
                            analysis.get(
                                "recommended_resolution"
                            ),
                            "No recommendation available.",
                        )


                        st.markdown(
                            f"""
                            <div class="resolution-box">
                                {resolution}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


                        st.markdown(
                            '<div class="card-title" '
                            'style="margin-top:18px;">'
                            'Routing'
                            '</div>',
                            unsafe_allow_html=True,
                        )


                        department = safe_text(
                            analysis.get(
                                "recommended_department"
                            )
                        )


                        st.write(
                            f"**Recommended Department:** "
                            f"{department}"
                        )


                        st.markdown(
                            "</div>",
                            unsafe_allow_html=True,
                        )


                    with right:

                        st.markdown(
                            '<div class="card">',
                            unsafe_allow_html=True,
                        )


                        st.markdown(
                            '<div class="card-title">'
                            'AI Response'
                            '</div>',
                            unsafe_allow_html=True,
                        )


                        ai_response = safe_text(
                            analysis.get(
                                "ai_response"
                            ),
                            "No response generated.",
                        )


                        st.markdown(
                            f"""
                            <div class="result-box">
                                {ai_response}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


                        recurring = analysis.get(
                            "recurring_issue"
                        )


                        if recurring:

                            st.warning(
                                f"Recurring issue detected: "
                                f"{recurring}"
                            )


                        st.markdown(
                            "</div>",
                            unsafe_allow_html=True,
                        )


                else:

                    st.error(
                        f"API Error {response.status_code}: "
                        f"{response.text}"
                    )


            except requests.RequestException as error:

                st.error(
                    f"Could not connect to FastAPI: {error}"
                )


# ============================================================
# COMPLAINT RECORDS
# ============================================================

with tab_complaints:

    st.markdown(
        '<div class="page-title">'
        'Complaint Records'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-description">'
        'All complaints currently stored in PostgreSQL.'
        '</div>',
        unsafe_allow_html=True,
    )


    complaints = fetch_complaints()

    df = pd.DataFrame(complaints)


    if not df.empty:

        search = st.text_input(
            "Search",
            placeholder=(
                "Search complaint ID, customer, category, "
                "or complaint text..."
            ),
        )


        filtered = df.copy()


        if search.strip():

            search_mask = (
                filtered
                .astype(str)
                .apply(
                    lambda column:
                    column.str.contains(
                        search,
                        case=False,
                        na=False,
                    )
                )
                .any(axis=1)
            )


            filtered = filtered[
                search_mask
            ]


        st.write(
            f"**{len(filtered)}** complaint(s)"
        )


        columns = [
            "complaint_id",
            "customer_id",
            "complaint_text",
            "category",
            "sentiment",
            "urgency",
            "priority",
            "intent",
            "recommended_department",
            "status",
            "created_at",
        ]


        available = [
            col
            for col in columns
            if col in filtered.columns
        ]


        st.dataframe(
            filtered[available],
            width="stretch",
            hide_index=True,
        )


    else:

        st.info(
            "No complaint records found."
        )


# ============================================================
# INSIGHTS
# ============================================================

with tab_insights:

    st.markdown(
        '<div class="page-title">'
        'Complaint Insights'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-description">'
        'Understand complaint patterns and operational trends.'
        '</div>',
        unsafe_allow_html=True,
    )


    complaints = fetch_complaints()

    df = pd.DataFrame(complaints)


    if df.empty:

        st.info(
            "Submit complaints to generate insights."
        )


    else:

        # ====================================================
        # CATEGORY
        # ====================================================

        if "category" in df.columns:

            category_data = (
                df["category"]
                .fillna("Unknown")
                .value_counts()
                .reset_index()
            )


            category_data.columns = [
                "Category",
                "Count",
            ]


            fig_category = px.bar(
                category_data,
                x="Category",
                y="Count",
                title="Complaints by Category",
            )


            st.plotly_chart(
                fig_category,
                width="stretch",
            )


        # ====================================================
        # SENTIMENT + PRIORITY
        # ====================================================

        left, right = st.columns(2)


        with left:

            if "sentiment" in df.columns:

                sentiment_data = (
                    df["sentiment"]
                    .fillna("Unknown")
                    .value_counts()
                    .reset_index()
                )


                sentiment_data.columns = [
                    "Sentiment",
                    "Count",
                ]


                fig_sentiment = px.pie(
                    sentiment_data,
                    names="Sentiment",
                    values="Count",
                    title="Sentiment Distribution",
                )


                st.plotly_chart(
                    fig_sentiment,
                    width="stretch",
                )


        with right:

            if "priority" in df.columns:

                priority_data = (
                    df["priority"]
                    .fillna("Unknown")
                    .value_counts()
                    .reset_index()
                )


                priority_data.columns = [
                    "Priority",
                    "Count",
                ]


                fig_priority = px.bar(
                    priority_data,
                    x="Priority",
                    y="Count",
                    title="Priority Distribution",
                )


                st.plotly_chart(
                    fig_priority,
                    width="stretch",
                )


        # ====================================================
        # DEPARTMENT
        # ====================================================

        if "recommended_department" in df.columns:

            department_data = (
                df["recommended_department"]
                .fillna("Unknown")
                .value_counts()
                .reset_index()
            )


            department_data.columns = [
                "Department",
                "Count",
            ]


            fig_department = px.bar(
                department_data,
                x="Department",
                y="Count",
                title="Department Routing",
            )


            st.plotly_chart(
                fig_department,
                width="stretch",
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        ResolveIQ · AI Customer Complaint Intelligence & Resolution
    </div>
    """,
    unsafe_allow_html=True,
)
```
