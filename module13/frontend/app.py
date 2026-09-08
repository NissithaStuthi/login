import requests
import streamlit as st
import pandas as pd
import plotly.express as px


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ResolveIQ",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# API CONFIGURATION
# ============================================================

API_URL = "https://login-j0hk.onrender.com"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Hide Streamlit header */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    /* Hide toolbar */
    [data-testid="stToolbar"] {
        display: none !important;
    }

    /* Hide decoration */
    [data-testid="stDecoration"] {
        display: none !important;
    }

    /* Main page */
    .block-container {
        max-width: 1200px;
        padding-top: 20px !important;
        padding-bottom: 50px !important;
    }

    .stApp {
        background-color: #f7f5f0;
    }

    /* ResolveIQ logo */
    .brand {
        font-size: 40px;
        font-weight: 800;
        color: #173f3a;
        letter-spacing: -1.5px;
        line-height: 1.1;
    }

    .tagline {
        color: #747d7a;
        font-size: 14px;
        margin-top: 4px;
    }

    /* Online status */
    .online {
        background-color: #e7f4ee;
        color: #28735f;
        padding: 7px 13px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }

    /* Page headings */
    .page-title {
        color: #173f3a;
        font-size: 30px;
        font-weight: 800;
        margin-top: 25px;
        margin-bottom: 5px;
    }

    .page-description {
        color: #747d7a;
        font-size: 15px;
        margin-bottom: 22px;
    }

    /* Cards */
    .card {
        background-color: white;
        border: 1px solid #e8e5de;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 16px rgba(30, 45, 40, 0.04);
    }

    .card-title {
        color: #173f3a;
        font-size: 17px;
        font-weight: 750;
        margin-bottom: 14px;
    }

    /* Metric cards */
    .metric {
        background-color: white;
        border: 1px solid #e8e5de;
        border-radius: 15px;
        padding: 18px;
        min-height: 100px;
        box-shadow: 0 4px 16px rgba(30, 45, 40, 0.04);
    }

    .metric-label {
        color: #7d8582;
        font-size: 12px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #173f3a;
        font-size: 26px;
        font-weight: 800;
    }

    /* Result boxes */
    .result-box {
        background-color: #f1f7f4;
        border: 1px solid #dbeae4;
        border-radius: 13px;
        padding: 18px;
        line-height: 1.6;
        color: #34423e;
    }

    .resolution-box {
        background-color: #faf8f3;
        border: 1px solid #e9e3d8;
        border-radius: 13px;
        padding: 18px;
        line-height: 1.6;
        color: #3f4946;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 600;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #969d9a;
        font-size: 12px;
        padding-top: 35px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_backend():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=10
        )

        if response.status_code == 200:
            return response.json()

    except requests.RequestException:
        pass

    return None


def fetch_complaints():
    try:
        response = requests.get(
            f"{API_URL}/complaints",
            timeout=15
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("complaints", [])

    except requests.RequestException:
        pass

    return []


def safe_text(value, default="—"):

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if not text:
        return default

    return text


def show_metric(label, value):

    st.markdown(
        f"""
        <div class="metric">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [5, 1],
    vertical_alignment="center"
)


with header_left:

    st.markdown(
        '<div class="brand">ResolveIQ</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="tagline">'
        'AI Customer Complaint Intelligence & Resolution'
        '</div>',
        unsafe_allow_html=True
    )


with header_right:

    health = check_backend()

    if health:

        st.markdown(
            """
            <div style="text-align:right;">
                <span class="online">● System Online</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div style="text-align:right;">
                <span class="online">● Offline</span>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# TABS
# ============================================================

tab_overview, tab_analyze, tab_complaints, tab_insights = st.tabs(
    [
        "Overview",
        "Analyze Complaint",
        "Complaints",
        "Insights"
    ]
)


# ============================================================
# OVERVIEW TAB
# ============================================================

with tab_overview:

    st.markdown(
        '<div class="page-title">Complaint Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-description">'
        'A simple view of your customer complaint operations.'
        '</div>',
        unsafe_allow_html=True
    )

    complaints = fetch_complaints()
    df = pd.DataFrame(complaints)

    total = len(df)

    if not df.empty and "priority" in df.columns:

        critical = int(
            (df["priority"].astype(str).str.upper() == "P1").sum()
        )

        high = int(
            (df["priority"].astype(str).str.upper() == "P2").sum()
        )

    else:

        critical = 0
        high = 0

    if not df.empty and "status" in df.columns:

        open_cases = int(
            df["status"]
            .astype(str)
            .str.lower()
            .eq("open")
            .sum()
        )

    else:

        open_cases = 0


    m1, m2, m3, m4 = st.columns(4)


    with m1:
        show_metric(
            "Total Complaints",
            total
        )


    with m2:
        show_metric(
            "Critical Cases",
            critical
        )


    with m3:
        show_metric(
            "High Priority",
            high
        )


    with m4:
        show_metric(
            "Open Cases",
            open_cases
        )


    st.write("")


    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card-title">Recent Complaints</div>',
        unsafe_allow_html=True
    )


    if not df.empty:

        columns = [
            "complaint_id",
            "customer_id",
            "category",
            "priority",
            "recommended_department",
            "status"
        ]

        available_columns = [
            column
            for column in columns
            if column in df.columns
        ]

        if available_columns:

            st.dataframe(
                df[available_columns].head(8),
                width="stretch",
                hide_index=True
            )

        else:

            st.info(
                "Complaint records are available, "
                "but display fields were not found."
            )

    else:

        st.info(
            "No complaints have been submitted yet."
        )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# ANALYZE COMPLAINT TAB
# ============================================================

with tab_analyze:

    st.markdown(
        '<div class="page-title">'
        'Analyze Customer Complaint'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-description">'
        'Enter a complaint and let ResolveIQ understand, '
        'prioritize, route and recommend a resolution.'
        '</div>',
        unsafe_allow_html=True
    )


    input_col1, input_col2 = st.columns(2)


    with input_col1:

        complaint_id = st.text_input(
            "Complaint ID",
            value="UI-TEST-001"
        )


    with input_col2:

        customer_id = st.text_input(
            "Customer ID",
            value="CUS-001"
        )


    complaint_text = st.text_area(
        "Customer Complaint",
        height=160,
        placeholder=(
            "Example: I was charged twice and still haven't "
            "received my refund."
        )
    )


    analyze_button = st.button(
        "Analyze Complaint",
        type="primary",
        width="stretch"
    )


    if analyze_button:

        if not complaint_text.strip():

            st.warning(
                "Please enter a complaint before analyzing."
            )

        else:

            payload = {
                "complaint_id": complaint_id.strip(),
                "customer_id": customer_id.strip(),
                "complaint_text": complaint_text.strip()
            }


            try:

                with st.spinner(
                    "ResolveIQ is analyzing the complaint..."
                ):

                    response = requests.post(
                        f"{API_URL}/analyze",
                        json=payload,
                        timeout=180
                    )


                if response.status_code == 200:

                    result = response.json()

                    analysis = result.get(
                        "analysis",
                        {}
                    )


                    st.success(
                        "Complaint analyzed and saved successfully."
                    )


                    # ------------------------------------------------
                    # AI ANALYSIS
                    # ------------------------------------------------

                    st.markdown(
                        '<div class="card">',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        '<div class="card-title">AI Analysis</div>',
                        unsafe_allow_html=True
                    )


                    r1, r2, r3, r4 = st.columns(4)


                    with r1:

                        show_metric(
                            "Category",
                            safe_text(
                                analysis.get("category")
                            )
                        )


                    with r2:

                        show_metric(
                            "Sentiment",
                            safe_text(
                                analysis.get("sentiment")
                            )
                        )


                    with r3:

                        show_metric(
                            "Urgency",
                            safe_text(
                                analysis.get("urgency")
                            )
                        )


                    with r4:

                        show_metric(
                            "Priority",
                            safe_text(
                                analysis.get("priority")
                            )
                        )


                    r5, r6, r7, r8 = st.columns(4)


                    with r5:

                        show_metric(
                            "Emotion",
                            safe_text(
                                analysis.get("emotion")
                            )
                        )


                    with r6:

                        show_metric(
                            "Intent",
                            safe_text(
                                analysis.get("intent")
                            )
                        )


                    with r7:

                        show_metric(
                            "Department",
                            safe_text(
                                analysis.get(
                                    "recommended_department"
                                )
                            )
                        )


                    with r8:

                        anomaly = (
                            "Detected"
                            if analysis.get("is_anomaly")
                            else "No"
                        )

                        show_metric(
                            "Anomaly",
                            anomaly
                        )


                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )


                    # ------------------------------------------------
                    # ENTITIES
                    # ------------------------------------------------

                    entities = analysis.get(
                        "entities",
                        {}
                    )


                    actual_entities = {}

                    if isinstance(entities, dict):

                        for key, value in entities.items():

                            if value not in [
                                None,
                                "",
                                "null",
                                "Not Found"
                            ]:

                                actual_entities[key] = value


                    if actual_entities:

                        st.markdown(
                            '<div class="card">',
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            '<div class="card-title">'
                            'Extracted Information'
                            '</div>',
                            unsafe_allow_html=True
                        )


                        entity_rows = []


                        for key, value in actual_entities.items():

                            entity_rows.append(
                                {
                                    "Entity": key.replace(
                                        "_",
                                        " "
                                    ).title(),

                                    "Value": value
                                }
                            )


                        entity_df = pd.DataFrame(
                            entity_rows
                        )


                        st.dataframe(
                            entity_df,
                            width="stretch",
                            hide_index=True
                        )


                        st.markdown(
                            "</div>",
                            unsafe_allow_html=True
                        )

                    else:

                        st.info(
                            "No specific customer, order, payment, "
                            "date or location details were detected."
                        )


                    # ------------------------------------------------
                    # RESOLUTION + RESPONSE
                    # ------------------------------------------------

                    resolution_col, response_col = st.columns(2)


                    with resolution_col:

                        st.markdown(
                            '<div class="card">',
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            '<div class="card-title">'
                            'Recommended Resolution'
                            '</div>',
                            unsafe_allow_html=True
                        )


                        resolution = safe_text(
                            analysis.get(
                                "recommended_resolution"
                            ),
                            "No recommendation available."
                        )


                        st.markdown(
                            f"""
                            <div class="resolution-box">
                                {resolution}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                        st.markdown(
                            '<div class="card-title" '
                            'style="margin-top:18px;">'
                            'Routing'
                            '</div>',
                            unsafe_allow_html=True
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
                            unsafe_allow_html=True
                        )


                    with response_col:

                        st.markdown(
                            '<div class="card">',
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            '<div class="card-title">'
                            'AI Response'
                            '</div>',
                            unsafe_allow_html=True
                        )


                        ai_response = safe_text(
                            analysis.get(
                                "ai_response"
                            ),
                            "No response generated."
                        )


                        st.markdown(
                            f"""
                            <div class="result-box">
                                {ai_response}
                            </div>
                            """,
                            unsafe_allow_html=True
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
                            unsafe_allow_html=True
                        )


                else:

                    st.error(
                        f"API Error {response.status_code}"
                    )

                    st.code(
                        response.text
                    )


            except requests.RequestException as error:

                st.error(
                    "Could not connect to the ResolveIQ backend."
                )

                st.caption(
                    str(error)
                )


# ============================================================
# COMPLAINTS TAB
# ============================================================

with tab_complaints:

    st.markdown(
        '<div class="page-title">Complaint Records</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-description">'
        'All complaints currently stored in the database.'
        '</div>',
        unsafe_allow_html=True
    )


    complaints = fetch_complaints()
    df = pd.DataFrame(complaints)


    if not df.empty:

        search = st.text_input(
            "Search complaints",
            placeholder=(
                "Search complaint ID, customer, category "
                "or complaint text..."
            )
        )


        filtered_df = df.copy()


        if search.strip():

            search_value = search.strip()


            search_mask = (
                filtered_df
                .astype(str)
                .apply(
                    lambda column:
                    column.str.contains(
                        search_value,
                        case=False,
                        na=False
                    )
                )
                .any(axis=1)
            )


            filtered_df = filtered_df[
                search_mask
            ]


        st.write(
            f"**{len(filtered_df)}** complaint(s) found"
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
            "created_at"
        ]


        available_columns = [
            column
            for column in columns
            if column in filtered_df.columns
        ]


        if available_columns:

            st.dataframe(
                filtered_df[available_columns],
                width="stretch",
                hide_index=True
            )

        else:

            st.dataframe(
                filtered_df,
                width="stretch",
                hide_index=True
            )


    else:

        st.info(
            "No complaint records found."
        )


# ============================================================
# INSIGHTS TAB
# ============================================================

with tab_insights:

    st.markdown(
        '<div class="page-title">Complaint Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-description">'
        'Understand complaint patterns and operational trends.'
        '</div>',
        unsafe_allow_html=True
    )


    complaints = fetch_complaints()
    df = pd.DataFrame(complaints)


    if df.empty:

        st.info(
            "Submit complaints to generate insights."
        )

    else:

        # --------------------------------------------------------
        # CATEGORY CHART
        # --------------------------------------------------------

        if "category" in df.columns:

            category_data = (
                df["category"]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .reset_index()
            )


            category_data.columns = [
                "Category",
                "Count"
            ]


            fig_category = px.bar(
                category_data,
                x="Category",
                y="Count",
                title="Complaints by Category"
            )


            st.plotly_chart(
                fig_category,
                width="stretch"
            )


        # --------------------------------------------------------
        # SENTIMENT
        # --------------------------------------------------------

        left_chart, right_chart = st.columns(2)


        with left_chart:

            if "sentiment" in df.columns:

                sentiment_data = (
                    df["sentiment"]
                    .fillna("Unknown")
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )


                sentiment_data.columns = [
                    "Sentiment",
                    "Count"
                ]


                fig_sentiment = px.pie(
                    sentiment_data,
                    names="Sentiment",
                    values="Count",
                    title="Sentiment Distribution"
                )


                st.plotly_chart(
                    fig_sentiment,
                    width="stretch"
                )


        # --------------------------------------------------------
        # PRIORITY
        # --------------------------------------------------------

        with right_chart:

            if "priority" in df.columns:

                priority_data = (
                    df["priority"]
                    .fillna("Unknown")
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )


                priority_data.columns = [
                    "Priority",
                    "Count"
                ]


                fig_priority = px.bar(
                    priority_data,
                    x="Priority",
                    y="Count",
                    title="Priority Distribution"
                )


                st.plotly_chart(
                    fig_priority,
                    width="stretch"
                )


        # --------------------------------------------------------
        # DEPARTMENT
        # --------------------------------------------------------

        if "recommended_department" in df.columns:

            department_data = (
                df["recommended_department"]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .reset_index()
            )


            department_data.columns = [
                "Department",
                "Count"
            ]


            fig_department = px.bar(
                department_data,
                x="Department",
                y="Count",
                title="Department Routing"
            )


            st.plotly_chart(
                fig_department,
                width="stretch"
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
    unsafe_allow_html=True
)
```
