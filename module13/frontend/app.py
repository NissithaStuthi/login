import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import uuid
import html


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
# SESSION STATE
# ============================================================

if "theme" not in st.session_state:
    st.session_state.theme = "Bright"

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ============================================================
# THEME COLORS
# ============================================================

if st.session_state.theme == "Dark":

    BG = "#101817"
    CARD = "#182321"
    CARD_BORDER = "#344842"

    TEXT = "#F5FAF8"
    HEADING = "#E5F5F0"
    MUTED = "#B7C7C2"

    INPUT_BG = "#202D2A"
    INPUT_BORDER = "#465B54"

    RESULT_BG = "#18302B"
    RESULT_BORDER = "#3C7466"

    RESOLUTION_BG = "#292820"
    RESOLUTION_BORDER = "#5A5545"

    TEAL = "#65D2B7"

    ONLINE_BG = "#173D34"
    ONLINE_TEXT = "#76E2C9"

    SHADOW = "rgba(0,0,0,0.30)"

    TAB_INACTIVE_BG = "#1E2B28"
    TAB_INACTIVE_TEXT = "#E8F2EF"

    PLOT_TEMPLATE = "plotly_dark"

else:

    BG = "#F7F5F0"
    CARD = "#FFFFFF"
    CARD_BORDER = "#E1DED6"

    TEXT = "#263833"
    HEADING = "#173F3A"
    MUTED = "#6F7B76"

    INPUT_BG = "#FFFFFF"
    INPUT_BORDER = "#D4DBD7"

    RESULT_BG = "#F1F7F4"
    RESULT_BORDER = "#D3E7DF"

    RESOLUTION_BG = "#FAF8F3"
    RESOLUTION_BORDER = "#E5DFD1"

    TEAL = "#28735F"

    ONLINE_BG = "#E7F4EE"
    ONLINE_TEXT = "#28735F"

    SHADOW = "rgba(30,45,40,0.06)"

    TAB_INACTIVE_BG = "#FFFFFF"
    TAB_INACTIVE_TEXT = "#263833"

    PLOT_TEMPLATE = "plotly_white"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    f"""
    <style>

    header[data-testid="stHeader"] {{
        display: none !important;
    }}

    [data-testid="stToolbar"] {{
        display: none !important;
    }}

    [data-testid="stDecoration"] {{
        display: none !important;
    }}

    .stApp {{
        background-color: {BG} !important;
        color: {TEXT} !important;
    }}

    .block-container {{
        max-width: 1200px;
        padding-top: 25px !important;
        padding-bottom: 50px !important;
    }}

    .brand {{
        font-size: 42px;
        font-weight: 800;
        color: {HEADING} !important;
        letter-spacing: -1.5px;
        line-height: 1.1;
    }}

    .tagline {{
        color: {MUTED} !important;
        font-size: 14px;
        margin-top: 5px;
    }}

    .page-title {{
        color: {HEADING} !important;
        font-size: 30px;
        font-weight: 800;
        margin-top: 25px;
        margin-bottom: 5px;
    }}

    .page-description {{
        color: {MUTED} !important;
        font-size: 15px;
        margin-bottom: 22px;
    }}

    .online {{
        background-color: {ONLINE_BG};
        color: {ONLINE_TEXT} !important;
        padding: 8px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }}

    .card {{
        background-color: {CARD} !important;
        border: 1px solid {CARD_BORDER};
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 16px {SHADOW};
    }}

    .card-title {{
        color: {HEADING} !important;
        font-size: 17px;
        font-weight: 750;
        margin-bottom: 14px;
    }}

    .metric {{
        background-color: {CARD} !important;
        border: 1px solid {CARD_BORDER};
        border-radius: 15px;
        padding: 18px;
        min-height: 100px;
        box-shadow: 0 4px 16px {SHADOW};
    }}

    .metric-label {{
        color: {MUTED} !important;
        font-size: 12px;
        margin-bottom: 8px;
    }}

    .metric-value {{
        color: {HEADING} !important;
        font-size: 24px;
        font-weight: 800;
        word-break: break-word;
    }}

    .result-box {{
        background-color: {RESULT_BG} !important;
        border: 1px solid {RESULT_BORDER};
        border-radius: 13px;
        padding: 18px;
        line-height: 1.6;
        color: {TEXT} !important;
    }}

    .resolution-box {{
        background-color: {RESOLUTION_BG} !important;
        border: 1px solid {RESOLUTION_BORDER};
        border-radius: 13px;
        padding: 18px;
        line-height: 1.6;
        color: {TEXT} !important;
    }}

    div[data-baseweb="input"] {{
        background-color: {INPUT_BG} !important;
        border-radius: 10px !important;
    }}

    div[data-baseweb="textarea"] {{
        background-color: {INPUT_BG} !important;
        border-radius: 10px !important;
    }}

    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {{
        background-color: {INPUT_BG} !important;
        color: {TEXT} !important;
        -webkit-text-fill-color: {TEXT} !important;
    }}

    div[data-baseweb="input"] input::placeholder,
    div[data-baseweb="textarea"] textarea::placeholder {{
        color: {MUTED} !important;
        opacity: 1 !important;
    }}

    div[data-baseweb="select"] > div {{
        background-color: {CARD} !important;
        color: {TEXT} !important;
        border: 2px solid {TEAL} !important;
        border-radius: 10px !important;
        min-width: 110px !important;
    }}

    div[data-baseweb="select"] > div * {{
        color: {TEXT} !important;
        -webkit-text-fill-color: {TEXT} !important;
        font-weight: 700 !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: transparent !important;
        padding: 8px 0 18px 0;
    }}

    .stTabs button[data-baseweb="tab"] {{
        background-color: {TAB_INACTIVE_BG} !important;
        border: 1px solid {CARD_BORDER} !important;
        border-radius: 10px !important;
        padding: 11px 20px !important;
        min-height: 44px !important;
        box-shadow: none !important;
        opacity: 1 !important;
    }}

    .stTabs button[data-baseweb="tab"],
    .stTabs button[data-baseweb="tab"] *,
    .stTabs button[data-baseweb="tab"] p,
    .stTabs button[data-baseweb="tab"] div,
    .stTabs button[data-baseweb="tab"] span {{
        color: {TAB_INACTIVE_TEXT} !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: {TAB_INACTIVE_TEXT} !important;
    }}

    .stTabs button[data-baseweb="tab"]:hover,
    .stTabs button[data-baseweb="tab"]:hover * {{
        background-color: {RESULT_BG} !important;
        color: {TEAL} !important;
        border-color: {TEAL} !important;
        -webkit-text-fill-color: {TEAL} !important;
    }}

    .stTabs button[data-baseweb="tab"][aria-selected="true"],
    .stTabs button[data-baseweb="tab"][aria-selected="true"] * {{
        background-color: {RESULT_BG} !important;
        color: {TEAL} !important;
        border: 2px solid {TEAL} !important;
        font-weight: 800 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: {TEAL} !important;
    }}

    .stButton > button {{
        border-radius: 10px !important;
        font-weight: 700 !important;
        min-height: 44px !important;
    }}

    [data-testid="stAlert"] {{
        border-radius: 12px !important;
    }}

    [data-testid="stDataFrame"] {{
        border-radius: 10px !important;
        overflow: hidden;
    }}

    .footer {{
        text-align: center;
        color: {MUTED} !important;
        font-size: 12px;
        padding-top: 35px;
    }}

    .customer-note {{
        background-color: {RESULT_BG};
        border: 1px solid {RESULT_BORDER};
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 20px;
        color: {TEXT};
        font-size: 14px;
        line-height: 1.5;
    }}

    .complaint-id-box {{
        background-color: {RESULT_BG};
        border: 1px solid {RESULT_BORDER};
        border-radius: 12px;
        padding: 14px 16px;
        color: {TEXT};
        margin-bottom: 18px;
    }}

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
            timeout=20
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
            <div class="metric-label">{html.escape(str(label))}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def generate_complaint_id():

    return "CAA-" + uuid.uuid4().hex[:8].upper()


def generate_customer_id():

    return "CUS-" + uuid.uuid4().hex[:8].upper()


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
# THEME SELECTOR
# ============================================================

theme_left, theme_right = st.columns([3, 2])

with theme_right:

    st.caption("Theme")

    selected_theme = st.selectbox(
        "Theme",
        ["Bright", "Dark"],
        index=(
            0
            if st.session_state.theme == "Bright"
            else 1
        ),
        label_visibility="collapsed"
    )

new_theme = (
    "Dark"
    if selected_theme == "Dark"
    else "Bright"
)

if new_theme != st.session_state.theme:

    st.session_state.theme = new_theme

    st.rerun()


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
            (
                df["priority"]
                .astype(str)
                .str.upper()
                == "P1"
            ).sum()
        )

        high = int(
            (
                df["priority"]
                .astype(str)
                .str.upper()
                == "P2"
            ).sum()
        )

    else:

        critical = 0
        high = 0

    if not df.empty and "status" in df.columns:

        open_cases = int(
            (
                df["status"]
                .astype(str)
                .str.lower()
                == "open"
            ).sum()
        )

    else:

        open_cases = 0

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        show_metric("Total Complaints", total)

    with m2:
        show_metric("Critical Cases", critical)

    with m3:
        show_metric("High Priority", high)

    with m4:
        show_metric("Open Cases", open_cases)

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
        'Tell us what happened. ResolveIQ will understand your issue, '
        'prioritize it, route it to the right team and recommend a resolution.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="customer-note">
            <b>How it works:</b>
            Describe your problem in your own words.
            You do not need to know the complaint category, priority,
            department or technical details. ResolveIQ will determine
            these automatically.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # CUSTOMER INPUT
    # --------------------------------------------------------

    name_col, id_col = st.columns(2)

    with name_col:

        customer_name = st.text_input(
            "Customer Name",
            placeholder="Enter customer name"
        )

    with id_col:

        transaction_id = st.text_input(
            "Order / Transaction ID",
            placeholder="Optional"
        )

    complaint_text = st.text_area(
        "Customer Complaint",
        height=180,
        placeholder=(
            "Tell us what happened...\n\n"
            "Example: My payment was deducted but my order "
            "was cancelled. I have been waiting for my refund for 7 days."
        )
    )

    st.caption(
        "Please provide as much detail as possible so ResolveIQ "
        "can understand your issue accurately."
    )

    analyze_button = st.button(
        "🔍 Analyze Complaint",
        type="primary",
        width="stretch"
    )


    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    if analyze_button:

        if not customer_name.strip():

            st.warning(
                "Please enter your name."
            )

        elif not complaint_text.strip():

            st.warning(
                "Please describe your complaint before analyzing."
            )

        elif len(complaint_text.strip()) < 10:

            st.warning(
                "Please provide a little more detail about your problem."
            )

        else:

            complaint_id = generate_complaint_id()
            customer_id = generate_customer_id()

            payload = {
                "complaint_id": complaint_id,
                "customer_id": customer_id,
                "complaint_text": complaint_text.strip()
            }

            # Send optional name as an extra field only if backend accepts it.
            # The main API request remains compatible with the current backend.

            try:

                with st.spinner(
                    "ResolveIQ is understanding your complaint..."
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

                    # Keep generated IDs for display
                    analysis["generated_complaint_id"] = complaint_id
                    analysis["generated_customer_id"] = customer_id
                    analysis["customer_name"] = customer_name.strip()

                    st.session_state.last_result = analysis

                    st.success(
                        "Your complaint has been analyzed successfully."
                    )

                    st.markdown(
                        f"""
                        <div class="complaint-id-box">
                            <b>Complaint ID:</b> {html.escape(complaint_id)}
                            <br>
                            <span style="font-size:13px;">
                                Keep this ID for future reference.
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                    # ============================================
                    # SIMPLE CUSTOMER SUMMARY
                    # ============================================

                    st.markdown(
                        '<div class="card">',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        '<div class="card-title">'
                        'Your Complaint Summary'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    summary_col1, summary_col2, summary_col3 = st.columns(3)

                    with summary_col1:

                        show_metric(
                            "Issue",
                            safe_text(
                                analysis.get("category")
                            )
                        )

                    with summary_col2:

                        show_metric(
                            "Urgency",
                            safe_text(
                                analysis.get("urgency")
                            )
                        )

                    with summary_col3:

                        show_metric(
                            "Assigned Team",
                            safe_text(
                                analysis.get(
                                    "recommended_department"
                                )
                            )
                        )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )


                    # ============================================
                    # AI ANALYSIS
                    # ============================================

                    st.markdown(
                        '<div class="card">',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        '<div class="card-title">'
                        'AI Analysis'
                        '</div>',
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


                    # ============================================
                    # EXTRACTED INFORMATION
                    # ============================================

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
                            'Information Detected'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        entity_rows = []

                        for key, value in actual_entities.items():

                            entity_rows.append(
                                {
                                    "Information": key.replace(
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


                    # ============================================
                    # RESOLUTION + RESPONSE
                    # ============================================

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
                                {html.escape(resolution)}
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
                                {html.escape(ai_response)}
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

                    # Customer-friendly error handling
                    try:

                        error_data = response.json()

                        detail = str(
                            error_data.get(
                                "detail",
                                ""
                            )
                        ).lower()

                    except Exception:

                        detail = ""

                    if response.status_code == 409:

                        st.warning(
                            "We couldn't create this complaint right now. "
                            "Please try again."
                        )

                    elif response.status_code == 422:

                        st.warning(
                            "Please check your complaint details "
                            "and try again."
                        )

                    elif response.status_code >= 500:

                        st.error(
                            "ResolveIQ is temporarily unable to process "
                            "your complaint. Please try again in a moment."
                        )

                    else:

                        st.error(
                            "We couldn't analyze your complaint. "
                            "Please try again."
                        )


            except requests.Timeout:

                st.error(
                    "The analysis is taking longer than expected. "
                    "Please try again."
                )

            except requests.RequestException:

                st.error(
                    "ResolveIQ could not connect to the analysis service. "
                    "Please try again shortly."
                )

            except Exception:

                st.error(
                    "Something went wrong while analyzing your complaint. "
                    "Please try again."
                )


# ============================================================
# COMPLAINTS TAB
# ============================================================

with tab_complaints:

    st.markdown(
        '<div class="page-title">'
        'Complaint Records'
        '</div>',
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
        '<div class="page-title">'
        'Complaint Insights'
        '</div>',
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
                title="Complaints by Category",
                template=PLOT_TEMPLATE
            )

            st.plotly_chart(
                fig_category,
                width="stretch"
            )

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
                    title="Sentiment Distribution",
                    template=PLOT_TEMPLATE
                )

                st.plotly_chart(
                    fig_sentiment,
                    width="stretch"
                )

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
                    title="Priority Distribution",
                    template=PLOT_TEMPLATE
                )

                st.plotly_chart(
                    fig_priority,
                    width="stretch"
                )

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
                title="Department Routing",
                template=PLOT_TEMPLATE
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

