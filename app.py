from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# REVIEWRADAR AI — STARTUP-STYLE CUSTOMER INTELLIGENCE PLATFORM
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

CLEANED_FILE = DATA_DIR / "cleaned_reviews.csv"
DEPARTMENT_FILE = DATA_DIR / "department_summary.csv"
TOPIC_FILE = DATA_DIR / "topic_summary.csv"
SATISFACTION_FILE = DATA_DIR / "topic_satisfaction.csv"
PRIORITY_FILE = DATA_DIR / "issue_priority_report.csv"
MODEL_FILE = MODEL_DIR / "best_sentiment_model_day5.pkl"

st.set_page_config(
    page_title="ReviewRadar AI | Customer Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- THEME ---------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root {
      --ink:#f2f4ff; --muted:#9ba8c5; --line:#283452; --panel:#151f36;
      --canvas:#0a1020; --blue:#8b9cff; --violet:#a78bfa; --green:#4fd1aa; --red:#fb7185;
    }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:radial-gradient(circle at 80% 0%,rgba(100,83,190,.18),transparent 28%),#0a1020; color:var(--ink); }
    [data-testid="stHeader"] { background:rgba(10,16,32,.92); }
    [data-testid="stToolbar"] { right:1rem; }
    [data-testid="stSidebar"] { background:#0d162b; border-right:1px solid #202c48; }
    [data-testid="stSidebar"] * { color:#e7ebff !important; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"], [data-testid="stSidebar"] .stMarkdown p { color:#aab7d0 !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] > label { display:none; }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] { gap:7px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label {
      padding:12px 13px; border-radius:13px; margin:0; min-height:44px;
      border:1px solid transparent; background:transparent; transition:all .18s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:hover {
      background:#172642 !important; border-color:#2c3c60;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
      background:linear-gradient(100deg,rgba(91,111,220,.24),rgba(139,92,246,.17)) !important;
      border:1px solid #46578a; box-shadow:inset 3px 0 0 #8b8cff;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label > div:first-child { display:none !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label p {
      color:#dce4fb !important; font-size:13px; font-weight:600; margin:0;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p { color:#ffffff !important; }
    [data-testid="stSidebar"] hr { border-color:#283452; }
    .brand-lockup { display:flex; align-items:center; gap:11px; margin:8px 0 28px; }
    .brand-icon { width:43px;height:43px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:21px;background:linear-gradient(135deg,#6478ff,#9b69e8);box-shadow:0 8px 26px rgba(100,110,255,.3); }
    .brand-name { font-family:'Manrope',sans-serif;font-size:18px;font-weight:800;color:#fff; }
    .brand-sub { font-size:9px;color:#aab7d0;letter-spacing:1.5px;text-transform:uppercase; }
    .side-label { color:#8292b5 !important;font-size:10px;font-weight:800;letter-spacing:1.7px;text-transform:uppercase;margin:16px 0 7px; }
    .main .block-container { max-width:1500px;padding-top:2.1rem;padding-bottom:3.5rem; }
    [data-testid="stSidebar"] .block-container { padding-top:1.5rem; }
    h1,h2,h3 { font-family:'Manrope',sans-serif !important;color:#f2f4ff !important; }
    h1 { font-size:2rem !important;font-weight:800 !important;letter-spacing:-.8px; }
    h2 { font-size:1.25rem !important;font-weight:750 !important; }
    h3 { font-size:1rem !important;font-weight:700 !important; }
    p, label, .stMarkdown, [data-testid="stCaptionContainer"] { color:#c2cbe0; }
    .eyebrow { color:#9aa8ff;font-size:10px;font-weight:800;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px; }
    .page-subtitle { color:#9ba8c5;font-size:14px;margin-top:-8px;margin-bottom:24px; }
    .hero { background:linear-gradient(115deg,#182746 0%,#283b70 55%,#57438e 100%);padding:32px 34px;border:1px solid rgba(169,158,255,.2);border-radius:22px;color:white;margin-bottom:25px;box-shadow:0 18px 50px rgba(0,0,0,.24); }
    .hero h2 { color:#fff !important;font-size:26px !important;margin:0 0 10px; }
    .hero p { color:#d8def4;margin:0;font-size:14px;max-width:720px; }
    .hero-pill { display:inline-block;border:1px solid rgba(255,255,255,.24);color:#e9eaff;padding:7px 11px;border-radius:30px;font-size:10px;letter-spacing:.6px;margin-bottom:15px;background:rgba(255,255,255,.08); }
    .metric-card { background:linear-gradient(145deg,#18243d,#121c32);border:1px solid #2a3858;border-radius:18px;padding:20px;min-height:128px;box-shadow:0 12px 28px rgba(0,0,0,.16);transition:transform .18s ease,border-color .18s ease; }
    .metric-card:hover { transform:translateY(-3px);border-color:#5969a5; }
    .metric-label { color:#9ba8c5;font-size:12px;font-weight:600; }
    .metric-value { font-family:'Manrope',sans-serif;font-size:28px;font-weight:800;color:#f5f6ff;margin-top:8px;letter-spacing:-.8px; }
    .metric-note { color:#8291b0;font-size:11px;margin-top:3px; }
    .section-head { display:flex;justify-content:space-between;align-items:center;margin:25px 0 13px; }
    .panel { background:#151f36;border:1px solid #283653;border-radius:18px;padding:20px 22px;box-shadow:0 10px 28px rgba(0,0,0,.14);margin-bottom:15px; }
    .panel-title { font-family:'Manrope',sans-serif;font-weight:750;font-size:15px;color:#eef1ff;margin-bottom:4px; }
    .panel-caption { color:#93a1bf;font-size:12px;margin-bottom:12px; }
    [data-testid="stVerticalBlockBorderWrapper"] { background:linear-gradient(145deg,rgba(21,31,54,.92),rgba(15,24,43,.96));border:1px solid #283653;border-radius:18px;box-shadow:0 12px 30px rgba(0,0,0,.12); }
    [data-testid="stPlotlyChart"] { background:transparent;border-radius:14px; }
    [data-testid="stPlotlyChart"] .modebar { background:rgba(16,26,46,.8) !important;border-radius:8px; }
    [data-testid="stPlotlyChart"] .modebar-btn path { fill:#aebce0 !important; }
    [data-testid="stPlotlyChart"] .modebar-btn:hover path { fill:#ffffff !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] { margin-top:5px; }
    .insight { border-left:4px solid #8995ff;background:rgba(104,119,220,.13);padding:14px 16px;border-radius:0 12px 12px 0;color:#d5dcf5;font-size:13px;margin:12px 0 17px; }
    .tag { display:inline-block;border-radius:20px;padding:5px 9px;font-size:10px;font-weight:700;background:rgba(93,211,174,.12);color:#67dfba; }
    .stButton > button { border-radius:12px;border:1px solid #354363;background:#1a2742;color:#e9edff;font-weight:700;min-height:42px;transition:all .15s ease; }
    .stButton > button:hover { border-color:#8794ff;color:#fff;background:#243454; }
    .stButton > button[kind="primary"] { background:linear-gradient(100deg,#596ff0,#8b5fe0);border:0;color:white;box-shadow:0 7px 20px rgba(91,103,230,.25); }
    .stTextArea textarea,.stTextInput input { border-radius:13px !important;border:1px solid #354363 !important;background:#101a2e !important;color:#f1f3ff !important; }
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div { background:#101a2e;color:#f1f3ff;border-color:#354363; }
    [data-testid="stMetric"] { background:#151f36;border:1px solid #283653;border-radius:15px;padding:14px 16px; }
    [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] { color:#aab7d0 !important; }
    [data-testid="stMetricValue"] { color:#f2f4ff !important; }
    [data-testid="stDataFrame"] { border:1px solid #2b3958;border-radius:12px;overflow:hidden; }
    [data-testid="stAlert"] { background:#17243c;border:1px solid #354363;color:#e9edff; }
    .footer { color:#71809f;font-size:10px;text-align:center;padding:30px 0 6px;letter-spacing:1.2px; }
    [data-testid="stSidebar"] .tag { color:#67dfba !important;background:rgba(93,211,174,.12) !important; }
    .sidebar-note { color:#aab7d0 !important;font-size:11px;line-height:1.7;margin-top:12px; }
    .sidebar-footer { color:#7f8eac !important;font-size:10px;line-height:1.8;border-top:1px solid #283653;padding-top:14px;margin-top:17px; }
    .prediction-card { border-radius:16px;padding:20px;border:1px solid #2b3958;background:#151f36; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------- DATA ----------------------------

@st.cache_data
def load_data():
    required = [CLEANED_FILE, DEPARTMENT_FILE, TOPIC_FILE, SATISFACTION_FILE, PRIORITY_FILE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required data files:\n" + "\n".join(missing))
    return (
        pd.read_csv(CLEANED_FILE),
        pd.read_csv(DEPARTMENT_FILE),
        pd.read_csv(TOPIC_FILE),
        pd.read_csv(SATISFACTION_FILE),
        pd.read_csv(PRIORITY_FILE),
    )


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    return joblib.load(MODEL_FILE)


try:
    reviews, departments, topics, satisfaction, priorities = load_data()
except Exception as exc:
    st.error("ReviewRadar could not load its data files. Check the project folders and filenames.")
    st.exception(exc)
    st.stop()

for col in ["Rating", "Recommended_IND", "Positive_Feedback_Count"]:
    if col in reviews.columns:
        reviews[col] = pd.to_numeric(reviews[col], errors="coerce")

rating_data = reviews.dropna(subset=["Rating"])
total_reviews = len(reviews)
average_rating = rating_data["Rating"].mean()
recommendation_rate = reviews["Recommended_IND"].eq(1).mean() * 100
rating_counts = (
    rating_data["Rating"].value_counts().sort_index()
    .rename_axis("Rating").reset_index(name="Review Count")
)
sentiment_distribution = pd.DataFrame({
    "Sentiment": ["Negative", "Neutral", "Positive"],
    "Reviews": [
        int(rating_data["Rating"].isin([1, 2]).sum()),
        int(rating_data["Rating"].eq(3).sum()),
        int(rating_data["Rating"].isin([4, 5]).sum()),
    ],
})

# --------------------------- SIDEBAR -------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
          <div class="brand-icon">◉</div>
          <div><div class="brand-name">ReviewRadar AI</div>
          <div class="brand-sub">Customer intelligence</div></div>
        </div>
        <div class="side-label">Workspace</div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        [
            "Business Overview",
            "Customer Feedback",
            "Issue Prioritization",
            "Sentiment Intelligence",
            "AI Prediction Studio",
        ],
        label_visibility="collapsed",
        format_func=lambda item: {
            "Business Overview": "▦   Overview",
            "Customer Feedback": "◫   Feedback Explorer",
            "Issue Prioritization": "⚑   Issue Priorities",
            "Sentiment Intelligence": "◉   Sentiment Analysis",
            "AI Prediction Studio": "✧   AI Prediction Studio",
        }[item],
    )
    st.divider()
    st.markdown('<div class="side-label">Project status</div>', unsafe_allow_html=True)
    st.markdown('<span class="tag">●  MODEL READY</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-note">NLP&nbsp; · &nbsp;Machine Learning&nbsp; · &nbsp;Business Intelligence</div>'
        '<div class="sidebar-footer">REVIEWRADAR AI · PORTFOLIO EDITION<br>Developed by Hamna Fazil</div>',
        unsafe_allow_html=True,
    )

# --------------------------- HELPERS -------------------------

def page_heading(eyebrow, title, subtitle):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def metric_card(label, value, note):
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div></div>""",
        unsafe_allow_html=True,
    )


def panel_title(title, caption=None):
    st.markdown(
        f'<div class="panel-title">{title}</div>'
        + (f'<div class="panel-caption">{caption}</div>' if caption else ""),
        unsafe_allow_html=True,
    )


def chart_style(fig, height=340):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=45, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#aab7d0", size=11),
        title=dict(font=dict(size=14, color="#eef1ff")),
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color="#aab7d0")
    fig.update_yaxes(gridcolor="#293653", zeroline=False, color="#aab7d0")
    return fig


# --------------------------- OVERVIEW ------------------------

if page == "Business Overview":
    st.markdown(
        """
        <div class="hero">
          <span class="hero-pill">✦ AI-POWERED CUSTOMER INTELLIGENCE</span>
          <h2>Turn customer feedback into your next business advantage.</h2>
          <p>Understand what customers love, uncover recurring problems, and prioritize
          improvements with data-driven insights.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="eyebrow">YOUR BUSINESS AT A GLANCE</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    with cols[0]:
        metric_card("Total reviews", f"{total_reviews:,}", "Records in the analysis dataset")
    with cols[1]:
        metric_card("Average rating", f"{average_rating:.2f} / 5", "Mean customer star rating")
    with cols[2]:
        metric_card("Recommendation rate", f"{recommendation_rate:.1f}%", "Reviews marked as recommended")
    with cols[3]:
        metric_card("Feedback themes", f"{len(topics):,}", "NMF-discovered topics")

    st.markdown('<div class="section-head"><div class="panel-title">Customer pulse</div><span class="tag">DATA OVERVIEW</span></div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            panel_title("Rating distribution", "How customers rated their experience")
            fig = px.bar(rating_counts, x="Rating", y="Review Count", color="Rating",
                         color_continuous_scale=["#dce5ff", "#4169e1"])
            st.plotly_chart(chart_style(fig), use_container_width=True)
    with right:
        with st.container(border=True):
            panel_title("Recommendation split", "Share of customers who recommend the product")
            rec_data = pd.DataFrame({
                "Recommendation": ["Recommended", "Not recommended"],
                "Count": [int(reviews["Recommended_IND"].eq(1).sum()),
                          int(reviews["Recommended_IND"].eq(0).sum())],
            })
            fig = px.pie(rec_data, names="Recommendation", values="Count", hole=.68,
                         color="Recommendation",
                         color_discrete_map={"Recommended": "#4169e1", "Not recommended": "#e58a95"})
            fig.update_traces(textposition="outside", textinfo="percent")
            st.plotly_chart(chart_style(fig), use_container_width=True)
    with st.container(border=True):
        panel_title("Customer sentiment snapshot", "Sentiment categories inferred from star ratings")
        fig = px.bar(sentiment_distribution, x="Sentiment", y="Reviews", color="Sentiment",
                     color_discrete_map={"Negative": "#df6876", "Neutral": "#e7ad55", "Positive": "#35a889"})
        st.plotly_chart(chart_style(fig, 300), use_container_width=True)
    st.markdown('<div class="insight"><b>How to read this dashboard:</b> sentiment categories are derived from star ratings (1–2 negative, 3 neutral, 4–5 positive). They are not manually annotated labels.</div>', unsafe_allow_html=True)

# --------------------------- FEEDBACK -----------------------

elif page == "Customer Feedback":
    page_heading("VOICE OF THE CUSTOMER", "Feedback Explorer", "Explore the themes customers mention and compare satisfaction across topics.")
    if "Topic_Label" in satisfaction.columns:
        display_topics = satisfaction.copy()
        for col in ["Average_Rating", "Non_Recommended_Percentage"]:
            if col in display_topics:
                display_topics[col] = pd.to_numeric(display_topics[col], errors="coerce")
        with st.container(border=True):
            panel_title("Topic satisfaction", "Summary of the eight themes discovered through NMF topic modelling")
            st.dataframe(display_topics, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                panel_title("Average rating by topic")
                fig = px.bar(display_topics, x="Topic_Label", y="Average_Rating",
                             color="Average_Rating", color_continuous_scale="Blues")
                fig.update_layout(xaxis_tickangle=-35)
                st.plotly_chart(chart_style(fig, 390), use_container_width=True)
        with c2:
            with st.container(border=True):
                panel_title("Non-recommendation by topic")
                fig = px.bar(display_topics, x="Topic_Label", y="Non_Recommended_Percentage",
                             color="Non_Recommended_Percentage", color_continuous_scale="Reds")
                fig.update_layout(xaxis_tickangle=-35)
                st.plotly_chart(chart_style(fig, 390), use_container_width=True)
    else:
        st.warning("The topic satisfaction file does not contain the expected Topic_Label column.")

    st.subheader("Browse customer reviews")
    review_col = "Review_Text" if "Review_Text" in reviews.columns else None
    if review_col:
        search = st.text_input("Search review text", placeholder="Try: size, quality, comfortable...")
        filtered = reviews.dropna(subset=[review_col]).copy()
        if search.strip():
            filtered = filtered[filtered[review_col].astype(str).str.contains(search.strip(), case=False, na=False)]
        visible_cols = [c for c in ["Rating", "Department_Name", "Class_Name", "Review_Text"] if c in filtered.columns]
        st.caption(f"{len(filtered):,} matching reviews")
        st.dataframe(filtered[visible_cols].head(250), use_container_width=True, hide_index=True)
    else:
        st.info("Review_Text column is not available in the cleaned dataset.")

# --------------------------- PRIORITIES ----------------------

elif page == "Issue Prioritization":
    page_heading("BUSINESS ACTION CENTER", "Issue Prioritization", "See which feedback themes deserve closer investigation.")
    priorities = priorities.copy()
    if "Priority_Score" in priorities.columns:
        priorities["Priority_Score"] = pd.to_numeric(priorities["Priority_Score"], errors="coerce")
        priorities = priorities.sort_values("Priority_Score", ascending=False)
    with st.container(border=True):
        panel_title("Priority ranking", "Ranked using the project's defined volume, dissatisfaction and non-recommendation heuristic")
        st.dataframe(priorities, use_container_width=True, hide_index=True)
    if not priorities.empty and "Priority_Score" in priorities.columns:
        fig = px.bar(priorities, x="Priority_Score", y="Topic_Label", orientation="h",
                     color="Priority_Score", color_continuous_scale="Oranges",
                     text="Priority_Score")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(chart_style(fig, 410), use_container_width=True)
        highest = priorities.iloc[0]
        st.markdown(
            f'<div class="insight"><b>Highest-ranked theme:</b> {highest["Topic_Label"]} '
            f'— priority score {highest["Priority_Score"]:.2f}. Use this as a signal for investigation, '
            f'not as a validated business-risk measurement.</div>',
            unsafe_allow_html=True,
        )
    st.caption("Priority scores are based on a project-defined heuristic and are not validated business risk measurements.")

# --------------------------- SENTIMENT -----------------------

elif page == "Sentiment Intelligence":
    page_heading("CUSTOMER MOOD", "Sentiment Intelligence", "Understand the balance of positive, neutral and negative feedback.")
    counts = {row["Sentiment"]: int(row["Reviews"]) for _, row in sentiment_distribution.iterrows()}
    cols = st.columns(3)
    for col, label, color in zip(cols, ["Positive", "Neutral", "Negative"], ["#159b76", "#c58a31", "#dc5967"]):
        with col:
            metric_card(f"{label} reviews", f'{counts[label]:,}', "Rating-derived category")
    with st.container(border=True):
        panel_title("Sentiment distribution", "Based on star-rating categories, not manually labelled sentiment")
        fig = px.pie(sentiment_distribution, names="Sentiment", values="Reviews", hole=.62,
                     color="Sentiment",
                     color_discrete_map={"Positive": "#35a889", "Neutral": "#e7ad55", "Negative": "#df6876"})
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(chart_style(fig, 370), use_container_width=True)
    st.subheader("Model performance")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Day 4 baseline accuracy", "76.97%")
        st.caption("Logistic Regression baseline · macro F1: 0.60")
    with c2:
        st.metric("Selected Day 5 model", "Complement Naive Bayes")
        st.caption("Selected according to macro F1 comparison.")
    st.markdown('<div class="insight"><b>Interpretation:</b> the dataset is imbalanced, with substantially more positive reviews. Macro F1 is useful when comparing performance across all sentiment classes.</div>', unsafe_allow_html=True)

# --------------------------- PREDICTION ----------------------

elif page == "AI Prediction Studio":
    page_heading("AI LAB", "Review Sentiment Studio", "Paste a customer review and let your trained machine-learning model classify it.")
    st.markdown(
        """
        <div class="hero">
          <span class="hero-pill">✧ COMPLEMENT NAIVE BAYES · DAY 5 MODEL</span>
          <h2>What is your customer really saying?</h2>
          <p>Run a quick sentiment prediction on a review. Use the examples below to explore the model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    examples = {
        "Positive example": "The dress is beautiful, comfortable, and fits perfectly. I love the quality.",
        "Neutral example": "The product is okay. The colour is nice, but the fit is average.",
        "Negative example": "The material is poor, the dress is uncomfortable, and the sizing is completely wrong.",
    }
    st.markdown("**Try a sample review**")
    sample_cols = st.columns(3)
    for col, (label, sample) in zip(sample_cols, examples.items()):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state["review_input"] = sample

    review_text = st.text_area(
        "Customer review",
        key="review_input",
        height=180,
        placeholder="Paste or type a customer review here...",
        help="The model predicts Positive, Neutral or Negative sentiment.",
    )
    left, right = st.columns([1, 4])
    with left:
        analyze = st.button("✦  Analyze review", type="primary", use_container_width=True)
    with right:
        st.caption("Your review is processed locally by the model loaded in this Streamlit application.")

    if analyze:
        if not review_text.strip():
            st.warning("Please enter a review or choose one of the sample reviews.")
        else:
            try:
                model = load_model()
                prediction = str(model.predict([review_text])[0]).strip()
                normalized = prediction.lower()
                if "positive" in normalized:
                    st.success("Positive sentiment detected")
                    st.markdown('<div class="insight"><b>Model result:</b> The review is classified as positive.</div>', unsafe_allow_html=True)
                elif "negative" in normalized:
                    st.error("Negative sentiment detected")
                    st.markdown('<div class="insight"><b>Model result:</b> The review is classified as negative.</div>', unsafe_allow_html=True)
                else:
                    st.info("Neutral sentiment detected")
                    st.markdown('<div class="insight"><b>Model result:</b> The review is classified as neutral.</div>', unsafe_allow_html=True)
                st.caption(f"Predicted class returned by the saved model: {prediction}")
            except Exception as exc:
                st.error("The saved Day 5 model could not be loaded or used.")
                st.exception(exc)

# --------------------------- FOOTER --------------------------

st.markdown(
    '<div class="footer">REVIEWRADAR AI · CUSTOMER INTELLIGENCE PLATFORM · BUILT WITH PYTHON, NLP & MACHINE LEARNING</div>',
    unsafe_allow_html=True,
)
