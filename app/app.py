import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import os
import sys

# ---------------- PATH SETUP ----------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models/sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models/vectorizer.pkl"))

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Sentiment Dashboard", layout="wide")

st.title("🚀 AI-Powered Sentiment Intelligence Dashboard")
st.caption("Analyze social media sentiment with ML, visualize insights, and extract business signals.")

# ---------------- SIDEBAR ----------------
mode = st.sidebar.radio("Choose Mode", ["Single Text", "Upload CSV"])

# ================= SINGLE TEXT =================
if mode == "Single Text":
    st.subheader("🧪 Single Text Prediction")

    text = st.text_area("Enter text to analyze")

    if st.button("Analyze"):
        text_clean = clean_text(text)
        vec = vectorizer.transform([text_clean])

        pred = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]
        confidence = probs.max()

        st.success(f"Sentiment: {pred}")
        st.info(f"Confidence: {confidence:.2f}")

        # Confidence breakdown
        prob_df = pd.DataFrame({
            "Sentiment": model.classes_,
            "Probability": probs
        })

        fig = px.bar(prob_df, x="Sentiment", y="Probability",
                     title="Prediction Confidence Breakdown")
        st.plotly_chart(fig, use_container_width=True)

# ================= CSV MODE =================
else:
    st.subheader("📂 Batch Sentiment Analysis")

    file = st.file_uploader("Upload CSV file")

    # ---------------- LOAD DATA ----------------
    if file is None:
        st.warning("No file uploaded. Using sample dataset...")
        df = pd.read_csv(os.path.join(BASE_DIR, "data/processed/sample_cleaned.csv"))
    else:
        df = pd.read_csv(file)

    # ---------------- HANDLE COLUMN NAMES ----------------
    if 'text' not in df.columns:
        if 'Cleaned_Text' in df.columns:
            df.rename(columns={'Cleaned_Text': 'text'}, inplace=True)
        else:
            st.error("CSV must contain 'text' or 'Cleaned_Text' column")
            st.stop()

    # ---------------- CLEAN ONLY IF NEEDED ----------------
    if 'Cleaned_Text' not in df.columns:
        df['text'] = df['text'].apply(clean_text)

    # ---------------- PREDICTIONS ----------------
    vec = vectorizer.transform(df['text'])
    df['Prediction'] = model.predict(vec)
    df['Confidence'] = model.predict_proba(vec).max(axis=1)

    # Fake timestamps for trend
    df['timestamp'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='h')

    # ================= KPI =================
    st.subheader("📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(df))
    col2.metric("Positive", (df['Prediction'] == 'positive').sum())
    col3.metric("Negative", (df['Prediction'] == 'negative').sum())
    col4.metric("Avg Confidence", f"{df['Confidence'].mean():.2f}")

    # ================= TABS =================
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Explore", "🧠 Insights"])

    # -------- TAB 1 --------
    with tab1:
        fig1 = px.pie(df, names='Prediction', title="Sentiment Distribution")
        st.plotly_chart(fig1, use_container_width=True)

        bar = df['Prediction'].value_counts().reset_index()
        bar.columns = ['Sentiment', 'Count']
        fig2 = px.bar(bar, x='Sentiment', y='Count', title="Sentiment Count")
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.histogram(df, x="Confidence", title="Confidence Distribution")
        st.plotly_chart(fig3, use_container_width=True)

    # -------- TAB 2 --------
    with tab2:
        sentiment_filter = st.selectbox("Filter by Sentiment",
                                        ["All", "positive", "negative", "neutral"])

        search = st.text_input("Search text")

        filtered = df.copy()

        if sentiment_filter != "All":
            filtered = filtered[filtered['Prediction'] == sentiment_filter]

        if search:
            filtered = filtered[filtered['text'].str.contains(search, case=False)]

        st.dataframe(filtered)

        # Download
        csv = filtered.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download Results", csv, "results.csv", "text/csv")

    # -------- TAB 3 --------
    with tab3:
        trend = df.groupby([df['timestamp'].dt.hour, 'Prediction']).size().reset_index(name='count')
        trend.columns = ['Hour', 'Sentiment', 'Count']

        fig_trend = px.line(trend, x="Hour", y="Count", color="Sentiment",
                            title="Sentiment Trend")
        st.plotly_chart(fig_trend, use_container_width=True)

        # Word cloud
        text_data = " ".join(df['text'])
        wc = WordCloud(width=800, height=400).generate(text_data)

        st.subheader("☁️ Word Cloud")
        plt.imshow(wc)
        plt.axis("off")
        st.pyplot(plt)

        # Keywords
        words = text_data.split()
        common_words = Counter(words).most_common(10)

        st.subheader("🔑 Top Keywords")
        st.write(pd.DataFrame(common_words, columns=["Word", "Count"]))

        # Business insight
        st.subheader("💼 Business Insight")

        pos = (df['Prediction'] == 'positive').sum()
        neg = (df['Prediction'] == 'negative').sum()

        if pos > neg:
            st.success("Overall sentiment is positive ✅")
        else:
            st.error("Negative sentiment is high ⚠️")