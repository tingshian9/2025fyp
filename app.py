import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from datetime import datetime
from nltk.corpus import stopwords
from model_script import run_keyword_prediction
import json

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Clean tweet function
def clean_tweet(tweet):
    tweet = re.sub(r"http\S+|@\S+|#\S+", "", tweet)
    tweet = re.sub(r"[^A-Za-z0-9\s]", "", tweet.lower())
    tweet = " ".join([word for word in tweet.split() if word not in STOPWORDS and len(word) > 2])
    return tweet.strip()

# Load data and models 
@st.cache_resource
def load_data_and_models():
    df = pd.read_csv("data/training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None)
    df = df[[0, 2, 5]]
    df.columns = ['sentiment', 'timestamp', 'text']
    df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['cleaned'] = df['text'].apply(clean_tweet)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return df, model, vectorizer

st.set_page_config(
    page_title="Sentiment Trend Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load data
df, model, vectorizer = load_data_and_models()

# Determine dataset range
dataset_min = df['timestamp'].min().date()
dataset_max = df['timestamp'].max().date()

# Full calendar bounds
calendar_min = datetime(2000, 1, 1).date()
calendar_max = datetime(2035, 12, 31).date()

# Streamlit UI
st.set_page_config(page_title="Sentiment Trend Predictor", layout="centered")
st.title("📈 Sentiment Media Trend Prediction")

# Input
keyword = st.text_input("🔍 Search for keywords here")

col1, col2 = st.columns(2)

with col1:
    date_range = st.date_input(
        "📅 Date filter (start and end)",
        value=[dataset_min, dataset_max],
        min_value=calendar_min,
        max_value=calendar_max
    )
    st.caption(f"📅 Tweets available from {dataset_min.strftime('%b %Y')} to {dataset_max.strftime('%b %Y')}")

with col2:
    sentiment = st.radio("🎯 Sentiment filter", ["Positive", "Negative"], index=0)

# Predict button with custom styling
predict_button = st.button("🚀 Predict")

st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        font-size: 20px;
        padding: 15px 40px;
        border-radius: 12px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

if predict_button:
    if not keyword:
        st.warning("Please enter a keyword.")
    elif len(date_range) != 2:
        st.warning("Please select both start and end dates.")
    else:
        st.session_state['keyword'] = keyword
        st.session_state['sentiment'] = sentiment
        st.session_state['start_date'] = str(date_range[0])
        st.session_state['end_date'] = str(date_range[1])
        st.switch_page("pages/results.py")

with st.expander("📊 View Model Evaluation Metrics"):
    try:
        with open("data/metrics.json", "r") as f:
            metrics = json.load(f)
        st.write("### Evaluation Results")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy (Overall Correctness)", f"{metrics['Accuracy']}%")
            st.metric("Recall (Trustworthiness)", f"{metrics['Recall']}%")
        with col2:
            st.metric("Precision (Trustworthiness)", f"{metrics['Precision']}%")
            st.metric("F1-Score (Balanced Score for Recall and Precision)", f"{metrics['F1-Score']}%")
    except FileNotFoundError:
        st.warning("⚠️ Model metrics not available. Please re-run training to generate `metrics.json`.")
