import streamlit as st
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime
from nltk.corpus import stopwords
from model_script import run_keyword_prediction

st.set_page_config(
    page_title="Sentiment Trend Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

STOPWORDS = set(stopwords.words('english'))

def clean_tweet(tweet):
    tweet = re.sub(r"http\S+|@\S+|#\S+", "", tweet)
    tweet = re.sub(r"[^A-Za-z0-9\s]", "", tweet.lower())
    tweet = " ".join([word for word in tweet.split() if word not in STOPWORDS and len(word) > 2])
    return tweet.strip()

# Guard: if user navigated directly
if 'keyword' not in st.session_state:
    st.error("Please return to the home page and run a prediction first.")
    st.stop()

# Load data + model
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

df, model, vectorizer = load_data_and_models()

# Extract session inputs
keyword = st.session_state['keyword']
sentiment = st.session_state['sentiment']
start_date = pd.to_datetime(st.session_state['start_date'])
end_date = pd.to_datetime(st.session_state['end_date'])

# Title
st.set_page_config(page_title="Prediction Results")
st.title("📊 Prediction Results")

# Run prediction
result = run_keyword_prediction(
    keyword=keyword,
    df=df,
    sentiment_target=sentiment,
    start_date=start_date,
    end_date=end_date,
    model=model,
    vectorizer=vectorizer
)

if isinstance(result, dict):
    confidence_pct = round(result["sentiment_ratio"] * 100, 2)

    # Confidence message
    if result["will_trend"]:
        st.success(f"✅ Topic likely to trend with {confidence_pct}% confidence.")
    else:
        st.warning(f"⚠️ Topic unlikely to trend. Confidence: {confidence_pct}%.")

    # Breakdown
    st.markdown("### 📊 Prediction Breakdown")
    st.markdown(f"- **Trend Score**: `{result['trend_score']:.2f}`")
    st.markdown(f"- **Sentiment Match Ratio**: `{result['sentiment_ratio']:.2%}`")
    st.markdown(f"- **Tweet Volume**: `{result['tweet_volume']}`")

else:
    st.error(result)  # handles string error like "❌ Not enough tweets..."

# Filter tweets for analysis
filtered = df[df['text'].str.lower().str.contains(keyword.lower(), na=False)]
filtered = filtered[
    (filtered['timestamp'] >= start_date) &
    (filtered['timestamp'] <= end_date)
]

# Plot: Sentiment Distribution
st.subheader("📊 Sentiment Distribution")
sentiment_counts = filtered['sentiment'].map({0: 'Negative', 1: 'Positive'}).value_counts()

# Bar Chart
fig_bar, ax_bar = plt.subplots()
ax_bar.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red'])
ax_bar.set_ylabel("Tweet Count")
ax_bar.set_title("Distribution of Sentiment (Bar Chart)")
st.pyplot(fig_bar)

# Pie Chart
fig_pie, ax_pie = plt.subplots()
ax_pie.pie(
    sentiment_counts.values,
    labels=sentiment_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=['green', 'red']
)
ax_pie.axis('equal')
ax_pie.set_title("Distribution of Sentiment (Pie Chart)")
st.pyplot(fig_pie)

# WordCloud
st.subheader("☁️ Trending Keywords (WordCloud)")
all_words = " ".join(filtered['cleaned'].dropna().tolist())
if all_words:
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis("off")
    st.pyplot(fig_wc)
else:
    st.info("No tweets available to generate WordCloud.")

# --- Start Over Button ---
st.markdown("---")
col_start, col_export = st.columns([1, 1])

with col_start:
    if st.button("🔁 Start Over"):
        st.switch_page("app.py")

# --- Export CSV Button ---
with col_export:
    if not filtered.empty:
        export_df = filtered[['timestamp', 'text', 'sentiment']].copy()
        export_df['sentiment'] = export_df['sentiment'].map({0: 'Negative', 1: 'Positive'})
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export CSV",
            data=csv,
            file_name=f"{keyword}_tweets.csv",
            mime='text/csv'
        )
    else:
        st.info("No data available to export.")
