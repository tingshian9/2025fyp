#!/usr/bin/env python
# coding: utf-8

# Step 1: Import libraries
import os
import re
import pickle
import shutil
import kagglehub
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Step 2: Download Sentiment140 dataset
path = kagglehub.dataset_download("kazanova/sentiment140")
print("✅ Dataset downloaded to:", path)

# Target destination
target_folder = "./data"
os.makedirs(target_folder, exist_ok=True)

# Move dataset to local /data folder
source_csv = os.path.join(path, "training.1600000.processed.noemoticon.csv")
target_csv = os.path.join(target_folder, "training.1600000.processed.noemoticon.csv")

if not os.path.exists(target_csv):
    shutil.copy(source_csv, target_csv)
    print(f"📁 Dataset copied to: {target_csv}")
else:
    print(f"📁 Dataset already exists at: {target_csv}")

# Step 3: Load and prepare dataset
csv_path = target_csv
df = pd.read_csv(csv_path, encoding='latin-1', header=None)
df = df[[0, 2, 5]]  # sentiment, timestamp, tweet
df.columns = ['sentiment', 'timestamp', 'text']
df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})  # 0 = negative, 1 = positive

# Step 4: Parse timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Step 5: Clean tweets
STOPWORDS = set(stopwords.words('english'))

def clean_tweet(tweet):
    tweet = re.sub(r"http\S+|@\S+|#\S+", "", tweet)
    tweet = re.sub(r"[^A-Za-z0-9\s]", "", tweet.lower())
    tweet = " ".join([word for word in tweet.split() if word not in STOPWORDS and len(word) > 2])
    return tweet.strip()

df['cleaned'] = df['text'].apply(clean_tweet)

# Step 6–9: Load or train model and vectorizer
model_path = "model.pkl"
vectorizer_path = "vectorizer.pkl"

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    print("🔁 Loading existing model and vectorizer...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
else:
    print("🧠 Training new model...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned'])
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    import json

    # Step 8: Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n📊 Model Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))

    # Step 8.1: Save metrics to JSON
    metrics = {
    "Accuracy": round(accuracy * 100, 2),
    "Precision": round(precision * 100, 2),
    "Recall": round(recall * 100, 2),
    "F1-Score": round(f1 * 100, 2)
}
    metrics_path = os.path.join(target_folder, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    print(f"✅ Metrics saved to: {metrics_path}")

    # Step 9: Save
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print("✅ Model and vectorizer saved.")

# Step 10: Predict trend based on keyword and date range
def run_keyword_prediction(keyword, df, sentiment_target="Positive", start_date=None, end_date=None, model=None, vectorizer=None):
    keyword = keyword.lower()
    keyword_tweets = df[df['text'].str.lower().str.contains(keyword, na=False)]

    if start_date and end_date:
        keyword_tweets = keyword_tweets[
            (keyword_tweets['timestamp'] >= start_date) & 
            (keyword_tweets['timestamp'] <= end_date)
        ]

    if len(keyword_tweets) < 30:
        return f"❌ Not enough tweets with keyword '{keyword}' in the selected date range."

    tweets = [clean_tweet(t) for t in keyword_tweets['text']]
    X_input = vectorizer.transform(tweets)
    probas = model.predict_proba(X_input)
    positive_probs = probas[:, 1]

    if sentiment_target.lower() == "positive":
        match = positive_probs >= 0.6
    else:
        match = positive_probs <= 0.4

    sentiment_ratio = match.sum() / len(tweets)
    tweet_volume = len(tweets)
    trend_score = sentiment_ratio * np.log1p(tweet_volume)

    return {
        "will_trend": trend_score > 1.2,
        "trend_score": trend_score,
        "sentiment_ratio": sentiment_ratio,
        "tweet_volume": tweet_volume
    }