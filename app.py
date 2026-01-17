import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# WAJIB: fungsi HARUS ADA
# SEBELUM joblib.load
# ==============================
def to_1d_str(x):
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.ravel()
    return [str(v) for v in x]

# ==============================
# Page config
# ==============================
st.set_page_config(
    page_title="Lasso Engagement Predictor",
    layout="centered"
)

st.title("📊 Engagement Rate Prediction")
st.write("Model: Lasso Regression")

# ==============================
# Load model (SETELAH fungsi)
# ==============================
model = joblib.load("model.pkl")

# ==============================
# Input UI
# ==============================
sentiment_score = st.number_input(
    "Sentiment Score",
    value=0.0
)

toxicity_score = st.number_input(
    "Toxicity Score",
    value=0.0
)

post_hour = st.slider(
    "Post Hour",
    0, 23, 12
)

is_weekend = st.selectbox(
    "Is Weekend?",
    [0, 1]
)

day_of_week = st.selectbox(
    "Day of Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday",
     "Friday", "Saturday", "Sunday"]
)

platform = st.selectbox(
    "Platform",
    ["Facebook", "Twitter", "Reddit", "Instagram"]
)

# ==============================
# Predict
# ==============================
text_input = st.text_area("Post Text")

if st.button("Predict Engagement Rate"):
    
    platform_map = {
    "Facebook": 0,
    "Twitter": 1,
    "Reddit": 2,
    "Instagram": 3
}

    platform_num = platform_map.get(platform, 0)

    today = pd.Timestamp.now()

    df = pd.DataFrame({
        "sentiment_score": [sentiment_score],
        "toxicity_score": [toxicity_score],
        "user_past_sentiment_avg": [0],
        "user_engagement_growth": [0],
        "buzz_change_rate": [0],
        "platform": [platform_num],
        "text_all_clean": [text_input.lower()]
    })

    df["text_all_clean_len"] = df["text_all_clean"].str.len()
    df["text_all_clean_n_words"] =       df["text_all_clean"].str.split().apply(len)

    today = pd.Timestamp.now()
    df["timestamp"] = int(today.timestamp())
    df["date"] = int(today.strftime("%Y%m%d"))
    df["year"] = today.year
    df["month_num"] = today.month
    df["month"] = today.month
    df["hour"] = post_hour
    df["day_of_week_num"] = today.dayofweek
    df["is_weekend"] = 1 if today.dayofweek >= 5 else 0

    df["engagement_rate_log"] = 0
    df["is_high_engagement"] = 0


    # ===== TEXT FEATURES =====
    df["text_all_clean_len"] = df["text_all_clean"].str.len()
    df["text_all_clean_n_words"] =              df["text_all_clean"].str.split().apply(len)

    # ===== DATE FEATURES =====
    today = pd.Timestamp.now()

    df["timestamp"] = int(today.timestamp())
    df["date"] = int(today.strftime("%Y%m%d"))


    df["year"] = today.year
    df["month_num"] = today.month
    df["month"] = today.month
    df["hour"] = post_hour
    df["day_of_week_num"] = today.dayofweek
    df["day_of_week"] = today.dayofweek
    df["is_weekend"] = 1 if today.dayofweek >= 5 else 0
    df["day_type"] = "Weekend" if today.dayofweek >= 5 else "Weekday"


    # ===== OTHERS =====
    df["language_bin"] = "EN"
    df["language_group"] = "EN"
    df["country"] = "Unknown"
    df["continent"] = "Unknown"
    df["consumer_industry"] = "General"

    df["engagement_rate_log"] = 0
    df["is_high_engagement"] = 0

    # ===== REORDER COLUMNS =====
    
    prediction = model.predict(df)

    st.success(f"Predicted Engagement Rate: {prediction[0]:.4f}")
