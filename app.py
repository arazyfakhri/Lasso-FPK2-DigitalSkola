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
if st.button("Predict Engagement Rate"):
    input_df = pd.DataFrame({
        'sentiment_score': [sentiment_score],
        'toxicity_score': [toxicity_score],
        'user_past_sentiment_avg': [0],
        'user_engagement_growth': [0],
        'buzz_change_rate': [0],
        'post_year': [2024],
        'post_month': [1],
        'post_day': [1],
        'post_hour': [post_hour],
        'is_weekend': [is_weekend],
        'day_of_week': [day_of_week],
        'platform': [platform],
        'topic_category': ['Support'],
        'sentiment_label': ['Neutral'],
        'emotion_type': ['Neutral'],
        'brand_name': ['Unknown'],
        'product_name': ['Unknown'],
        'campaign_name': ['Unknown'],
        'campaign_phase': ['Awareness'],
        'time_of_day': ['day'],
        'language_bin': ['EN'],
        'country': ['US'],
        'consumer_industry': ['General'],
        'continent': ['North America'],
        'text_all_clean': ['neutral post']
    })

    prediction = model.predict(input_df)

    st.write("Prediction raw output:", prediction)
    st.success(f"Predicted Engagement Rate: {prediction[0]:.4f}")
