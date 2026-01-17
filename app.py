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

    # mapping platform -> numeric
    platform_map = {
        "Facebook": 0,
        "Twitter": 1,
        "Reddit": 2,
        "Instagram": 3
    }
    platform_num = platform_map.get(platform, 0)

    # === LOAD TEMPLATE TRAINING (INI KUNCI) ===
    df = pd.read_csv("schema_row.csv")

    # === OVERWRITE INPUT USER ===
    df["sentiment_score"] = sentiment_score
    df["toxicity_score"] = toxicity_score
    df["platform"] = platform_num
    df["hour"] = post_hour

    # (kalau di schema ada kolom lain yang mau kamu kontrol, set di sini)

    # === PREDICT ===

    prediction = model.predict(df)

    st.success(f"Predicted Engagement Rate: {prediction[0]:.4f}")
