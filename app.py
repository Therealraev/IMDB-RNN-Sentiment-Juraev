import streamlit as st
import tensorflow as tf
import numpy as np
import json
import pickle

# -----------------------------
# Load Model (Only once)
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("imdb_lstm_model.h5")
    return model

model = load_model()

# -----------------------------
# Load Word Index + Config
# -----------------------------
with open("imdb_word_index.json", "r") as f:
    word_index = json.load(f)

with open("config.pkl", "rb") as f:
    config = pickle.load(f)

MAXLEN = config["maxlen"]

# -----------------------------
# Helper function: Encode text
# -----------------------------
def encode_text(text):
    words = text.lower().split()
    encoded = [word_index.get(word, 2) for word in words]  # 2 = unknown token
    padded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=MAXLEN)
    return padded

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎬 IMDB Movie Review Classifier by **Juraev**")
st.write("This app predicts whether a movie review is **Positive** or **Negative** based on an LSTM model trained on the IMDB dataset.")

user_input = st.text_area("✍️ Enter a movie review below:")

if st.button("🔍 Predict Sentiment"):
    if user_input.strip():
        encoded = encode_text(user_input)
        prediction = model.predict(encoded)[0][0]

        sentiment = "👍 **Positive**" if prediction > 0.5 else "👎 **Negative**"
        
        st.subheader("📌 Prediction Result:")
        st.write(sentiment)
        st.write(f"Confidence Score: **{prediction:.4f}**")
    else:
        st.warning("Please enter a review first!")

# -----------------------------
# Show example predictions
# -----------------------------
st.subheader("🎯 Example Reviews:")

example_reviews = [
    "The movie was fantastic, I loved every moment!",
    "It was boring and too long.",
    "Absolutely brilliant acting and storyline!",
    "Worst film I have ever watched.",
    "Pretty good, but could have been shorter."
]

for review in example_reviews:
    encoded = encode_text(review)
    prediction = model.predict(encoded)[0][0]
    sentiment = "🙂 Positive" if prediction > 0.5 else "🙁 Negative"
    st.write(f"📌 \"{review}\" → **{sentiment} ({prediction:.2f})**")
