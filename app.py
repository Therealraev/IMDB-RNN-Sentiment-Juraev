import streamlit as st
import tensorflow as tf
import numpy as np
import json

# -------------------------------------
# Streamlit Page Setup
# -------------------------------------
st.set_page_config(
    page_title="IMDB Sentiment Classifier",
    layout="centered"
)

st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.write("This tool predicts whether a movie review is **Positive** or **Negative** using an LSTM model trained on the IMDB dataset.")

# -------------------------------------
# Load the Model (Cached)
# -------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("imdb_lstm_model.h5", compile=False)

model = load_model()

# -------------------------------------
# Load Word Index
# -------------------------------------
with open("word_index.json", "r") as f:
    word_index = json.load(f)

MAX_LEN = 200


# -------------------------------------
# Text Preprocessing Function
# -------------------------------------
def preprocess_text(text):
    text = text.lower().split()
    encoded = [word_index.get(word, 2) for word in text]  # 2 = unknown token
    encoded = encoded[:MAX_LEN]
    padding = [0] * (MAX_LEN - len(encoded))
    final = encoded + padding
    return np.array([final])


# -------------------------------------
# UI Input
# -------------------------------------
review_text = st.text_area("✍ Write a movie review:")

if st.button("Predict Sentiment"):
    if review_text.strip() == "":
        st.warning("⚠ Please enter a review before predicting.")
    else:
        processed = preprocess_text(review_text)
        prediction = model.predict(processed)[0][0]

        sentiment = "👍 Positive" if prediction > 0.5 else "👎 Negative"

        st.subheader(f"Result: {sentiment}")
        st.write(f"Confidence Score: **{prediction:.4f}**")

# -------------------------------------
# Footer
# -------------------------------------
st.write("---")
st.caption("Model: LSTM trained on IMDB Reviews dataset | Built by Feruz Juraev (NLP Course Project)")
