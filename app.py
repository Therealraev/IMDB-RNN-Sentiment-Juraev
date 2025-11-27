import streamlit as st
import tensorflow as tf
import numpy as np
import json
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
# -------------------------------
# Streamlit App Header
# -------------------------------
st.set_page_config(page_title="IMDB Sentiment Classifier")
st.title("🎬 IMDB Movie Review Sentiment Classifier by Juraev")
st.write("Enter a movie review below and the model will classify it as **Positive** or **Negative**.")

# -------------------------------
# Load Model
# -------------------------------

@st.cache_resource
def load_model_fn():
    return load_model(
        "imdb_streamlit_model.keras",
        compile=False
    )

model = load_model_fn()

# -------------------------------
# Load Tokenizer
# -------------------------------
@st.cache_resource
def load_tokenizer():
    from tensorflow.keras.preprocessing.text import Tokenizer
    with open("imdb_word_index.json") as f:
        word_index = json.load(f)

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.word_index = word_index
    return tokenizer


tokenizer = load_tokenizer()

MAX_LEN = 200  # same as training


# -------------------------------
# Prediction Function
# -------------------------------
def predict_review(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded)[0][0]

    sentiment = "Positive 😃" if pred > 0.5 else "Negative 😡"
    confidence = round(float(pred if pred > 0.5 else 1 - pred) * 100, 2)

    return sentiment, confidence, pred


# -------------------------------
# User Input
# -------------------------------
user_input = st.text_area("✍️ Write a movie review to analyze:", height=180)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        with st.spinner("Analyzing review..."):
            sentiment, confidence, raw_score = predict_review(user_input)

        st.success(f"**Prediction:** {sentiment}")
        st.write(f"**Confidence:** {confidence}%")
        st.write(f"Raw model probability: `{raw_score:.4f}`")

    else:
        st.warning("⚠️ Please enter a review before clicking the button.")


# -------------------------------
# Footer
# -------------------------------
st.write("---")
st.caption("Model: IMDB LSTM Sentiment Classifier | Built by Juraev Feruz")
