import streamlit as st
import numpy as np
import json
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------
# Load model and resources
# -----------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("imdb_lstm_model.h5")
    return model

@st.cache_resource
def load_word_index_and_config():
    with open("imdb_word_index.json", "r") as f:
        word_index = json.load(f)
    with open("config.pkl", "rb") as f:
        config = pickle.load(f)
    return word_index, config

model = load_model()
word_index, config = load_word_index_and_config()

VOCAB_SIZE = config["vocab_size"]
MAXLEN = config["maxlen"]

# Reserved indices as in Keras IMDB dataset / class notes:
# 0 = <PAD>, 1 = <START>, 2 = <UNK>, 3 = first word, ...
INDEX_OFFSET = 3  # actual words start at index 3

# -----------------------
# Helper: encode review text
# -----------------------

def encode_review(text, word_index, maxlen=MAXLEN):
    # Very simple tokenizer: lowercase + split by space
    words = text.lower().split()
    sequence = []
    for w in words:
        if w in word_index:
            idx = word_index[w] + INDEX_OFFSET
            if idx < VOCAB_SIZE:
                sequence.append(idx)
            else:
                sequence.append(2)  # <UNK>
        else:
            sequence.append(2)      # <UNK>

    # Add <START> token at beginning
    sequence = [1] + sequence
    padded = pad_sequences([sequence], maxlen=maxlen)
    return padded

def predict_sentiment(text):
    seq = encode_review(text, word_index, MAXLEN)
    proba = model.predict(seq, verbose=0)[0][0]
    label = "Positive 😀" if proba >= 0.5 else "Negative 😞"
    return label, float(proba)

# -----------------------
# Streamlit UI
# -----------------------

st.title("IMDB Movie Review Classifier by Juraev")  # <- change to your last name exactly as requested

st.write("This app uses an LSTM RNN model trained on the IMDB dataset to classify movie reviews as positive or negative.")

st.subheader("Sample Reviews (5 examples)")
sample_reviews = [
    "This movie was absolutely fantastic, I loved every minute of it.",
    "The plot was boring and predictable, I almost fell asleep.",
    "Amazing cinematography, but the story was very weak.",
    "I would not recommend this film to anyone, it was terrible.",
    "One of the best movies I have ever seen in my life!"
]

for i, review in enumerate(sample_reviews, start=1):
    label, proba = predict_sentiment(review)
    st.markdown(f"**Review {i}:** {review}")
    st.markdown(f"**Prediction:** {label} (score = {proba:.3f})")
    st.markdown("---")

st.subheader("Try your own review")
user_text = st.text_area("Type a movie review in English:")

if st.button("Classify Review"):
    if user_text.strip():
        label, proba = predict_sentiment(user_text.strip())
        st.write("**Result:**", label)
        st.write(f"**Score:** {proba:.3f}")
    else:
        st.warning("Please type a review first.")
