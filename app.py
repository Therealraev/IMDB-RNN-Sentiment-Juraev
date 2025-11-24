import streamlit as st
import numpy as np
import json
import random
import tflite_runtime.interpreter as tflite
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb


# -----------------------------
# App Title
# -----------------------------
st.title("🎬 IMDB Movie Review Classifier by Juraev")


# -----------------------------
# Load IMDB word index
# -----------------------------
@st.cache_data
def load_word_index():
    with open("imdb_word_index.json", "r") as f:
        return json.load(f)

word_index = load_word_index()
index_to_word = {v: k for k, v in word_index.items()}


# -----------------------------
# Load TFLite model
# -----------------------------
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Get input/output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

MAXLEN = 200


# -----------------------------
# Preprocessing Function
# -----------------------------
def encode_review(text):
    words = text.lower().split()
    encoded = [word_index.get(word, 2) for word in words]  # 2 = "UNK"
    padded = pad_sequences([encoded], maxlen=MAXLEN)
    return padded


def predict_sentiment(text):
    encoded = encode_review(text)
    interpreter.set_tensor(input_details[0]["index"], encoded.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])[0][0]
    return prediction


# -----------------------------
# Show 5 Random Real Reviews
# -----------------------------
st.subheader("📌 Random IMDB Reviews Classification")

(x_train, y_train), (_, _) = imdb.load_data(num_words=10000)

def decode_imdb_review(encoded_review):
    return " ".join([index_to_word.get(i, "UNK") for i in encoded_review])

random_indices = random.sample(range(len(x_train)), 5)

for idx in random_indices:
    review_text = decode_imdb_review(x_train[idx])
    pred = predict_sentiment(review_text)
    label = "😊 Positive" if pred > 0.5 else "😡 Negative"
    
    st.write("---")
    st.write(f"📝 **Review:** {review_text[:300]}...")
    st.write(f"🔍 **Prediction:** `{label}` (score: {pred:.3f})")


# -----------------------------
# User Input Section
# -----------------------------
st.subheader("✍️ Try your own review")

user_input = st.text_area("Enter a movie review here:")

if st.button("Classify"):
    if len(user_input.strip()) == 0:
        st.warning("⚠ Please enter a review!")
    else:
        pred = predict_sentiment(user_input)
        label = "😊 Positive" if pred > 0.5 else "😡 Negative"
        st.success(f"Prediction: **{label}** (score: {pred:.3f})")
