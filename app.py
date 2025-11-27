import streamlit as st
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="IMDB Sentiment Classifier", layout="centered")

st.title("🎬 IMDB Movie Review Sentiment Classifier by Juraev")
st.write("Enter a movie review below and the model will classify it as **Positive** or **Negative**.")

MAX_LEN = 200
VOCAB_SIZE = 10000


# -------------------------------
# Load TFLite Model
# -------------------------------
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="imdb_sentiment.tflite")
    interpreter.allocate_tensors()
    return interpreter


interpreter = load_tflite_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# -------------------------------
# Load Word Index
# -------------------------------
@st.cache_resource
def load_word_index():
    with open("imdb_word_index.json") as f:
        index = json.load(f)

    # Match IMDB encoding rules
    index = {k:(v+3) for k, v in index.items()}
    index["<PAD>"] = 0
    index["<START>"] = 1
    index["<UNK>"] = 2
    index["<UNUSED>"] = 3
    return index


word_index = load_word_index()


# -------------------------------
# Encoding Function (IMDB style)
# -------------------------------
def encode_review(text):
    words = text.lower().split()
    encoded = [1]  # <START>

    for w in words:
        encoded.append(word_index.get(w, 2))  # 2 = <UNK>

    return encoded


# -------------------------------
# Prediction Function
# -------------------------------
def predict_review(text):
    encoded = encode_review(text)
    padded = pad_sequences([encoded], maxlen=MAX_LEN).astype("float32")

    interpreter.set_tensor(input_details[0]['index'], padded)
    interpreter.invoke()
    pred = float(interpreter.get_tensor(output_details[0]['index'])[0][0])

    sentiment = "Positive 😃" if pred > 0.5 else "Negative 😡"
    confidence = round((pred if pred > 0.5 else 1 - pred) * 100, 2)

    return sentiment, confidence, pred


# -------------------------------
# Input UI
# -------------------------------
review = st.text_area("🧠 Write a movie review to analyze:", height=160)

if st.button("Analyze Sentiment"):
    if review.strip():
        with st.spinner("Analyzing... 🧪"):
            sentiment, confidence, score = predict_review(review)

        st.success(f"**Prediction:** {sentiment}")
        st.write(f"Confidence: **{confidence}%**")
        st.write(f"Raw model probability: `{score:.4f}`")
    else:
        st.warning("⚠️ Please enter a review before clicking the button.")


# -------------------------------
# Footer
# -------------------------------
st.write("---")
st.caption("Model: IMDB LSTM Sentiment Classifier | Built by Juraev Feruz")
