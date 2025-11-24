import streamlit as st
import numpy as np
import json
import tflite_runtime.interpreter as tflite

# -------------------------
# Load Model + Word Index
# -------------------------

@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def load_word_index():
    with open("imdb_word_index.json") as f:
        return json.load(f)

interpreter = load_model()
word_index = load_word_index()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]


# -------------------------
# Helper: Text → Model Input
# -------------------------

MAXLEN = 200

def preprocess_text(text):
    words = text.lower().split()
    seq = [word_index.get(w, 2) for w in words]  # unknown token=2

    if len(seq) > MAXLEN:
        seq = seq[:MAXLEN]
    else:
        seq += [0] * (MAXLEN - len(seq))

    return np.array([seq], dtype=np.int32)


# -------------------------
# Prediction Function
# -------------------------

def predict(text):
    x = preprocess_text(text)

    interpreter.set_tensor(input_index, x)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)[0][0]

    label = "Positive 😃" if pred > 0.5 else "Negative 😡"
    return label, float(pred)


# -------------------------
# Streamlit UI
# -------------------------

st.title("IMDB Movie Review Classifier by Juraev")

st.write("This app predicts whether a movie review expresses a **positive** or **negative** sentiment.")

# Sample reviews required by assignment
sample_reviews = [
    "This movie was amazing! I really enjoyed every scene.",
    "Worst movie ever. I wasted 2 hours of my life.",
    "The acting was okay but the storyline was weak and boring.",
    "Absolutely loved it! The visuals and plot were perfect.",
    "Not bad, but could have been much better. Mixed feelings."
]

st.subheader("📌 Example Reviews and Predictions")

for review in sample_reviews:
    label, prob = predict(review)
    st.write(f"**Review:** {review}")
    st.write(f"**Prediction:** {label} ({prob:.4f})")
    st.divider()

# Custom Single Input
st.subheader("🔍 Try Your Own Review")

user_input = st.text_area("Enter movie review here:")

if st.button("Classify"):
    if user_input.strip():
        label, prob = predict(user_input)
        st.success(f"Prediction: **{label}**  ({prob:.4f})")
    else:
        st.warning("Please enter a review before clicking classify.")
