import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Load model and embedder
# -----------------------------
@st.cache_resource
def load_models():
    model = joblib.load("model.pkl")
    embedder = joblib.load("embedder.pkl")
    return model, embedder

model, embedder = load_models()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="LLM Hallucination Detector", layout="centered")

st.title("🔍 LLM Hallucination Detector")
st.write(
    "This app checks whether an AI-generated answer is **factually reliable** "
    "or likely a **hallucination**."
)

st.markdown("---")

question = st.text_area(
    "🧠 Enter Question",
    placeholder="e.g., Who invented the Python programming language?"
)

answer = st.text_area(
    "🤖 Enter AI-Generated Answer",
    placeholder="e.g., Python was invented by Guido van Rossum in 1991."
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Check Hallucination"):
    if not question.strip() or not answer.strip():
        st.warning("Please enter both a question and an answer.")
    else:
        # Prepare input text (must match training format)
        text = f"Question: {question}\nAnswer: {answer}\nIs this answer factually correct?"

        embedding = embedder.encode([text])

        # 🔥 Add dummy verification feature (0.0)
        verification_feature = np.array([[0.0]])

        # Combine to match training shape (385 features)
        final_features = np.hstack([embedding, verification_feature])

        prob_factual = model.predict_proba(final_features)[0][1]


        st.markdown("---")
        st.subheader("📊 Prediction Result")

        # Threshold logic (IMPORTANT)
        if prob_factual > 0.65:
            label = "✅ FACTUAL"
            color = "green"
        elif prob_factual < 0.35:
            label = "❌ HALLUCINATED"
            color = "red"
        else:
            label = "⚠️ UNCERTAIN"
            color = "orange"

        st.markdown(
            f"<h2 style='color:{color};'>{label}</h2>",
            unsafe_allow_html=True
        )

        st.write(f"**Factual Confidence:** `{prob_factual:.2f}`")

        st.progress(prob_factual)

        # Explanation
        st.markdown("### 🧠 Explanation")
        if label == "❌ HALLUCINATED":
            st.write(
                "The answer shows **low factual confidence**, which may indicate "
                "fabricated or unsupported information."
            )
        elif label == "⚠️ UNCERTAIN":
            st.write(
                "The model is **not confident**. This answer may require "
                "external verification."
            )
        else:
            st.write(
                "The answer aligns well with factual patterns learned during training."
            )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "⚠️ This system provides probabilistic predictions and does not guarantee correctness."
)
