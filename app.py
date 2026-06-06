import streamlit as st
import pickle
import numpy as np
import re

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📩",
    layout="centered"
)

# =========================
# LOAD MODEL FILES
# =========================

model = pickle.load(open("model/logistic_regression_combined_features_model.pkl", "rb"))
tfidf = pickle.load(open("model/tfidf (1).pkl", "rb"))
w2v_model = pickle.load(open("model/w2v_model.pkl", "rb"))

# =========================
# TEXT CLEANING
# =========================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# =========================
# WORD2VEC VECTOR
# =========================

def sentence_vector(sentence):

    words = sentence.split()

    vectors = []

    for word in words:

        if word in w2v_model.wv:
            vectors.append(w2v_model.wv[word])

    if len(vectors) == 0:
        return np.zeros(w2v_model.vector_size)

    return np.mean(vectors, axis=0)

# =========================
# HEADER
# =========================

st.title("📩 SMS Spam Detection System")

st.markdown(
"""
Detect whether a message is **Spam** or **Ham (Normal Message)** using
Machine Learning, TF-IDF, Word2Vec and Logistic Regression.
"""
)

st.divider()

# =========================
# USER INPUT
# =========================

message = st.text_area(
    "Enter SMS Message",
    height=150,
    placeholder="Example: Congratulations! You won a free iPhone..."
)

# =========================
# PREDICTION
# =========================

if st.button("🔍 Predict Message Type"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:

        cleaned = clean_text(message)

        # TF-IDF
        tfidf_features = tfidf.transform(
            [cleaned]
        ).toarray()

        # Word2Vec
        w2v_features = sentence_vector(
            cleaned
        ).reshape(1, -1)

        # Combined Features
        combined_features = np.concatenate(
            (tfidf_features, w2v_features),
            axis=1
        )

        # Prediction
        prediction = model.predict(
            combined_features
        )[0]

        # Probability
        probability = model.predict_proba(
            combined_features
        )

        spam_prob = probability[0][1] * 100
        ham_prob = probability[0][0] * 100

        st.divider()

        if prediction == 1:

            st.error("🚨 SPAM MESSAGE DETECTED")

            st.metric(
                label="Spam Confidence",
                value=f"{spam_prob:.2f}%"
            )

        else:

            st.success("✅ HAM (NORMAL MESSAGE)")

            st.metric(
                label="Ham Confidence",
                value=f"{ham_prob:.2f}%"
            )

        st.write("### Prediction Details")

        st.progress(int(max(spam_prob, ham_prob)))

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"Ham Probability: {ham_prob:.2f}%")

        with col2:
            st.warning(f"Spam Probability: {spam_prob:.2f}%")

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.header("📊 Model Information")

    st.write("Algorithm:")
    st.success("Logistic Regression")

    st.write("Feature Engineering:")
    st.success("TF-IDF + Word2Vec")

    st.write("Project Type:")
    st.success("SMS Spam Detection")

    st.divider()

    st.write("### Developer")

    st.info(
        """
        👨‍💻 Developed by Mohith
        
        Engineering Student
        
        Machine Learning Enthusiast
        """
    )

# =========================
# FOOTER
# =========================

st.divider()

st.caption(
    "🚀 Developed by Mohith | Spam Detection using Machine Learning"
)