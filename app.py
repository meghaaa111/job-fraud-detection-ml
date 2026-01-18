import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Job Fraud Detection",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f2027, #203a43, #2c5364);
    color: white;
}

h1 {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: 1px;
}

textarea {
    background-color: #1e1e2f !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #444 !important;
}

.stButton > button {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    border-radius: 10px;
    font-size: 1rem;
    padding: 0.6em 1.2em;
    border: none;
}

.stAlert {
    border-radius: 14px;
    font-size: 1.1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("job_fraud_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ---------------- HELPER FUNCTION ----------------
def highlight_keywords(text, keywords):
    for word in keywords:
        text = text.replace(
            word,
            f"<span style='background-color:#ff4b4b; padding:2px 6px; border-radius:6px; color:white'>{word}</span>"
        )
    return text

# ---------------- UI ----------------
st.title("Job Fraud Detection System")
st.write("Paste a job description below to assess its fraud risk.")

job_text = st.text_area("Job Description", height=200)

# ---------------- PREDICTION ----------------
if st.button("Analyze Job"):
    st.write("Button clicked")  # DEBUG LINE (you can remove later)

    if job_text.strip() == "":
        st.warning("Please enter a job description.")
    else:
        # Transform input
        text_tfidf = tfidf.transform([job_text])

        # Predict probability
        probability = model.predict_proba(text_tfidf)[0][1]
        

        # Risk levels
        if probability >= 0.30:
            label = " High Fraud Risk"
            color = "error"
        elif probability >= 0.10:
            label = " Medium Fraud Risk"
            color = "warning"
        else:
            label = "Low Fraud Risk"
            color = "success"

        # Show result
        if color == "error":
            st.error(f"{label}\n\nFraud Risk Score: {probability:.2%}")
        elif color == "warning":
            st.warning(f"{label}\n\nFraud Risk Score: {probability:.2%}")
        else:
            st.success(f"{label}\n\nFraud Risk Score: {probability:.2%}")

        # Risk meter
        st.markdown("### Fraud Risk Meter")
        st.progress(min(int(probability * 100), 100))
        st.caption("Higher percentage indicates higher fraud risk")

        # Explanation
        st.markdown("""
### How to interpret this result

- **Low Risk (<10%)** → Job description looks professional and structured  
- **Medium Risk (10–30%)** → Some suspicious patterns detected  
- **High Risk (>30%)** → Strong indicators of fraudulent behavior
""")

        # Highlight suspicious words
        suspicious_words = ["urgent", "whatsapp", "fee", "immediate", "earn", "limited"]

        highlighted_text = highlight_keywords(
            job_text.lower(),
            suspicious_words
        )

        st.markdown("### Suspicious Patterns Detected")
        st.markdown(highlighted_text, unsafe_allow_html=True)
