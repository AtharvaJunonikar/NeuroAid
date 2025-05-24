#       cd /Users/acharnandish/hcai_app && source venv/bin/activate && streamlit run app.py


import streamlit as st
import csv
from datetime import datetime
import os
import time
from textblob import TextBlob

# --- File path setup ---
feedback_path = os.path.join(os.getcwd(), 'feedback.csv')

# --- Check for duplicate Participant ID ---
def has_already_submitted(participant_id):
    if not os.path.exists(feedback_path):
        return False
    with open(feedback_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("Participant ID") == participant_id:
                return True
    return False

# --- Sentiment Analysis ---
def analyze_sentiment(comment):
    if not comment.strip():
        return "Neutral"
    blob = TextBlob(comment)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return "Positive"
    elif polarity < -0.2:
        return "Negative"
    else:
        return "Neutral"

# --- Save feedback to CSV ---
def save_feedback(pid, role, age, gender, symptoms, diagnosis, explanation, clarity, trust, ux_score, comment, sentiment):
    now = datetime.now()
    date_str = now.strftime("%d.%m.%Y")
    time_str = now.strftime("%H:%M:%S")

    with open(feedback_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow([
                "Date", "Time", "Participant ID", "User Role", "Age", "Gender", "Symptoms",
                "Diagnosis", "Explanation", "Clarity Score", "Trust Score", "UX Score", "Comment", "Sentiment"
            ])
        writer.writerow([
            date_str, time_str, pid, role, age, gender, ", ".join(symptoms),
            diagnosis, explanation, clarity, trust, ux_score, comment, sentiment
        ])

# --- Streamlit UI Starts Here ---
st.title("NeuroAid: Symptom Checker")

# Session State
if "participant_id" not in st.session_state:
    st.session_state.participant_id = ""
if "id_applied" not in st.session_state:
    st.session_state.id_applied = False
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# Step 0: Participant ID
if not st.session_state.id_applied:
    st.subheader("Enter your Participant ID")
    pid_input = st.text_input("Participant ID", key="pid_input")
    if st.button("Apply") or (pid_input and pid_input != st.session_state.participant_id):
        st.session_state.participant_id = pid_input
        st.session_state.id_applied = True
        st.rerun()

elif has_already_submitted(st.session_state.participant_id):
    st.warning("✅ Thank you! You have already submitted your response.")
else:
    st.subheader("Select your role")
    user_role = st.selectbox("Who are you?", ["Student", "Doctor", "Elderly"])

    st.subheader("Tell us about yourself")
    age = st.number_input("Enter your age", min_value=10, max_value=100, step=1)
    gender = st.selectbox("Gender (optional)", ["Prefer not to say", "Male", "Female", "Other"])

    st.subheader("Describe your symptoms")
    user_input = st.text_area("Type your symptoms here", placeholder="e.g., I have a sore throat and fever.")

    symptom_keywords = [
        "fever", "cough", "sore throat", "headache", "fatigue",
        "vomiting", "nausea", "diarrhea", "body ache", "cold", "chills"
    ]

    def extract_symptoms(text):
        return [kw for kw in symptom_keywords if kw in text.lower()]

    if not st.session_state.submitted and st.button("Submit"):
        st.session_state.extracted_symptoms = extract_symptoms(user_input)
        st.session_state.predicted_diagnosis = "Flu"
        st.session_state.explanation = (
            f"As a **{user_role}**, based on your reported symptoms "
            f"({', '.join(st.session_state.extracted_symptoms)}), this may be **{st.session_state.predicted_diagnosis}**. "
            "Please consult a doctor if symptoms persist."
        )
        st.session_state.submitted = True

    if st.session_state.submitted:
        st.write("🩺 **Extracted Symptoms:**", st.session_state.extracted_symptoms)
        st.success(f"Predicted Condition: {st.session_state.predicted_diagnosis}")
        st.markdown("**Explanation:**")
        st.write(st.session_state.explanation)

        st.subheader("Your Feedback")
        clarity = st.slider("How clear was the explanation?", 1, 5, 3)
        trust = st.slider("How much do you trust the result?", 1, 5, 3)
        ux_score = st.slider("How easy was it to use this system?", 1, 5, 3)
        comment = st.text_input("Any additional thoughts?")
        consent = st.checkbox("I agree to let my anonymized responses be used for research purposes.")

        if st.button("Submit Feedback"):
            if consent:
                sentiment = analyze_sentiment(comment)
                save_feedback(
                    pid=st.session_state.participant_id,
                    role=user_role,
                    age=age,
                    gender=gender,
                    symptoms=st.session_state.extracted_symptoms,
                    diagnosis=st.session_state.predicted_diagnosis,
                    explanation=st.session_state.explanation,
                    clarity=clarity,
                    trust=trust,
                    ux_score=ux_score,
                    comment=comment,
                    sentiment=sentiment
                )
                st.success("✅ Thank you! Your feedback has been recorded.")
                time.sleep(3)
                st.session_state.participant_id = ""
                st.session_state.id_applied = False
                st.session_state.submitted = False
                st.rerun()
            else:
                st.warning("⚠️ You must agree to the research consent checkbox before submitting.")