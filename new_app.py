import streamlit as st
import os
import csv
import time
import json
import re
import requests
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from textblob import TextBlob
# --- SymSpell for Spelling Correction ---
from symspellpy import SymSpell, Verbosity
# --- SciSpacy for NER ---
import spacy
import streamlit as st
import subprocess
import os
import sys
import socket
import random
from dotenv import load_dotenv
load_dotenv()

# Detect local IP dynamically using socket
def get_local_ip():
    try:
        # Use socket to connect to an external address and get the IP used for the connection
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # doesn't actually send data
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

api_key = os.getenv("TOGETHER_AI_API_KEY")
credentials_path = os.getenv("GOOGLE_SHEET_CREDENTIALS")
sheet_id = os.getenv("GOOGLE_SHEET_ID")

# --- Initialize SymSpell ---
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

# Load standard dictionary
dictionary_path = "frequency_dictionary_en_82_765.txt"
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Load medical terms
medical_dict_path = "medical_terms.txt"
if os.path.exists(medical_dict_path):
    medical_terms_added = 0
    with open(medical_dict_path, "r") as f:
        for line in f:
            term = line.strip()
            if term:
                sym_spell.create_dictionary_entry(term, 100000)  # High frequency
                medical_terms_added += 1
    # print(f"✅ Loaded {medical_terms_added} medical terms into SymSpell!")
else:
    raise FileNotFoundError(f"Custom medical dictionary not found at {medical_dict_path}")

# --- Correct spelling ---
def correct_spelling(text):
    text = text.lower()
    terms = re.split(r'[,\s]+', text.strip())  # Split by comma or whitespace
    corrected_terms = []

    for term in terms:
        if not term:
            continue
        suggestions = sym_spell.lookup(term, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrected_terms.append(suggestions[0].term)
        else:
            corrected_terms.append(term)

    return " ".join(corrected_terms)

# --- Initialize SciSpacy NER Model ---
nlp = spacy.load("en_ner_bc5cdr_md")  # Clinical NER model

# --- Extract symptoms ---
def correct_and_extract_symptoms(text):
    corrected_text = correct_spelling(text)
    doc = nlp(corrected_text)
    symptoms = [ent.text for ent in doc.ents if ent.label_ == "DISEASE"]
    return corrected_text, symptoms

# --- Auto-generate participant ID ---
def generate_participant_id():
    """Generate a unique participant ID in format p001, p002, etc."""
    feedback_path = os.path.join(os.getcwd(), 'feedback.csv')
    existing_ids = set()
    
    # Read existing participant IDs from CSV
    if os.path.exists(feedback_path):
        with open(feedback_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                pid = row.get("Participant ID", "").strip()
                if pid:
                    existing_ids.add(pid)
    
    # Find the next available ID
    counter = 1
    while True:
        new_id = f"a{counter:03d}"  # Format: a001, a002, etc.
        if new_id not in existing_ids:
            return new_id
        counter += 1

# --- Disease Prediction using Together.ai ---
def predict_disease_together_ai(api_key, symptoms_text):
    """
    Use Together.ai API to predict disease based on symptoms
    """
    prompt = f"""
    You are a medical AI assistant. Based on the following symptoms, predict the most likely medical condition or disease.
    
    Symptoms: {symptoms_text}
    
    Instructions:
    - Provide only the disease/condition name
    - Be specific but concise
    - If multiple conditions are possible, choose the most likely one
    - Use proper medical terminology
    - Respond with just the condition name, no additional explanation
    
    Predicted condition:
    """
    
    api_url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,  # Lower temperature for more consistent predictions
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['choices'][0]['message']['content'].strip()
            # Clean up the response to get just the condition name
            prediction = prediction.replace("Predicted condition:", "").strip()
            return prediction
        else:
            return f"Error: Unable to get prediction (Status: {response.status_code})"
    except Exception as e:
        return f"Error: {str(e)}"

def generate_explanation_together_ai(api_key, user_role, symptoms_list, predicted_disease):
    symptoms_text = ", ".join(symptoms_list)
    
    disease_name = predicted_disease.title()
    symptoms_text = ", ".join([symptom.lower() for symptom in symptoms_list])

    # 🧠 Dynamic Prompt based on user_role
    if user_role == "Student":
        role_instruction = (
            "Explain in 4–5 sentences, simple and clear, suitable for a student. "
            "Focus on explaining the predicted disease, its symptoms, and treatments without questioning the disease prediction."
        )
    elif user_role == "Doctor":
        role_instruction = (
            "Explain in detail using clinical language, suggest possible diagnostic tests and treatments, "
            "and focus on the predicted disease without questioning it."
        )
    elif user_role == "Elderly":
        role_instruction = (
            "Explain very simply in 2–3 sentences and provide 3–5 bullet points with easy lifestyle tips. "
            "Keep a comforting tone and focus on explaining the predicted disease without questioning it."
        )
    else:
        role_instruction = "Explain clearly."  # default fallback

    prompt = (
        f"You are a medical assistant. The user is a {user_role}. "
        f"Based on the predicted disease: {disease_name}, and the following symptoms: {symptoms_text}, "
        f"{role_instruction}"
    )
    
    api_url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"❌ Failed to generate explanation! Error: {response.status_code}"
    except Exception as e:
        return f"❌ Failed to generate explanation! Error: {str(e)}"

# --- File path setup ---
feedback_path = os.path.join(os.getcwd(), 'feedback.csv')

# --- Google Sheets setup ---
def get_google_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id).sheet1  # ✅ Secure usage
    return sheet

# --- Save feedback to Google Sheets and CSV ---
def save_feedback(pid, role, age, gender, symptoms, diagnosis, explanation, clarity, trust, ux_score, comment, sentiment):
    now = datetime.now()
    date_str = now.strftime("%d.%m.%Y")
    time_str = now.strftime("%H:%M:%S")

    data_row = [
        date_str, time_str, pid, role, age, gender, ", ".join(symptoms),
        diagnosis, explanation, clarity, trust, ux_score, comment, sentiment
    ]

    # # --- Save to Google Sheets ---
    try:
        sheet = get_google_sheet()
        sheet.append_row(data_row)
    except Exception as e:
         st.error(f"❌ Failed to save to Google Sheets: {e}")

    # --- Save to local CSV (backup) ---
    with open(feedback_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow([
                "Date", "Time", "Participant ID", "User Role", "Age", "Gender", "Symptoms",
                "Diagnosis", "Explanation", "Clarity Score", "Trust Score", "UX Score", "Comment", "Sentiment"
            ])
        writer.writerow(data_row)

def has_already_submitted(participant_id):
    feedback_path = os.path.join(os.getcwd(), 'feedback.csv')
    if not os.path.exists(feedback_path):
        return False
    with open(feedback_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("Participant ID") == participant_id:
                return True
    return False

def analyze_sentiment(comment):
    """Simple sentiment analysis using TextBlob"""
    if not comment.strip():
        return "Neutral"
    
    try:
        blob = TextBlob(comment)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return "Positive"
        elif polarity < -0.1:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        return "Neutral"

#------------------------------------------------------------------------------
# --- Developer Dashboard Button ---
with st.expander("🛠 Developer Tools", expanded=False):
    password = st.text_input("Enter developer password", type="password")
    if st.button("Open Dashboard"):
        if password == "1234":
            dashboard_script = "feedback_dashboard_streamlit.py"
            dashboard_port = "8506"  # run on a different port

            subprocess.Popen(["streamlit", "run", dashboard_script, "--server.port", dashboard_port])

            # ✅ Dynamically detect the IP of this laptop
            dev_ip = get_local_ip()
            dashboard_url = f"http://{dev_ip}:{dashboard_port}"
            st.success(f"✅ Dashboard started at: [Click to open]({dashboard_url})")
        else:
            st.error("❌ Incorrect password. Access denied.")

# --- Streamlit UI Starts Here ---
st.title("NeuroAid: Symptom Checker")

# --- Initialize Session State ---
if "participant_id" not in st.session_state:
    st.session_state.participant_id = generate_participant_id()
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# --- Display Participant ID ---
st.info(f"🆔 Your Participant ID: **{st.session_state.participant_id}**")

# --- Check for duplicate submissions ---
if has_already_submitted(st.session_state.participant_id):
    st.warning("✅ Thank you! You have already submitted your response.")
    if st.button("Start New Session"):
        st.session_state.participant_id = generate_participant_id()
        st.session_state.submitted = False
        st.rerun()

# --- Main Form ---
else:
    st.subheader("Select your role")
    user_role = st.selectbox("Who are you?", ["Student", "Doctor", "Elderly"])

    st.subheader("Tell us about yourself")
    age = st.number_input("Enter your age", min_value=10, max_value=100, step=1)
    gender = st.selectbox("Gender (optional)", ["Prefer not to say", "Male", "Female", "Other"])

    st.subheader("Describe your symptoms")
    user_input = st.text_area("Type your symptoms here", placeholder="e.g., I have a sore throat and fever.")

    if not st.session_state.submitted and st.button("Submit"):
        if not user_input.strip():
            st.error("❌ Please describe your symptoms before submitting.")
        else:
            with st.spinner("Processing your symptoms..."):
                # Process symptoms
                corrected_text, extracted_symptoms = correct_and_extract_symptoms(user_input)
                st.session_state.corrected_text = corrected_text
                st.session_state.extracted_symptoms = extracted_symptoms
                
                # Get prediction from Together.ai
                predicted_diagnosis = predict_disease_together_ai(api_key, corrected_text)
                st.session_state.predicted_diagnosis = predicted_diagnosis

            with st.spinner("Generating explanation... Please wait."):
                # Generate explanation
                mistral_explanation = generate_explanation_together_ai(
                    api_key,
                    user_role,
                    st.session_state.extracted_symptoms,
                    st.session_state.predicted_diagnosis
                )
                st.session_state.explanation = mistral_explanation
                st.session_state.submitted = True

    if st.session_state.submitted:
        st.write("📝 **Corrected Symptoms Input:**", st.session_state.corrected_text)
        st.write("🩺 **Extracted Symptoms:**", st.session_state.extracted_symptoms)
        st.success(f"Predicted Condition: {st.session_state.predicted_diagnosis}")
        st.markdown("**Explanation:**")
        st.write(st.session_state.explanation)

        st.subheader("Your Feedback")
        clarity = st.slider("How clear was the explanation?", 1, 5, 3)
        trust = st.slider("How much do you trust the result?", 1, 5, 3)
        ux_score = st.slider("How easy was it to use this system?", 1, 5, 3)
        comment = st.text_input(
            "Any additional thoughts?",
            placeholder="Tell us what you liked, what could be improved, or any issues you faced."
        )

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
                time.sleep(2)
                st.session_state.participant_id = generate_participant_id()
                st.session_state.submitted = False
                st.rerun()
            else:
                st.warning("⚠️ You must agree to the research consent checkbox before submitting.")