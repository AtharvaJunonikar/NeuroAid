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
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from textblob import TextBlob
from transformers import pipeline
# --- SymSpell for Spelling Correction ---
from symspellpy import SymSpell, Verbosity
# --- SciSpacy for NER ---
import spacy
import streamlit as st
import subprocess
import os
import sys
import socket
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

# --- Load your DistilBERT Disease Prediction Model ---
model_path = "./ml_model/saved_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path, torch_dtype=torch.float32).to('cpu')
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# --- Extract symptoms ---
def correct_and_extract_symptoms(text):
    corrected_text = correct_spelling(text)
    doc = nlp(corrected_text)
    symptoms = [ent.text for ent in doc.ents if ent.label_ == "DISEASE"]
    return corrected_text, symptoms

# --- Load label mapping ---
with open(f"{model_path}/label_mapping.json") as f:
    id2label = {int(v): k for k, v in json.load(f).items()}



# --- Your Existing Prediction Function ---
def predict_disease(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    # Load label mapping
    with open(f"{model_path}/label_mapping.json") as f:
        id2label = {int(v): k for k, v in json.load(f).items()}
    predicted_label = id2label[prediction]
    return predicted_label


# --- Spell Correction + NER-based Symptom Extraction ---
def correct_and_extract_symptoms(text):
    corrected_text = correct_spelling(text)
    doc = nlp(corrected_text)
    symptoms = [ent.text for ent in doc.ents if ent.label_ == "DISEASE"]
    return corrected_text, symptoms



def fuzzy_match_symptoms(extracted):
    matched = set()
    for symptom in extracted:
        match, score, _ = process.extractOne(symptom, known_symptoms, scorer=fuzz.WRatio)
        if score > 80:  # Set threshold for match confidence
            matched.add(match)
    return list(matched)

# --- Full Pipeline ---
def full_pipeline(user_input):
    corrected_text = correct_spelling(user_input)
    extracted = extract_symptoms(corrected_text)
    matched = fuzzy_match_symptoms(extracted)
    return matched


# --- Label mapping ---
import json
with open(f"{model_path}/label_mapping.json") as f:
    id2label = {int(v): k for k, v in json.load(f).items()}

# --- Prediction Function ---
def predict_disease(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[prediction]
    return predicted_label

def generate_explanation_together_ai(api_key, user_role, symptoms_list, predicted_disease=None):
    symptoms_text = ", ".join([symptom.lower() for symptom in symptoms_list])

    if predicted_disease:
        disease_name = predicted_disease.title()
        prompt = (
            f"You are a medical assistant. The user is a {user_role}. "
            f"Based on the predicted disease: {disease_name}, and the following symptoms: {symptoms_text}, "
        )
    else:
        prompt = (
            f"You are a medical assistant. The user is a {user_role}. "
            f"The user reports the following symptoms: {symptoms_text}. "
        )

    # Role specific instructions
    if user_role == "Student":
        role_instruction = (
            "Explain in 4–5 sentences, simple and clear, suitable for a student. "
            "Focus on explaining the symptoms and possible conditions without referencing a predicted disease."
        )
    elif user_role == "Doctor":
        role_instruction = (
            "Explain in detail using clinical language, suggest possible diagnostic tests and treatments, "
            "and focus on the symptoms reported."
        )
    elif user_role == "Elderly":
        role_instruction = (
            "Explain very simply in 2–3 sentences and provide 3–5 bullet points with easy lifestyle tips. "
            "Keep a comforting tone and focus on the symptoms."
        )
    else:
        role_instruction = "Explain clearly."

    prompt += role_instruction

    api_url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        return f"❌ Failed to generate explanation! Error: {response.status_code}"



# --- File path setup ---
feedback_path = os.path.join(os.getcwd(), 'feedback.csv')

# --- Google Sheets setup ---
def get_google_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id).sheet1  # ✅ Secure usage
    return sheet

def generate_next_participant_id():
    try:
        sheet = get_google_sheet()
        all_records = sheet.get_all_records()
        used_ids = [row.get("Participant ID", "").lower() for row in all_records if row.get("Participant ID")]

        used_numbers = []
        for pid in used_ids:
            match = re.fullmatch(r"a(\d{3})", pid)
            if match:
                used_numbers.append(int(match.group(1)))

        next_number = 1
        if used_numbers:
            next_number = max(used_numbers) + 1

        next_pid = f"a{next_number:03d}"
        return next_pid
    except Exception as e:
        st.error(f"❌ Failed to generate Participant ID: {e}")
        return None


# --- Save feedback to Google Sheets and CSV ---
def save_feedback(pid, role, age, gender, symptoms, diagnosis, explanation, clarity, trust, ux_score, comment, sentiment):
    now = datetime.now()
    date_str = now.strftime("%d.%m.%Y")
    time_str = now.strftime("%H:%M:%S")

    data_row = [
        date_str, time_str, pid, role, age, gender, ", ".join(symptoms),
        diagnosis, explanation, clarity, trust, ux_score, comment, sentiment
    ]

    # --- Save to Google Sheets ---
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
    try:
        sheet = get_google_sheet()  # your existing function to get the Google Sheet
        all_records = sheet.get_all_records()
        for row in all_records:
            if row.get("Participant ID") == participant_id:
                return True
        return False
    except Exception as e:
        st.error(f"❌ Error checking Google Sheets for participant ID: {e}")
        # fallback to assume not submitted, so user can submit again
        return False







# Load Hugging Face sentiment analysis model (RoBERTa trained on tweets)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

def analyze_sentiment(comment):
    if not comment.strip():
        return "Neutral"
    result = sentiment_pipeline(comment)[0]['label']
    if result == "LABEL_0":
        return "Negative"
    elif result == "LABEL_1":
        return "Neutral"
    elif result == "LABEL_2":
        return "Positive"
    else:
        return "Neutral"
#------------------------------------------------------------------------------
# --- Developer Dashboard Button ---
# with st.expander("🛠 Developer Tools", expanded=False):
#     password = st.text_input("Enter developer password", type="password")
#     if st.button("Open Dashboard"):
#         if password == "hcai1234":
#             dashboard_script = "feedback_dashboard_streamlit.py"
#             dashboard_port = "8506"  # run on a new port

#             subprocess.Popen(["streamlit", "run", dashboard_script, "--server.port", dashboard_port])

#             # YOUR local network IP (same you used for app)
#             dev_ip = "192.168.0.25"
#             dashboard_url = f"http://{dev_ip}:{dashboard_port}"
#             st.success(f"✅ Dashboard started at: [Click to open]({dashboard_url})")
#         else:
#             st.error("❌ Incorrect password. Access denied.")
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


# --- Session State ---
if "participant_id" not in st.session_state:
    st.session_state.participant_id = ""
if "id_applied" not in st.session_state:
    st.session_state.id_applied = False
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "invalid_pid" not in st.session_state:
    st.session_state.invalid_pid = False
if "predicted_diagnosis" not in st.session_state:
    st.session_state.predicted_diagnosis = None
if "clicked_continue" not in st.session_state:
    st.session_state.clicked_continue = False



# --- UI for Participant ID ---
# --- Automatically assign Participant ID ---
if not st.session_state.id_applied:
    with st.spinner("Assigning your Participant ID..."):
        generated_id = generate_next_participant_id()
        if generated_id:
            st.session_state.participant_id = generated_id
            st.session_state.id_applied = True
            st.rerun()
        else:
            st.error("❌ Could not assign Participant ID. Please try again later.")


    if pid_input:
        if re.fullmatch(r"a\d{3}", pid_input.strip()):
            st.session_state.participant_id = pid_input.strip()
            st.session_state.id_applied = True
            st.session_state.invalid_pid = False
            st.rerun()
        else:
            st.session_state.invalid_pid = True

    if st.button("Apply"):
        with st.spinner("Checking Participant ID..."):
            if re.fullmatch(r"a\d{3}", pid_input.strip()):
                st.session_state.participant_id = pid_input.strip()
                st.session_state.id_applied = True
                st.session_state.invalid_pid = False
                st.rerun()
            else:
                st.session_state.invalid_pid = True


    if st.session_state.invalid_pid:
        st.error("❌ Invalid ID format.")

# --- Check for duplicate submissions ---
# --- Check for duplicate submissions ---
elif has_already_submitted(st.session_state.participant_id):
    st.warning("✅ Thank you! You have already submitted your response.")

else:
    # Show assigned participant ID
    st.info(f"🆔 Your assigned Participant ID: `{st.session_state.participant_id}`")

    # Show "Continue" button first
    if not st.session_state.clicked_continue:
        if st.button("Continue"):
            st.session_state.clicked_continue = True
            st.rerun()

    # After clicking Continue, show the full form
    elif st.session_state.clicked_continue:
        st.subheader("Select your role")
        user_role = st.selectbox("Who are you?", ["Student", "Doctor", "Elderly"])

        st.subheader("Tell us about yourself")
        age = st.number_input("Enter your age", min_value=10, max_value=100, step=1)
        gender = st.selectbox("Gender (optional)", ["Prefer not to say", "Male", "Female", "Other"])

        st.subheader("Describe your symptoms")
        user_input = st.text_area("Type your symptoms here", placeholder="e.g., I have a sore throat and fever.")

        # Show Submit button only after Continue
        if not st.session_state.submitted and st.button("Submit"):
            corrected_text, extracted_symptoms = correct_and_extract_symptoms(user_input)
            st.session_state.corrected_text = corrected_text
            st.session_state.extracted_symptoms = extracted_symptoms
            st.session_state.predicted_diagnosis = None

            with st.spinner("Generating explanation... Please wait."):
                mistral_explanation = generate_explanation_together_ai(
                    api_key,
                    user_role,
                    st.session_state.extracted_symptoms,
                    None
                )

            st.session_state.explanation = mistral_explanation
            st.session_state.submitted = True

        # Show feedback form after submission
        if st.session_state.submitted:
            st.write("📝 **Corrected Symptoms Input:**", st.session_state.corrected_text)
            st.write("🩺 **Extracted Symptoms:**", st.session_state.extracted_symptoms)

            st.markdown("**Explanation:**")
            st.write(st.session_state.explanation)

            st.subheader("Your Feedback")
            clarity = st.slider("How clear was the explanation?", 1, 5, 3)
            trust = st.slider("How much do you trust the result?", 1, 5, 3)
            ux_score = st.slider("How easy was it to use this system?", 1, 5, 3)
            comment = st.text_input("Any additional thoughts?", placeholder="Tell us what you liked or what could improve.")
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
                    st.session_state.clicked_continue = False
                    st.rerun()
                else:
                    st.warning("⚠️ You must agree to the research consent checkbox before submitting.")


        
