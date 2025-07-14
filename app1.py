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
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# --- SymSpell for Spelling Correction ---
from symspellpy import SymSpell, Verbosity
# --- SciSpacy for NER ---
import spacy
import subprocess
import socket
from dotenv import load_dotenv
from fuzzywuzzy import fuzz, process

# Load environment variables
load_dotenv()

# Detect local IP dynamically using socket
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

# Environment variables
api_key = os.getenv("TOGETHER_AI_API_KEY")
credentials_path = os.getenv("GOOGLE_SHEET_CREDENTIALS")
sheet_id = os.getenv("GOOGLE_SHEET_ID")

# --- Initialize SymSpell ---
@st.cache_resource
def initialize_symspell():
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    
    # Load standard dictionary
    dictionary_path = "frequency_dictionary_en_82_765.txt"
    if os.path.exists(dictionary_path):
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    else:
        st.warning("Standard dictionary not found. Spelling correction may be limited.")
    
    # Load medical terms
    medical_dict_path = "medical_terms.txt"
    if os.path.exists(medical_dict_path):
        medical_terms_added = 0
        with open(medical_dict_path, "r") as f:
            for line in f:
                for term in line.strip().lower().split(','):
                    term = term.strip()
                if term:
                    # Give medical terms higher frequency weight
                    sym_spell.create_dictionary_entry(term, 500000)
                    medical_terms_added += 1
        #st.success(f"✅ Loaded {medical_terms_added} medical terms into SymSpell!")
    else:
        st.warning("Medical dictionary not found. Using standard dictionary only.")
    
    return sym_spell

# Initialize SymSpell
sym_spell = initialize_symspell()

# --- Correct spelling ---
def correct_spelling(text):
    text = text.lower()
    terms = re.split(r'[,\s]+', text.strip())
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
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_ner_bc5cdr_md")
        return nlp
    except OSError:
        st.error("SciSpacy model 'en_ner_bc5cdr_md' not found. Please install it using: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz")
        return None

nlp = load_spacy_model()

# --- Load DistilBERT Disease Prediction Model ---
@st.cache_resource
def load_disease_model():
    model_path = "./ml_model/saved_model"
    
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        return None, None, None
    
    try:
        model = DistilBertForSequenceClassification.from_pretrained(
            model_path, 
            torch_dtype=torch.float32
        )
        model = model.to(torch.device("cpu"))
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        
        # Load label mapping
        label_mapping_path = f"{model_path}/label_mapping.json"
        if os.path.exists(label_mapping_path):
            with open(label_mapping_path) as f:
                label_mapping = json.load(f)
            id2label = {int(v): k for k, v in label_mapping.items()}
        else:
            st.error("Label mapping file not found")
            return None, None, None
            
        return model, tokenizer, id2label
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, tokenizer, id2label = load_disease_model()

# --- Load sentiment analysis model ---
@st.cache_resource
def load_sentiment_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    try:
        sentiment_tokenizer = AutoTokenizer.from_pretrained(model_name)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)
        return sentiment_pipeline
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}")
        return None

sentiment_pipeline = load_sentiment_model()

# --- Known symptoms list---
known_symptoms = [
    "fever", "headache", "cough", "sore throat", "runny nose", "fatigue",
    "body aches", "nausea", "vomiting", "diarrhea", "chest pain", "shortness of breath",
    "dizziness", "rash", "muscle pain", "joint pain", "abdominal pain", "back pain"
]

# --- Extract symptoms ---
def correct_and_extract_symptoms(text):
    if not nlp:
        return correct_spelling(text), []
    
    corrected_text = correct_spelling(text)
    doc = nlp(corrected_text)
    symptoms = [ent.text for ent in doc.ents if ent.label_ == "DISEASE"]
    return corrected_text, symptoms

# --- Fuzzy match symptoms ---
def fuzzy_match_symptoms(extracted):
    matched = set()
    for symptom in extracted:
        try:
            match, score = process.extractOne(symptom, known_symptoms, scorer=fuzz.WRatio)
            if score > 80:
                matched.add(match)
        except:
            continue
    return list(matched)

# --- Prediction Function ---
def predict_disease(text):
    if not model or not tokenizer or not id2label:
        return "Model not available", 0.0
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        
        # Get confidence score
        probabilities = torch.softmax(logits, dim=-1)
        confidence = probabilities[0][prediction].item()
        
        predicted_label = id2label[prediction]
        return predicted_label, confidence
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return "Prediction failed", 0.0

# --- Generate explanation using Together.ai ---
def generate_explanation_together_ai(api_key, user_role, symptoms_list, predicted_disease):
    if not api_key:
        return "API key not available. Please set TOGETHER_AI_API_KEY in your environment variables."
    
    symptoms_text = ", ".join(symptoms_list) if symptoms_list else "general symptoms"
    disease_name = predicted_disease.title()

    # Dynamic prompt based on user_role
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
        role_instruction = "Explain clearly."

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
            return f"❌ Failed to generate explanation! Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ Error connecting to API: {e}"

# --- Google Sheets setup ---
def get_google_sheet():
    if not credentials_path or not sheet_id:
        return None
    
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(sheet_id).sheet1
        return sheet
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        return None

# --- Save feedback ---
def save_feedback(pid, role, age, gender, symptoms, diagnosis, explanation, clarity, trust, ux_score, comment, sentiment):
    now = datetime.now()
    date_str = now.strftime("%d.%m.%Y")
    time_str = now.strftime("%H:%M:%S")

    data_row = [
        date_str, time_str, pid, role, age, gender, ", ".join(symptoms),
        diagnosis, explanation, clarity, trust, ux_score, comment, sentiment
    ]

    # Save to Google Sheets (optional)
    try:
        sheet = get_google_sheet()
        if sheet:
            sheet.append_row(data_row)
    except Exception as e:
        st.warning(f"Could not save to Google Sheets: {e}")

    # Save to local CSV (backup)
    feedback_path = os.path.join(os.getcwd(), 'feedback.csv')
    try:
        with open(feedback_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow([
                    "Date", "Time", "Participant ID", "User Role", "Age", "Gender", "Symptoms",
                    "Diagnosis", "Explanation", "Clarity Score", "Trust Score", "UX Score", "Comment", "Sentiment"
                ])
            writer.writerow(data_row)
    except Exception as e:
        st.error(f"Error saving to CSV: {e}")

# --- Check if already submitted ---
def has_already_submitted(participant_id):
    feedback_path = os.path.join(os.getcwd(), 'feedback.csv')
    if not os.path.exists(feedback_path):
        return False
    
    try:
        with open(feedback_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get("Participant ID") == participant_id:
                    return True
    except Exception as e:
        st.error(f"Error checking submissions: {e}")
    
    return False

# --- Sentiment analysis ---
def analyze_sentiment(comment):
    if not sentiment_pipeline or not comment.strip():
        return "Neutral"
    
    try:
        result = sentiment_pipeline(comment)[0]['label']
        if result == "LABEL_0":
            return "Negative"
        elif result == "LABEL_1":
            return "Neutral"
        elif result == "LABEL_2":
            return "Positive"
        else:
            return "Neutral"
    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")
        return "Neutral"

# --- Developer Dashboard ---
def show_developer_tools():
    with st.expander("🛠 Developer Tools", expanded=False):
        password = st.text_input("Enter developer password", type="password")
        if st.button("Open Dashboard"):
            if password == "1234":
                dashboard_script = "feedback_dashboard_streamlit.py"
                dashboard_port = "8506"
                
                try:
                    subprocess.Popen(["streamlit", "run", dashboard_script, "--server.port", dashboard_port])
                    dev_ip = get_local_ip()
                    dashboard_url = f"http://{dev_ip}:{dashboard_port}"
                    st.success(f"✅ Dashboard started at: [Click to open]({dashboard_url})")
                except Exception as e:
                    st.error(f"Error starting dashboard: {e}")
            else:
                st.error("❌ Incorrect password. Access denied.")

# --- Main Streamlit App ---
def main():
    st.title("NeuroAid: Symptom Checker")
    
    # Initialize session state
    if "participant_id" not in st.session_state:
        st.session_state.participant_id = ""
    if "id_applied" not in st.session_state:
        st.session_state.id_applied = False
    if "submitted" not in st.session_state:
        st.session_state.submitted = False
    if "invalid_pid" not in st.session_state:
        st.session_state.invalid_pid = False

    Show developer tools
    show_developer_tools()

    # UI for Participant ID
    if not st.session_state.id_applied:
        st.subheader("Enter your Participant ID")
        pid_input = st.text_input("Participant ID (format: p001, p002, etc.)")

        if st.button("Apply"):
            if pid_input and re.fullmatch(r"p\d{3}", pid_input.strip()):
                st.session_state.participant_id = pid_input.strip()
                st.session_state.id_applied = True
                st.session_state.invalid_pid = False
                st.rerun()
            else:
                st.session_state.invalid_pid = True

        if st.session_state.invalid_pid:
            st.error("❌ Invalid ID format. Please use format: p001, p002, etc.")

    # Check for duplicate submissions
    elif has_already_submitted(st.session_state.participant_id):
        st.warning("✅ Thank you! You have already submitted your response.")

    # Main Form
    else:
        st.subheader("Select your role")
        user_role = st.selectbox("Who are you?", ["Student", "Doctor", "Elderly"])

        st.subheader("Tell us about yourself")
        age = st.number_input("Enter your age", min_value=10, max_value=100, step=1)
        gender = st.selectbox("Gender (optional)", ["Prefer not to say", "Male", "Female", "Other"])

        st.subheader("Describe your symptoms")
        user_input = st.text_area("Type your symptoms here", placeholder="e.g., I have a sore throat and fever.")

        if st.button("Submit"):
            if not user_input.strip():
                st.error("Please enter your symptoms.")
                return

            with st.spinner("Analyzing your symptoms..."):
                # Process symptoms
                corrected_text, extracted_symptoms = correct_and_extract_symptoms(user_input)
                
                # Predict disease
                predicted_diagnosis, confidence = predict_disease(corrected_text)
                
                # Generate explanation
                explanation = generate_explanation_together_ai(
                    api_key, user_role, extracted_symptoms, predicted_diagnosis
                )

                # Store in session state
                st.session_state.corrected_text = corrected_text
                st.session_state.extracted_symptoms = extracted_symptoms
                st.session_state.predicted_diagnosis = predicted_diagnosis
                st.session_state.confidence = confidence
                st.session_state.explanation = explanation
                st.session_state.submitted = True

        if st.session_state.submitted:
            st.write("📝 **Corrected Symptoms Input:**", st.session_state.corrected_text)
            st.write("🩺 **Extracted Symptoms:**", st.session_state.extracted_symptoms)
            st.success(f"Predicted Condition: {st.session_state.predicted_diagnosis}")
            if hasattr(st.session_state, 'confidence'):
                st.write(f"Confidence: {st.session_state.confidence:.2%}")
            st.markdown("**Explanation:**")
            st.write(st.session_state.explanation)

            st.subheader("Your Feedback")
            clarity = st.slider("How clear was the explanation?", 1, 5, 3)
            trust = st.slider("How much do you trust the result?", 1, 5, 3)
            ux_score = st.slider("How easy was it to use this system?", 1, 5, 3)
            comment = st.text_area(
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
                    time.sleep(3)
                    
                    # Reset session state
                    st.session_state.participant_id = ""
                    st.session_state.id_applied = False
                    st.session_state.submitted = False
                    st.rerun()
                else:
                    st.warning("⚠️ You must agree to the research consent checkbox before submitting.")

if __name__ == "__main__":
    main()
