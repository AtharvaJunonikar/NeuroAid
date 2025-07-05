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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
# --- SymSpell for Spelling Correction ---
from symspellpy import SymSpell, Verbosity
# --- SciSpacy for NER ---
import spacy
import subprocess
import socket
from dotenv import load_dotenv
from fuzzywuzzy import fuzz, process
import numpy as np

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
                term = line.strip()
                if term:
                    sym_spell.create_dictionary_entry(term, 100000)
                    medical_terms_added += 1
        st.success(f"✅ Loaded {medical_terms_added} medical terms into SymSpell!")
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

# --- Load New BERT Disease Prediction Model ---
@st.cache_resource
def load_disease_model():
    # Update this path to point to your new BERT model directory
    # Use raw string (r"") or forward slashes to avoid escape character issues
    model_path = r"./ml_model/New/saved_model"  # Your actual model path
    
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        return None, None, None
    
    try:
        # Load tokenizer and model using AutoTokenizer and AutoModelForSequenceClassification
        # This is more flexible and works with different BERT variants
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            torch_dtype=torch.float32,
            local_files_only=True  # Ensure we use local files only
        )
        
        # Move model to CPU (change to 'cuda' if you have GPU and want to use it)
        device = torch.device("cpu")
        model = model.to(device)
        model.eval()  # Set model to evaluation mode
        
        # Load label mapping
        label_mapping_path = os.path.join(model_path, "label_mapping.json")
        config_path = os.path.join(model_path, "config.json")
        
        # Try to load label mapping from multiple possible locations
        id2label = None
        
        # Method 1: Try label_mapping.json
        if os.path.exists(label_mapping_path):
            with open(label_mapping_path, 'r') as f:
                label_mapping = json.load(f)
            id2label = {int(v): k for k, v in label_mapping.items()}
        
        # Method 2: Try config.json
        elif os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            if 'id2label' in config:
                id2label = {int(k): v for k, v in config['id2label'].items()}
            elif 'label2id' in config:
                label2id = config['label2id']
                id2label = {v: k for k, v in label2id.items()}
        
        # Method 3: If no label mapping found, create a default one
        if id2label is None:
            st.warning("Label mapping not found. Creating default mapping...")
            # You may need to adjust this based on your actual number of classes
            num_labels = model.config.num_labels
            id2label = {i: f"Disease_{i}" for i in range(num_labels)}
            
        st.success(f"✅ Successfully loaded BERT model with {len(id2label)} disease classes!")
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

# --- Load symptoms from your dataset ---
@st.cache_data
def load_symptoms_from_dataset():
    """
    Load symptoms from your dataset files.
    This function tries multiple methods to extract symptoms from your data.
    """
    symptoms_set = set()
    
    # Method 1: Load from a dedicated symptoms file
    
    symptoms_file_path = "./Symptoms_list.txt"  # Create this file with your symptoms
    ''''
    if os.path.exists(symptoms_file_path):
        with open(symptoms_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                symptom = line.strip().lower()
                if symptom:
                    symptoms_set.add(symptom)
        st.success(f"✅ Loaded {len(symptoms_set)} symptoms from symptoms_list.txt")
        '''
    if os.path.exists(symptoms_file_path):
         with open(symptoms_file_path, 'r', encoding='utf-8') as f:
               for line in f:
                   for symptom in line.strip().lower().split(','):
                      symptom = symptom.strip()
                      if symptom:
                         symptoms_set.add(symptom)
         st.success(f"✅ Loaded {len(symptoms_set)} symptoms from symptoms_list.txt")

        
    # Method 2: Load from JSON file (if you have symptom mappings)
    symptoms_json_path = "./symptoms_mapping.json"
    if os.path.exists(symptoms_json_path):
        with open(symptoms_json_path, 'r', encoding='utf-8') as f:
            symptom_data = json.load(f)
            if isinstance(symptom_data, dict):
                # If it's a dictionary, extract all values
                for key, value in symptom_data.items():
                    if isinstance(value, list):
                        symptoms_set.update([s.lower() for s in value])
                    else:
                        symptoms_set.add(str(value).lower())
            elif isinstance(symptom_data, list):
                symptoms_set.update([s.lower() for s in symptom_data])
        st.success(f"✅ Loaded additional symptoms from JSON file")
    
    # Method 3: Load from CSV training data (if available)
    training_data_path = "./Final_Augmented_dataset_Diseases_and_Symptoms.csv"  # Update path to your training data
    if os.path.exists(training_data_path):
        try:
            import pandas as pd
            df = pd.read_csv(training_data_path)
            
            # Look for columns that might contain symptoms
            symptom_columns = [col for col in df.columns if 'symptom' in col.lower() or 'text' in col.lower()]
            
            for col in symptom_columns:
                for text in df[col].dropna():
                    # Extract individual symptoms from text
                    text_lower = str(text).lower()
                    # Split by common delimiters and extract individual symptoms
                    potential_symptoms = re.split(r'[,;.\n\r]+', text_lower)
                    for symptom in potential_symptoms:
                        symptom = symptom.strip()
                        if len(symptom) > 2 and len(symptom) < 50:  # Filter reasonable lengths
                            symptoms_set.add(symptom)
            
            st.success(f"✅ Extracted symptoms from training data")
        except Exception as e:
            st.warning(f"Could not load training data: {e}")
    
    # Method 4: Fallback to expanded medical symptoms if no dataset found
    if not symptoms_set:
        fallback_symptoms = [
            # Common symptoms
            "fever", "headache", "cough", "sore throat", "runny nose", "fatigue",
            "body aches", "nausea", "vomiting", "diarrhea", "chest pain", "shortness of breath",
            "dizziness", "rash", "muscle pain", "joint pain", "abdominal pain", "back pain",
            "weight loss", "night sweats", "loss of appetite", "constipation", "bloating",
            "heartburn", "difficulty swallowing", "hoarseness", "ear pain", "vision problems",
            
            # Extended medical symptoms
            "chills", "sweating", "weakness", "malaise", "confusion", "irritability",
            "anxiety", "depression", "insomnia", "drowsiness", "memory loss", "concentration problems",
            "tremor", "seizures", "numbness", "tingling", "burning sensation", "itching",
            "swelling", "bruising", "bleeding", "pale skin", "flushing", "cold hands",
            "hot flashes", "palpitations", "irregular heartbeat", "high blood pressure",
            "low blood pressure", "rapid pulse", "slow pulse", "wheezing", "snoring",
            "difficulty breathing", "shallow breathing", "rapid breathing", "hiccups",
            "bad breath", "dry mouth", "excessive thirst", "frequent urination",
            "painful urination", "blood in urine", "incontinence", "retention",
            "sexual dysfunction", "menstrual irregularities", "vaginal discharge",
            "pelvic pain", "testicular pain", "breast pain", "nipple discharge"
        ]
        symptoms_set.update(fallback_symptoms)
        st.warning("Using fallback symptom list. Consider creating a symptoms_list.txt file with your dataset's symptoms.")
    
    return list(symptoms_set)

# Load symptoms from your dataset
known_symptoms = load_symptoms_from_dataset()

# --- Enhanced symptom extraction ---
def correct_and_extract_symptoms(text):
    """
    Enhanced symptom extraction that works better with your dataset
    """
    corrected_text = correct_spelling(text)
    
    # Method 1: Use SciSpacy NER if available
    extracted_symptoms = []
    if nlp:
        doc = nlp(corrected_text)
        extracted_symptoms = [ent.text.lower() for ent in doc.ents if ent.label_ in ["DISEASE", "SYMPTOM"]]
    
    # Method 2: Direct text matching with known symptoms
    text_lower = corrected_text.lower()
    matched_symptoms = []
    for symptom in known_symptoms:
        if symptom in text_lower:
            matched_symptoms.append(symptom)
    
    # Method 3: Fuzzy matching for partial matches
    words = re.findall(r'\b\w+\b', text_lower)
    fuzzy_matched = []
    for word in words:
        if len(word) > 3:  # Skip very short words
            for symptom in known_symptoms:
                if fuzz.ratio(word, symptom) > 80:  # High similarity threshold
                    fuzzy_matched.append(symptom)
    
    # Combine all methods and remove duplicates
    all_symptoms = list(set(extracted_symptoms + matched_symptoms + fuzzy_matched))
    
    return corrected_text, all_symptoms

# --- Enhanced fuzzy matching ---
def fuzzy_match_symptoms(extracted):
    """
    Enhanced fuzzy matching with better scoring
    """
    matched = set()
    for symptom in extracted:
        try:
            # Try exact match first
            if symptom.lower() in [s.lower() for s in known_symptoms]:
                matched.add(symptom.lower())
                continue
            
            # Then try fuzzy matching
            match, score = process.extractOne(symptom, known_symptoms, scorer=fuzz.WRatio)
            if score > 75:  # Lowered threshold for better matching
                matched.add(match)
        except Exception as e:
            continue
    return list(matched)

# --- Updated Prediction Function for New BERT Model ---
def predict_disease(text):
    if not model or not tokenizer or not id2label:
        return "Model not available", 0.0
    
    try:
        # Tokenize the input text
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512  # Increased max length for BERT
        )
        
        # Move inputs to the same device as the model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get logits and predictions
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        
        # Calculate confidence score
        probabilities = torch.softmax(logits, dim=-1)
        confidence = probabilities[0][prediction].item()
        
        # Get the predicted label
        predicted_label = id2label.get(prediction, f"Unknown_Disease_{prediction}")
        
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

# --- Helper function to create symptoms list from your dataset ---
def create_symptoms_list_from_dataset():
    """
    Helper function to extract unique symptoms from your dataset.
    Run this once to create a comprehensive symptoms list.
    """
    st.subheader("🔧 Create Symptoms List from Dataset")
    
    uploaded_file = st.file_uploader("Upload your training dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            import pandas as pd
            df = pd.read_csv(uploaded_file)
            
            st.write("Dataset columns:", df.columns.tolist())
            
            # Let user select which column contains symptoms/text
            text_column = st.selectbox("Select the column containing symptoms/text:", df.columns)
            
            if st.button("Extract Symptoms"):
                symptoms_set = set()
                
                for text in df[text_column].dropna():
                    text_str = str(text).lower()
                    
                    # Method 1: Extract individual words
                    words = re.findall(r'\b\w+\b', text_str)
                    for word in words:
                        if len(word) > 2 and len(word) < 30:  # Reasonable word length
                            symptoms_set.add(word)
                    
                    # Method 2: Extract phrases (for multi-word symptoms)
                    phrases = re.split(r'[,;.\n\r]+', text_str)
                    for phrase in phrases:
                        phrase = phrase.strip()
                        if len(phrase) > 2 and len(phrase) < 50:
                            symptoms_set.add(phrase)
                
                symptoms_list = sorted(list(symptoms_set))
                
                # Show preview
                st.write(f"Found {len(symptoms_list)} unique symptoms")
                st.write("Sample symptoms:", symptoms_list[:20])
                
                # Save to file
                symptoms_file_path = "./symptoms_list.txt"
                with open(symptoms_file_path, 'w', encoding='utf-8') as f:
                    for symptom in symptoms_list:
                        f.write(f"{symptom}\n")
                
                st.success(f"✅ Saved {len(symptoms_list)} symptoms to {symptoms_file_path}")
                
                # Also create a filtered version with medical terms only
                medical_symptoms = [s for s in symptoms_list if any(med in s for med in [
                    'pain', 'ache', 'fever', 'cough', 'sore', 'nausea', 'vomit', 'dizzy',
                    'headache', 'fatigue', 'weak', 'tired', 'rash', 'itch', 'swelling',
                    'breathing', 'chest', 'throat', 'stomach', 'abdominal', 'back'
                ])]
                
                if medical_symptoms:
                    filtered_file_path = "./medical_symptoms_filtered.txt"
                    with open(filtered_file_path, 'w', encoding='utf-8') as f:
                        for symptom in medical_symptoms:
                            f.write(f"{symptom}\n")
                    st.success(f"✅ Also saved {len(medical_symptoms)} filtered medical symptoms to {filtered_file_path}")
                
        except Exception as e:
            st.error(f"Error processing dataset: {e}")

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
        
        # Add symptom extraction tool
        if password == "1234":
            create_symptoms_list_from_dataset()

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

    # Show developer tools
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
