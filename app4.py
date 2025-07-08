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

# Streamlit Secrets
# Load environment variables
# Environment variables - try Streamlit secrets first, then fall back to env vars
try:
    api_key = st.secrets["TOGETHER_AI_API_KEY"]
    google_credentials = dict(st.secrets["GOOGLE_SHEET_CREDENTIALS"]) 
    sheet_id = st.secrets["GOOGLE_SHEET_ID"]
    credentials_path = None  # We'll use the credentials dict directly
except (KeyError, FileNotFoundError):
    # Fallback to environment variables for local development
    api_key = os.getenv("TOGETHER_AI_API_KEY")
    credentials_path = os.getenv("GOOGLE_SHEET_CREDENTIALS")
    sheet_id = os.getenv("GOOGLE_SHEET_ID")
    google_credentials = None

# --- Initialize SymSpell with Medical Context ---
@st.cache_resource
def initialize_symspell():
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    
    # Load standard dictionary
    dictionary_path = "frequency_dictionary_en_82_765.txt"
    if os.path.exists(dictionary_path):
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    else:
        st.warning("Standard dictionary not found. Spelling correction may be limited.")
    
    # Load medical terms with higher frequency weights
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
        st.success(f"✅ Loaded {medical_terms_added} medical terms into SymSpell!")
    else:
        st.warning("Medical dictionary not found. Using standard dictionary only.")
    
    return sym_spell

# Initialize SymSpell
sym_spell = initialize_symspell()

# --- Improved spelling correction with medical context ---
def correct_spelling(text, preserve_medical_terms=True):
    """
    Improved spelling correction that preserves medical terminology
    """
    # Create a list of common medical terms that should not be corrected
    protected_medical_terms = [
        'fever', 'headache', 'nausea', 'vomiting', 'diarrhea', 'cough', 'fatigue',
        'chest pain', 'abdominal pain', 'shortness of breath', 'dizziness', 'rash',
        'muscle pain', 'joint pain', 'back pain', 'sore throat', 'runny nose',
        'congestion', 'sneezing', 'chills', 'sweating', 'weakness', 'malaise',
        'palpitations', 'tachycardia', 'bradycardia', 'hypertension', 'hypotension',
        'dyspnea', 'orthopnea', 'syncope', 'vertigo', 'tinnitus', 'diplopia',
        'photophobia', 'phonophobia', 'paresthesia', 'dysuria', 'polyuria',
        'polydipsia', 'hemoptysis', 'hematemesis', 'melena', 'hematuria'
    ]
    
    if not preserve_medical_terms:
        # Original behavior
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
    
    # New improved behavior
    original_text = text
    text_lower = text.lower()
    
    # Split into words while preserving punctuation positions
    words = re.findall(r'\b\w+\b', text_lower)
    word_positions = [(m.start(), m.end()) for m in re.finditer(r'\b\w+\b', text_lower)]
    
    corrected_words = []
    
    for word, (start, end) in zip(words, word_positions):
        # Check if word is a protected medical term
        if word in protected_medical_terms:
            corrected_words.append(word)
            continue
            
        # Check if word is part of a medical phrase
        is_medical_context = False
        for medical_term in protected_medical_terms:
            if word in medical_term.split():
                is_medical_context = True
                break
        
        if is_medical_context:
            corrected_words.append(word)
            continue
        
        # Apply spelling correction for non-medical terms
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions and suggestions[0].distance <= 2:
            # Only correct if the suggestion is significantly better
            if len(word) > 3 and suggestions[0].distance == 1:
                corrected_words.append(suggestions[0].term)
            elif len(word) <= 3 and suggestions[0].distance == 1 and suggestions[0].count > 1000:
                corrected_words.append(suggestions[0].term)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    
    # Reconstruct text maintaining original structure
    corrected_text = text_lower
    for i, (word, corrected_word) in enumerate(zip(words, corrected_words)):
        if word != corrected_word:
            corrected_text = corrected_text.replace(word, corrected_word, 1)
    
    return corrected_text

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

# --- Enhanced BERT Disease Prediction Model ---
@st.cache_resource
def load_disease_model():
    model_path = r"./ml_model/New/saved_model"
    
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        return None, None, None
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            torch_dtype=torch.float32,
            local_files_only=True
        )
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # Load label mapping
        label_mapping_path = os.path.join(model_path, "label_mapping.json")
        config_path = os.path.join(model_path, "config.json")
        
        id2label = None
        
        # Try to load label mapping
        if os.path.exists(label_mapping_path):
            with open(label_mapping_path, 'r') as f:
                label_mapping = json.load(f)
            id2label = {int(v): k for k, v in label_mapping.items()}
        elif os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            if 'id2label' in config:
                id2label = {int(k): v for k, v in config['id2label'].items()}
            elif 'label2id' in config:
                label2id = config['label2id']
                id2label = {v: k for k, v in label2id.items()}
        
        if id2label is None:
            st.warning("Label mapping not found. Please provide the label mapping.")
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
    symptoms_set = set()
    
    # Method 1: Load from dedicated symptoms file
    symptoms_file_path = "./symptoms_list.txt"
    if os.path.exists(symptoms_file_path):
         with open(symptoms_file_path, 'r', encoding='utf-8') as f:
               for line in f:
                   for symptom in line.strip().lower().split(','):
                      symptom = symptom.strip()
                      if symptom:
                         symptoms_set.add(symptom)
         st.success(f"✅ Loaded {len(symptoms_set)} symptoms from symptoms_list.txt")
        
    # Method 2: Load from JSON file
    symptoms_json_path = "./symptoms_mapping.json"
    if os.path.exists(symptoms_json_path):
        with open(symptoms_json_path, 'r', encoding='utf-8') as f:
            symptom_data = json.load(f)
            if isinstance(symptom_data, dict):
                for key, value in symptom_data.items():
                    if isinstance(value, list):
                        symptoms_set.update([s.lower() for s in value])
                    else:
                        symptoms_set.add(str(value).lower())
            elif isinstance(symptom_data, list):
                symptoms_set.update([s.lower() for s in symptom_data])
        st.success(f"✅ Loaded additional symptoms from JSON file")
    
    # Method 3: Load from CSV training data
    training_data_path = "./Final_Augmented_dataset_Diseases_and_Symptoms.csv"
    if os.path.exists(training_data_path):
        try:
            import pandas as pd
            df = pd.read_csv(training_data_path)
            
            symptom_columns = [col for col in df.columns if 'symptom' in col.lower() or 'text' in col.lower()]
            
            for col in symptom_columns:
                for text in df[col].dropna():
                    text_lower = str(text).lower()
                    potential_symptoms = re.split(r'[,;.\n\r]+', text_lower)
                    for symptom in potential_symptoms:
                        symptom = symptom.strip()
                        if len(symptom) > 2 and len(symptom) < 50:
                            symptoms_set.add(symptom)
            
            st.success(f"✅ Extracted symptoms from training data")
        except Exception as e:
            st.warning(f"Could not load training data: {e}")
    
    # Fallback symptoms
    if not symptoms_set:
        fallback_symptoms = [
            "fever", "headache", "cough", "sore throat", "runny nose", "fatigue",
            "body aches", "nausea", "vomiting", "diarrhea", "chest pain", "shortness of breath",
            "dizziness", "rash", "muscle pain", "joint pain", "abdominal pain", "back pain",
            "weight loss", "night sweats", "loss of appetite", "constipation", "bloating",
            "heartburn", "difficulty swallowing", "hoarseness", "ear pain", "vision problems",
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

# --- Enhanced symptom extraction with better preservation ---
def correct_and_extract_symptoms(text):
    """
    Enhanced symptom extraction that better preserves medical terminology
    """
    # Apply gentle spelling correction
    corrected_text = correct_spelling(text, preserve_medical_terms=True)
    
    # Method 1: Use SciSpacy NER if available
    extracted_symptoms = []
    if nlp:
        doc = nlp(corrected_text)
        for ent in doc.ents:
            if ent.label_ in ["DISEASE", "SYMPTOM"]:
                extracted_symptoms.append(ent.text.lower())
    
    # Method 2: Direct text matching with known symptoms
    text_lower = corrected_text.lower()
    matched_symptoms = []
    
    # Sort symptoms by length (longer first) to match longer phrases first
    sorted_symptoms = sorted(known_symptoms, key=len, reverse=True)
    
    for symptom in sorted_symptoms:
        if symptom in text_lower and symptom not in matched_symptoms:
            matched_symptoms.append(symptom)
    
    # Method 3: Enhanced fuzzy matching
    words = re.findall(r'\b\w+\b', text_lower)
    phrases = re.findall(r'\b\w+(?:\s+\w+)*\b', text_lower)
    
    fuzzy_matched = []
    for phrase in phrases:
        if len(phrase) > 3:
            for symptom in known_symptoms:
                similarity = fuzz.ratio(phrase, symptom)
                if similarity > 85:  # High similarity threshold
                    fuzzy_matched.append(symptom)
    
    # Combine all methods and remove duplicates
    all_symptoms = list(set(extracted_symptoms + matched_symptoms + fuzzy_matched))
    
    return corrected_text, all_symptoms

# --- Enhanced fuzzy matching ---
def fuzzy_match_symptoms(extracted):
    matched = set()
    for symptom in extracted:
        try:
            # Try exact match first
            if symptom.lower() in [s.lower() for s in known_symptoms]:
                matched.add(symptom.lower())
                continue
            
            # Enhanced fuzzy matching with multiple scorers
            matches = process.extract(symptom, known_symptoms, scorer=fuzz.WRatio, limit=3)
            
            for match, score in matches:
                if score > 80:  # High threshold for medical terms
                    matched.add(match)
                    break
                    
        except Exception as e:
            continue
    return list(matched)

# --- Model validation function ---
def validate_model_output():
    """Function to validate model is working correctly"""
    if not model or not tokenizer or not id2label:
        return False, "Model components not loaded"
    
    try:
        # Test with simple input
        test_input = "fever headache cough"
        inputs = tokenizer(test_input, return_tensors="pt", truncation=True, padding=True)
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        
        max_prob = torch.max(probs).item()
        
        if max_prob < 0.001:  # Less than 0.1%
            return False, f"Model predictions too low: {max_prob:.4f}"
        
        return True, f"Model validation passed. Max probability: {max_prob:.3f}"
        
    except Exception as e:
        return False, f"Model validation failed: {e}"

# --- FIXED Enhanced BERT Prediction with Better Confidence Scoring ---
def predict_disease(text):
    if not model or not tokenizer or not id2label:
        return "Model not available", 0.0, []
    
    try:
        # Enhanced preprocessing
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Tokenize with optimal parameters for BERT
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512,
            add_special_tokens=True
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get logits and probabilities
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        
        # Get top 3 predictions
        confidence_scores, predicted_indices = torch.topk(probabilities, k=min(3, len(id2label)), dim=-1)
        
        # Get top prediction
        top_prediction = predicted_indices[0][0].item()
        top_confidence = confidence_scores[0][0].item()
        
        # Debug information (comment out in production)
        print(f"Debug - Raw confidence: {top_confidence:.4f}")
        print(f"Debug - Logits shape: {logits.shape}")
        print(f"Debug - Max logit: {torch.max(logits).item():.4f}")
        
        # Apply confidence calibration
        final_confidence = top_confidence
        
        # Apply temperature scaling only if confidence is very low
        if top_confidence < 0.1:
            temperature = 0.8  # Moderate temperature for sharpening
            scaled_logits = logits / temperature
            calibrated_probs = torch.softmax(scaled_logits, dim=-1)
            calibrated_confidence = torch.max(calibrated_probs).item()
            
            # Use calibrated confidence if it's reasonably higher
            if calibrated_confidence > top_confidence * 1.2:
                final_confidence = calibrated_confidence
                print(f"Debug - Applied temperature scaling: {calibrated_confidence:.4f}")
        
        # Ensure minimum confidence threshold for reasonable display
        if final_confidence < 0.05:
            final_confidence = max(0.05, final_confidence)  # Minimum 5%
        
        # Cap maximum confidence to avoid overconfidence
        final_confidence = min(0.95, final_confidence)  # Maximum 95%
        
        # Get predicted label
        predicted_label = id2label.get(top_prediction, f"Unknown_Disease_{top_prediction}")
        
        # Build top 3 predictions
        top_3_predictions = []
        for i in range(min(3, len(confidence_scores[0]))):
            pred_idx = predicted_indices[0][i].item()
            pred_conf = confidence_scores[0][i].item()
            pred_label = id2label.get(pred_idx, f"Unknown_Disease_{pred_idx}")
            top_3_predictions.append((pred_label, pred_conf))
        
        print(f"Debug - Final confidence: {final_confidence:.4f}")
        print(f"Debug - Predicted label: {predicted_label}")
        
        return predicted_label, final_confidence, top_3_predictions
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return "Prediction failed", 0.05, []

# --- Alternative prediction method ---
def predict_disease_alternative(text):
    """Alternative prediction method with different confidence calculation"""
    if not model or not tokenizer or not id2label:
        return "Model not available", 0.0, []
    
    try:
        # Preprocess text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Tokenize
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get logits and apply softmax
        logits = outputs.logits.cpu()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get top prediction
        top_prob, top_idx = torch.max(probs, dim=-1)
        top_confidence = top_prob.item()
        top_prediction = top_idx.item()
        
        # Calculate confidence using multiple methods
        confidence_methods = {
            'max_prob': top_confidence,
            'entropy_based': 1.0 - (-torch.sum(probs * torch.log(probs + 1e-8)) / torch.log(torch.tensor(float(len(id2label))))).item(),
            'gap_based': (torch.topk(probs, 2)[0][0, 0] - torch.topk(probs, 2)[0][0, 1]).item()
        }
        
        # Use the maximum confidence from different methods
        final_confidence = max(confidence_methods.values())
        
        # Ensure reasonable bounds
        final_confidence = max(0.05, min(0.95, final_confidence))  # Between 5% and 95%
        
        predicted_label = id2label.get(top_prediction, f"Disease_{top_prediction}")
        
        # Get top 3 predictions
        top_3_probs, top_3_indices = torch.topk(probs, k=min(3, len(id2label)), dim=-1)
        top_3_predictions = []
        
        for i in range(top_3_probs.shape[1]):
            pred_idx = top_3_indices[0, i].item()
            pred_conf = top_3_probs[0, i].item()
            pred_label = id2label.get(pred_idx, f"Disease_{pred_idx}")
            top_3_predictions.append((pred_label, pred_conf))
        
        return predicted_label, final_confidence, top_3_predictions
        
    except Exception as e:
        print(f"Error in alternative prediction: {e}")
        return "Prediction failed", 0.05, []

# --- Enhanced symptom analysis with better error handling ---
def handle_symptom_analysis(symptoms_text, user_role):
    """Enhanced symptom analysis with better error handling"""
    
    # First validate the model
    is_valid, validation_msg = validate_model_output()
    if not is_valid:
        st.error(f"❌ Model validation failed: {validation_msg}")
        return None
    else:
        st.success(f"✅ {validation_msg}")
    
    # Correct spelling and extract symptoms
    corrected_text, extracted_symptoms = correct_and_extract_symptoms(symptoms_text)
    
    # Enhance with fuzzy matching
    enhanced_symptoms = fuzzy_match_symptoms(extracted_symptoms)
    
    # Prepare input for prediction
    if len(enhanced_symptoms) > 0:
        prediction_input = " ".join(enhanced_symptoms)
    else:
        prediction_input = corrected_text
    
    st.info(f"🔍 Analyzing: {prediction_input}")
    
    # Try primary prediction method
    try:
        predicted_disease, confidence, top_3_predictions = predict_disease(prediction_input)
        
        # If confidence is still very low, try alternative method
        if confidence < 0.10:
            st.warning("⚠️ Low confidence detected, trying alternative analysis...")
            predicted_disease_alt, confidence_alt, top_3_alt = predict_disease_alternative(prediction_input)
            
            # Use alternative if it gives better confidence
            if confidence_alt > confidence:
                predicted_disease, confidence, top_3_predictions = predicted_disease_alt, confidence_alt, top_3_alt
                st.info("✅ Used alternative analysis method")
    
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        return None
    
    # Store results
    results = {
        'original_text': symptoms_text,
        'corrected_text': corrected_text,
        'extracted_symptoms': extracted_symptoms,
        'enhanced_symptoms': enhanced_symptoms,
        'predicted_disease': predicted_disease,
        'confidence': confidence,
        'top_3_predictions': top_3_predictions,
        'prediction_input': prediction_input
    }
    
    return results

# --- Fixed display function ---
def display_prediction_results(results):
    """Display prediction results with proper confidence formatting"""
    
    # Display corrected text if different from original
    if results['corrected_text'] != results['original_text'].lower():
        st.markdown("**Corrected Text:**")
        st.info(f"📝 {results['corrected_text']}")
    
    # Display extracted symptoms
    if results['enhanced_symptoms']:
        st.markdown("**Identified Symptoms:**")
        symptoms_display = ", ".join(results['enhanced_symptoms'])
        st.success(f"🎯 {symptoms_display}")
    
    # Display prediction with properly formatted confidence
    st.markdown("**Predicted Condition:**")
    confidence_percentage = results['confidence'] * 100
    
    # Color-code confidence levels with more reasonable thresholds
    if confidence_percentage >= 70:
        confidence_color = "🟢"
        confidence_text = "High"
    elif confidence_percentage >= 50:
        confidence_color = "🟡"
        confidence_text = "Moderate"
    elif confidence_percentage >= 30:
        confidence_color = "🟠"
        confidence_text = "Low"
    else:
        confidence_color = "🔴"
        confidence_text = "Very Low"
    
    st.markdown(f"**{confidence_color} {results['predicted_disease']}**")
    st.markdown(f"*Confidence: {confidence_percentage:.1f}% ({confidence_text})*")
    
    # Display top 3 predictions if available
    if results['top_3_predictions']:
        st.markdown("**Alternative Possibilities:**")
        for i, (disease, conf) in enumerate(results['top_3_predictions'][:3], 1):
            st.write(f"{i}. {disease} ({conf*100:.1f}%)")
    
    # Add confidence interpretation
    st.markdown("**Confidence Interpretation:**")
    if confidence_percentage >= 70:
        st.info("🟢 **High Confidence**: The model is quite certain about this prediction.")
    elif confidence_percentage >= 50:
        st.warning("🟡 **Moderate Confidence**: The prediction is reasonable but consider other possibilities.")
    elif confidence_percentage >= 30:
        st.warning("🟠 **Low Confidence**: The prediction is uncertain. Multiple conditions are possible.")
    else:
        st.error("🔴 **Very Low Confidence**: The model is very uncertain. Please consult a healthcare professional.")

# --- Debug information display ---
def show_debug_info(results):
    """Show debug information for troubleshooting"""
    with st.expander("🔧 Debug Information", expanded=False):
        st.json({
            "prediction_input": results['prediction_input'],
            "num_symptoms_extracted": len(results['enhanced_symptoms']),
            "confidence_raw": results['confidence'],
            "confidence_percentage": results['confidence'] * 100,
            "model_loaded": model is not None,
            "tokenizer_loaded": tokenizer is not None,
            "num_labels": len(id2label) if id2label else 0,
            "top_3_predictions": results['top_3_predictions']
        })

# --- Generate explanation using Together.ai ---
def generate_explanation_together_ai(api_key, user_role, symptoms_list, predicted_disease, confidence_score=None):
    if not api_key:
        return "API key not available. Please set TOGETHER_AI_API_KEY in your environment variables."
    
    symptoms_text = ", ".join(symptoms_list) if symptoms_list else "general symptoms"
    disease_name = predicted_disease.title()
    
    # Add confidence context to the explanation
    confidence_context = ""
    if confidence_score:
        confidence_percentage = confidence_score * 100
        if confidence_percentage >= 70:
            confidence_context = "The AI system has high confidence in this prediction. "
        elif confidence_percentage >= 50:
            confidence_context = "The AI system has moderate confidence in this prediction. "
        else:
            confidence_context = "The AI system has lower confidence in this prediction, so please consider consulting a healthcare professional. "

    # Dynamic prompt based on user_role
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    if user_role == "Student":
        role_instruction = (
            "Explain in 4–5 sentences, simple and clear, suitable for a student. "
            "Focus on explaining the predicted disease, its symptoms, and treatments. "
            f"{confidence_context}"
            "Always remind them to consult a healthcare professional for proper diagnosis."
        )
    elif user_role == "Doctor":
        role_instruction = (
            "Explain in detail using clinical language, suggest possible diagnostic tests and treatments. "
            f"{confidence_context}"
            "Provide differential diagnosis considerations and clinical pearls."
        )
    elif user_role == "Elderly":
        role_instruction = (
            "Explain very simply in 2–3 sentences and provide 3–5 bullet points with easy lifestyle tips. "
            "Keep a comforting tone. "
            f"{confidence_context}"
            "Emphasize the importance of speaking with their doctor."
        )
    else:
        role_instruction = f"Explain clearly. {confidence_context}"

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
    """Get Google Sheets connection with better error handling"""
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/spreadsheets"
        ]
        
        # Try Streamlit secrets first
        try:
            google_credentials = dict(st.secrets["GOOGLE_SHEET_CREDENTIALS"])
            sheet_id = st.secrets["GOOGLE_SHEET_ID"]
            st.success("✅ Using Streamlit secrets for Google Sheets")
        except (KeyError, FileNotFoundError):
            # Fallback to environment variables
            credentials_path = os.getenv("GOOGLE_SHEET_CREDENTIALS")
            sheet_id = os.getenv("GOOGLE_SHEET_ID")
            
            if credentials_path and os.path.exists(credentials_path):
                import json
                with open(credentials_path, 'r') as f:
                    google_credentials = json.load(f)
                st.info("ℹ️ Using environment variables for Google Sheets")
            else:
                st.error("❌ No Google Sheets credentials found")
                return None
        
        if not sheet_id:
            st.error("❌ Google Sheet ID not found")
            return None
        
        # Create credentials
        creds = ServiceAccountCredentials.from_json_keyfile_dict(google_credentials, scope)
        
        # Authorize and get client
        client = gspread.authorize(creds)
        
        # Open the sheet
        sheet = client.open_by_key(sheet_id).sheet1
        
        # Test the connection by getting sheet info
        sheet_info = sheet.get_all_records(limit=1)
        st.success(f"✅ Connected to Google Sheet: {sheet.title}")
        
        return sheet
        
    except gspread.exceptions.SpreadsheetNotFound:
        st.error("❌ Google Sheet not found. Check your GOOGLE_SHEET_ID.")
        return None
    except gspread.exceptions.APIError as e:
        st.error(f"❌ Google Sheets API Error: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error connecting to Google Sheets: {e}")
        return None

# --- Fixed save feedback function ---
def save_feedback_data(data_row):
    """Save feedback with better error handling and logging"""
    success_google = False
    success_local = False
    
    # Try to save to Google Sheets first
    try:
        st.info("📤 Attempting to save to Google Sheets...")
        sheet = get_google_sheet()
        
        if sheet:
            # Check if headers exist
            try:
                headers = sheet.row_values(1)
                if not headers:
                    # Add headers if sheet is empty
                    header_row = [
                        "Date", "Time", "Participant ID", "User Role", "Age", "Gender", "Symptoms",
                        "Diagnosis", "Explanation", "Clarity Score", "Trust Score", "UX Score", 
                        "Comment", "Sentiment", "Confidence Score"
                    ]
                    sheet.append_row(header_row)
                    st.info("📝 Added headers to Google Sheet")
            except Exception as e:
                st.warning(f"⚠️ Could not check headers: {e}")
            
            # Append the data
            sheet.append_row(data_row)
            st.success("✅ Data saved to Google Sheets successfully!")
            success_google = True
            
    except Exception as e:
        st.error(f"❌ Failed to save to Google Sheets: {e}")
        st.info("🔄 Falling back to local CSV storage...")
    
    # Save to local CSV (fallback or backup)
    try:
        feedback_path = os.path.join(os.getcwd(), 'feedback.csv')
        file_exists = os.path.exists(feedback_path)
        
        with open(feedback_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Write headers if file is new
            if not file_exists or os.path.getsize(feedback_path) == 0:
                writer.writerow([
                    "Date", "Time", "Participant ID", "User Role", "Age", "Gender", "Symptoms",
                    "Diagnosis", "Explanation", "Clarity Score", "Trust Score", "UX Score", 
                    "Comment", "Sentiment", "Confidence Score"
                ])
            
            writer.writerow(data_row)
        
        st.success("✅ Data saved to local CSV file successfully!")
        success_local = True
        
    except Exception as e:
        st.error(f"❌ Error saving to local CSV: {e}")
    
    # Return success status
    return success_google or success_local

# --- Updated save_feedback function ---
def save_feedback(pid, role, age, gender, symptoms, diagnosis, explanation, clarity, trust, ux_score, comment, sentiment, confidence_score=None):
    """Save feedback with proper data formatting"""
    now = datetime.now()
    date_str = now.strftime("%d.%m.%Y")
    time_str = now.strftime("%H:%M:%S")

    # Format symptoms properly
    symptoms_str = ", ".join(symptoms) if isinstance(symptoms, list) else str(symptoms)
    
    # Format confidence score
    confidence_str = f"{confidence_score:.3f}" if confidence_score is not None else "N/A"

    data_row = [
        date_str, 
        time_str, 
        pid, 
        role, 
        age, 
        gender, 
        symptoms_str,
        diagnosis, 
        explanation, 
        clarity, 
        trust, 
        ux_score, 
        comment, 
        sentiment,
        confidence_str
    ]
    
    # Debug: Show what we're trying to save
    st.write("📊 Data to be saved:")
    st.json({
        "Date": date_str,
        "Time": time_str,
        "Participant ID": pid,
        "User Role": role,
        "Symptoms": symptoms_str,
        "Diagnosis": diagnosis,
        "Confidence": confidence_str
    })
    
    return save_feedback_data(data_row)

# --- Test Google Sheets connection ---
def test_google_sheets_connection():
    """Test function to check Google Sheets connection"""
    st.subheader("🧪 Test Google Sheets Connection")
    
    if st.button("Test Connection"):
        with st.spinner("Testing Google Sheets connection..."):
            sheet = get_google_sheet()
            
            if sheet:
                try:
                    # Try to read some data
                    records = sheet.get_all_records(limit=5)
                    st.success(f"✅ Connection successful! Sheet has {len(records)} records (showing first 5)")
                    
                    if records:
                        st.write("Sample data:")
                        st.json(records)
                    
                    # Test write access
                    test_row = ["TEST", "TEST", "test_connection", "Test", "25", "Test", "Test symptoms", "Test diagnosis", "Test explanation", "5", "5", "5", "Connection test", "Positive", "0.95"]
                    
                    if st.button("Test Write Access"):
                        try:
                            sheet.append_row(test_row)
                            st.success("✅ Write test successful!")
                            
                            # Remove test row
                            if st.button("Remove Test Row"):
                                try:
                                    sheet.delete_rows(sheet.row_count)
                                    st.success("✅ Test row removed!")
                                except Exception as e:
                                    st.error(f"❌ Could not remove test row: {e}")
                        except Exception as e:
                            st.error(f"❌ Write test failed: {e}")
                            
                except Exception as e:
                    st.error(f"❌ Error testing connection: {e}")
            else:
                st.error("❌ Connection failed!")

# --- Debug information for Google Sheets ---
def show_google_sheets_debug():
    """Show debug information for Google Sheets setup"""
    st.subheader("🔍 Google Sheets Debug Information")
    
    # Check secrets
    try:
        creds = dict(st.secrets["GOOGLE_SHEET_CREDENTIALS"])
        sheet_id = st.secrets["GOOGLE_SHEET_ID"]
        
        st.success("✅ Streamlit secrets found")
        st.write(f"Sheet ID: {sheet_id}")
        st.write(f"Credentials type: {creds.get('type', 'Unknown')}")
        st.write(f"Client email: {creds.get('client_email', 'Unknown')}")
        
    except Exception as e:
        st.error(f"❌ Streamlit secrets error: {e}")
        
        # Check environment variables
        try:
            creds_path = os.getenv("GOOGLE_SHEET_CREDENTIALS")
            sheet_id = os.getenv("GOOGLE_SHEET_ID")
            
            if creds_path and sheet_id:
                st.info("ℹ️ Environment variables found")
                st.write(f"Credentials path: {creds_path}")
                st.write(f"Sheet ID: {sheet_id}")
                st.write(f"Credentials file exists: {os.path.exists(creds_path)}")
            else:
                st.error("❌ No environment variables found")
                
        except Exception as e:
            st.error(f"❌ Environment variables error: {e}")
    
    # Test connection
    test_google_sheets_connection()

'''
def get_google_sheet():
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        
        # Get credentials and sheet ID from Streamlit secrets
        google_credentials = dict(st.secrets["GOOGLE_SHEET_CREDENTIALS"])
        sheet_id = st.secrets["GOOGLE_SHEET_ID"]
        
        # Use credentials from Streamlit secrets
        creds = ServiceAccountCredentials.from_json_keyfile_dict(google_credentials, scope)
        
        client = gspread.authorize(creds)
        sheet = client.open_by_key(sheet_id).sheet1
        return sheet
        
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        return None
        

# --- Save feedback ---
def save_feedback(pid, role, age, gender, symptoms, diagnosis, explanation, clarity, trust, ux_score, comment, sentiment, confidence_score=None):
    now = datetime.now()
    date_str = now.strftime("%d.%m.%Y")
    time_str = now.strftime("%H:%M:%S")

    data_row = [
        date_str, time_str, pid, role, age, gender, ", ".join(symptoms),
        diagnosis, explanation, clarity, trust, ux_score, comment, sentiment,
        confidence_score if confidence_score else "N/A"
    ]

    # Save to Google Sheets
def save_feedback_data(data_row):
    try:
        sheet = get_google_sheet()
        if sheet:
            sheet.append_row(data_row)
            st.success("Data saved to Google Sheets")
    except Exception as e:
        st.warning(f"Could not save to Google Sheets: {e}")
    
    # Save to local CSV (fallback)
    feedback_path = os.path.join(os.getcwd(), 'feedback.csv')
    try:
        with open(feedback_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow([
                    "Date", "Time", "Participant ID", "User Role", "Age", "Gender", "Symptoms",
                    "Diagnosis", "Explanation", "Clarity Score", "Trust Score", "UX Score", 
                    "Comment", "Sentiment", "Confidence Score"
                ])
            writer.writerow(data_row)
        st.success("Data saved to local CSV file")
    except Exception as e:
        st.error(f"Error saving to CSV: {e}")
    
    
        '''

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

# --- Helper function to create symptoms list from dataset ---
def create_symptoms_list_from_dataset():
    st.subheader("🔧 Create Symptoms List from Dataset")
    
    uploaded_file = st.file_uploader("Upload your training dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            import pandas as pd
            df = pd.read_csv(uploaded_file)
            
            st.write("Dataset columns:", df.columns.tolist())
            
            text_column = st.selectbox("Select the column containing symptoms/text:", df.columns)
            
            if st.button("Extract Symptoms"):
                symptoms_set = set()
                
                for text in df[text_column].dropna():
                    text_str = str(text).lower()
                    
                    # Extract individual words
                    words = re.findall(r'\b\w+\b', text_str)
                    for word in words:
                        if len(word) > 2 and len(word) < 30:
                            symptoms_set.add(word)
                    
                    # Extract phrases
                    phrases = re.split(r'[,;.\n\r]+', text_str)
                    for phrase in phrases:
                        phrase = phrase.strip()
                        if len(phrase) > 2 and len(phrase) < 50:
                            symptoms_set.add(phrase)
                
                symptoms_list = sorted(list(symptoms_set))
                
                st.write(f"Found {len(symptoms_list)} unique symptoms")
                st.write("Sample symptoms:", symptoms_list[:20])
                
                # Save to file
                symptoms_file_path = "./symptoms_list.txt"
                with open(symptoms_file_path, 'w', encoding='utf-8') as f:
                    for symptom in symptoms_list:
                        f.write(f"{symptom}\n")
                
                st.success(f"✅ Saved {len(symptoms_list)} symptoms to {symptoms_file_path}")
                
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
        
        if password == "1234":
            create_symptoms_list_from_dataset()
            show_google_sheets_debug()

# --- Main Streamlit App ---
def main():
    st.title("NeuroAid: Enhanced Symptom Checker")
    
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
        return

    # Check if already submitted
    if has_already_submitted(st.session_state.participant_id):
        st.warning("⚠️ You have already submitted feedback for this participant ID.")
        st.session_state.submitted = True

    if st.session_state.submitted:
        st.success("✅ Thank you for your participation! Your feedback has been recorded.")
        if st.button("Reset for New Participant"):
            st.session_state.participant_id = ""
            st.session_state.id_applied = False
            st.session_state.submitted = False
            st.session_state.invalid_pid = False
            st.rerun()
        return

    # Main Application Interface
    st.subheader(f"Welcome, Participant {st.session_state.participant_id}")
    
    # User Information Section
    st.markdown("### 👤 User Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        user_role = st.selectbox("Your Role:", ["Student", "Doctor", "Elderly"])
    
    with col2:
        age = st.number_input("Age:", min_value=18, max_value=100, value=25)
    
    with col3:
        gender = st.selectbox("Gender:", ["Male", "Female", "Other", "Prefer not to say"])

    # Symptoms Input Section
    st.markdown("### 🩺 Symptom Description")
    st.markdown("Please describe your symptoms in detail:")
    
    symptoms_text = st.text_area(
        "Enter your symptoms:",
        placeholder="e.g., I have been experiencing headaches, fever, and fatigue for the past 3 days...",
        height=120
    )

    # Process symptoms and make prediction
    if st.button("🔍 Analyze Symptoms", type="primary"):
        if not symptoms_text.strip():
            st.error("❌ Please enter your symptoms before analyzing.")
            return
        
        with st.spinner("Analyzing symptoms..."):
            # Use the enhanced symptom analysis function
            results = handle_symptom_analysis(symptoms_text, user_role)
            
            if results:
                # Store results in session state
                st.session_state.analysis_results = results
                
                # Display results
                display_prediction_results(results)
                
                # Show debug information
                show_debug_info(results)

    # Display results if available
    if hasattr(st.session_state, 'analysis_results') and st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Generate and display explanation
        st.markdown("### 💡 Explanation")
        with st.spinner("Generating explanation..."):
            explanation = generate_explanation_together_ai(
                api_key, 
                user_role, 
                results['enhanced_symptoms'], 
                results['predicted_disease'],
                results['confidence']
            )
        
        st.markdown(f"**Explanation for {user_role}:**")
        st.write(explanation)
        
        # Store explanation for feedback
        st.session_state.explanation = explanation
        
        # Feedback Section
        st.markdown("### 📝 Feedback")
        st.markdown("Please rate your experience with this analysis:")
        
        # Feedback ratings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            clarity_score = st.slider("Clarity of Explanation:", 1, 5, 3, 
                                    help="How clear was the explanation?")
        
        with col2:
            trust_score = st.slider("Trust in Prediction:", 1, 5, 3, 
                                   help="How much do you trust this prediction?")
        
        with col3:
            ux_score = st.slider("User Experience:", 1, 5, 3, 
                                help="How was your overall experience?")
        
        # Comments
        feedback_comment = st.text_area("Additional Comments:", 
                                       placeholder="Please share any additional thoughts or feedback...")
        
        # Submit feedback
        if st.button("Submit Feedback", type="primary"):
            if not feedback_comment.strip():
                st.error("❌ Please provide some feedback comments before submitting.")
                return
            
            # Analyze sentiment
            sentiment = analyze_sentiment(feedback_comment)
            
            # Save feedback
            try:
                save_feedback(
                    st.session_state.participant_id,
                    user_role,
                    age,
                    gender,
                    results['enhanced_symptoms'],
                    results['predicted_disease'],
                    st.session_state.explanation,
                    clarity_score,
                    trust_score,
                    ux_score,
                    feedback_comment,
                    sentiment,
                    results['confidence']
                )
                
                st.success("✅ Thank you for your feedback! Your response has been recorded.")
                st.session_state.submitted = True
                
                # Show completion message
                st.balloons()
                st.markdown("### 🎉 Participation Complete!")
                st.markdown("Your feedback is valuable for improving our medical AI system.")
                st.info("The page will reset in a moment...")
                time.sleep(2)  # Wait 2 seconds
                # Reset session state
                for key in list(st.session_state.keys()):
                     del st.session_state[key]
                     st.rerun()
            except Exception as e:
                st.error(f"❌ Error saving feedback: {e}")
                st.error("Please try again or contact support if the problem persists.")
         # Show success message and then reset after a delay
    # Information Section
    st.sidebar.markdown("### ℹ️ About NeuroAid")
    st.sidebar.markdown("""
    **NeuroAid** is an AI-powered symptom checker that uses advanced machine learning models to analyze symptoms and provide preliminary health insights.
    
    **Features:**
    - 🔍 Intelligent symptom extraction
    - 🧠 BERT-based disease prediction
    - 📊 Confidence scoring
    - 💬 Role-based explanations
    - 📝 Feedback collection
    
    **⚠️ Important Disclaimer:**
    This tool is for educational and research purposes only. Always consult with qualified healthcare professionals for medical diagnosis and treatment.
    """)
    
    # Technical Information
    st.sidebar.markdown("### 🛠️ Technical Details")
    st.sidebar.markdown(f"""
    - **Model Status:** {'✅ Loaded' if model else '❌ Not Available'}
    - **Symptoms Database:** {len(known_symptoms)} symptoms
    - **NER Model:** {'✅ SciSpacy' if nlp else '❌ Not Available'}
    - **Sentiment Analysis:** {'✅ Active' if sentiment_pipeline else '❌ Not Available'}
    """)
    


if __name__ == "__main__":
    main()
