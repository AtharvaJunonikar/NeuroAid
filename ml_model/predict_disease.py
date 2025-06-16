import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import json

# --- Load model, tokenizer, and label mapping ---
model_path = "./saved_model"

# Load model
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()  # Important: evaluation mode (no dropout)

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# Load label mapping
with open(f"{model_path}/label_mapping.json", "r") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}  # Reverse mapping

# --- Define prediction function ---
def predict_disease(user_input):
    # Tokenize the input
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Forward pass (no gradient needed)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()
    
    # Map prediction to disease name
    predicted_disease = id2label[predicted_class_id]
    
    return predicted_disease

# --- Test it ---
if __name__ == "__main__":
    # Example user input
    user_input = "I have a sore throat and fever and body pain"
    prediction = predict_disease(user_input)
    print(f"Predicted Disease: {prediction}")
