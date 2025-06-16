# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import json

# 1. Load your dataset
df = pd.read_csv('dataset.csv')

# 2. Preprocess: Combine all symptoms into one text field
def combine_symptoms(row):
    symptoms = [str(row[f"Symptom_{i}"]).strip() for i in range(1, 18) if pd.notna(row.get(f"Symptom_{i}"))]
    return ", ".join(symptoms)

df['symptoms_text'] = df.apply(combine_symptoms, axis=1)

# 3. Encode labels
labels = df['Disease'].unique().tolist()
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}
df['label'] = df['Disease'].map(label2id)

# 4. Split into train and test
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['symptoms_text'],
    df['label'],
    test_size=0.2,
    random_state=42
)

# 5. Tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

# 6. Dataset preparation
class SymptomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SymptomDataset(train_encodings, train_labels.tolist())
val_dataset = SymptomDataset(val_encodings, val_labels.tolist())

# 7. Model initialization
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(labels)
)

# ✅ 8. Training arguments (Force CPU explicitly)
training_args = TrainingArguments(
    output_dir='./results',         
    num_train_epochs=3,              
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,   
    warmup_steps=100,                
    weight_decay=0.01,              
    logging_dir='./logs',            
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    no_cuda=True   # <--- This disables CUDA/MPS => FORCE CPU ✅
)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 10. Train
trainer.train()

# 11. Save the model and tokenizer
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

# Save the label mappings
with open("./saved_model/label_mapping.json", "w") as f:
    json.dump(label2id, f)

print("✅ Model training complete and saved successfully!")
