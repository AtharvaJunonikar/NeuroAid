import pandas as pd

# Step 1: Load dataset
df = pd.read_csv("dataset.csv")

# Step 2: Combine symptoms into one text field
def combine_symptoms(row):
    symptoms = []
    for i in range(1, 18):  # Symptom_1 to Symptom_17
        symptom = row.get(f'Symptom_{i}')
        if pd.notnull(symptom):
            symptoms.append(symptom.replace("_", " "))  # Replace underscores with spaces
    return ", ".join(symptoms)

# Create new 'Symptoms' column
df['Symptoms'] = df.apply(combine_symptoms, axis=1)

# Step 3: Keep only 'Symptoms' and 'Disease'
df = df[['Symptoms', 'Disease']]

# Step 4: See the data
print(df.head())
