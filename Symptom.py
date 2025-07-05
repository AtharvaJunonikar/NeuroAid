'''
import pandas as pd

df = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")
column_names = df.columns.tolist()

with open("symptoms_list.txt", "w", encoding="utf-8") as f:
    f.write(",".join(column_names))

'''
import pandas as pd

# Load CSV
df = pd.read_csv('Final_Augmented_dataset_Diseases_and_Symptoms.csv')

# Extract unique values from a column, e.g., 'ColumnName'
unique_values = df['diseases'].unique()

# Convert to list if needed
unique_values = unique_values.tolist()

with open("disease_list.txt", "w", encoding="utf-8") as f:
    f.write(",".join(unique_values))

#print(unique_values)



