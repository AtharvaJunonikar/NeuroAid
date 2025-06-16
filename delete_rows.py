import pandas as pd

file_path = 'feedback.csv'
df = pd.read_csv(file_path)

# Print total number of rows
print(f"Total rows in CSV: {len(df)}")

# Define row indices to delete
rows_to_delete = [0]

# Keep only valid indices
valid_rows_to_delete = [i for i in rows_to_delete if i < len(df)]

# Drop rows and reset index
df_updated = df.drop(index=valid_rows_to_delete).reset_index(drop=True)

# Overwrite the original CSV
df_updated.to_csv('feedback.csv', index=False)

print("Original file 'feedback.csv' updated successfully.")
