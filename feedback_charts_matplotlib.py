import pandas as pd
import matplotlib.pyplot as plt
import os

# Load CSV
df = pd.read_csv('feedback.csv')

# Create folder if not exists
os.makedirs('figures', exist_ok=True)

# Save histograms
for column in ['Clarity Score', 'Trust Score', 'UX Score']:
    plt.figure(figsize=(6, 4))
    plt.hist(df[column], bins=[1, 2, 3, 4, 5, 6], edgecolor='black', align='left')
    plt.title(f'{column} Distribution')
    plt.xlabel('Score')
    plt.ylabel('Number of Participants')
    plt.xticks([1, 2, 3, 4, 5])
    plt.savefig(f'figures/{column.lower().replace(" ", "_")}_distribution.png')
    plt.close()
