import pandas as pd
import re
import nltk
import os
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are downloaded
nltk.download('punkt')

# Get absolute file path for structured_data.csv
dataset_path = os.path.join(os.getcwd(), "datasets", "structured_data.csv")
print(f"Checking for dataset at: {dataset_path}")

# Load dataset
if not os.path.exists(dataset_path):
    print(f"Error: structured_data.csv not found at {dataset_path}")
    exit()

df = pd.read_csv(dataset_path)
print("Dataset loaded successfully!")

# Function to clean text data
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
        text = " ".join(word_tokenize(text))  # Tokenize text
    return text

# Apply text cleaning
df["cleaned_text"] = df["text"].apply(clean_text)

# Remove duplicates and missing values
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Save cleaned dataset
cleaned_data_path = os.path.join(os.getcwd(), "datasets", "cleaned_data.csv")
df.to_csv(cleaned_data_path, index=False)

print(f"Data Cleaning Completed: Cleaned dataset saved to {cleaned_data_path}")
