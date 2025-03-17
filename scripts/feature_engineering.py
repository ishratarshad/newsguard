import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os


os.makedirs('model', exist_ok=True)

# cleaned data from data_cleaning.py
print("Loading cleaned data")
df = pd.read_csv('datasets/cleaned_data.csv')

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_features=5000
)

# Fit and transform the text data
print("Generating TF-IDF features")
features = tfidf.fit_transform(df['cleaned_text'])

# Save the vectorizer and features
print("Saving outputs to model directory...")
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('model/features.pkl', 'wb') as f:
    pickle.dump(features, f)

print("Feature engineering complete!")
print(f"Number of articles processed: {features.shape[0]}")
print(f"Number of features generated: {features.shape[1]}")
print(f"Outputs saved: model/vectorizer.pkl and model/features.pkl")
