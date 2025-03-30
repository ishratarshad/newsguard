# Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer 
import pandas as pd 
import pickle 
import os
import numpy as np 
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import scipy.sparse 

# Define base directory
base_folder = "C:\\Users\\rayat\\OneDrive\\Desktop\\MyProjects\\NewsGuardAi\\newsguard"
model_folder = os.path.join(base_folder, "model")  
os.makedirs(model_folder, exist_ok=True)  # Ensure model folder exists

# Load cleaned dataset
cleaned_data_set = pd.read_csv(os.path.join(base_folder, "datasets", "cleaned_data_fixed.csv"))   

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer()  # You can adjust max_features if needed by doing max_features = whatever value you want. 
X_tfidf = tfidf.fit_transform(cleaned_data_set['cleaned_text'])  
y = cleaned_data_set['label']  # Assuming 'label' column contains scam/non-scam labels

# Save vectorizer
vectorizer_path = os.path.join(model_folder, "vectorizer.pkl")
with open(vectorizer_path, "wb") as f: 
    pickle.dump(tfidf, f)
print(f"Vectorizer saved at: {vectorizer_path}")

# Save extracted features
features_path = os.path.join(model_folder, "features.npz")
scipy.sparse.save_npz(features_path, X_tfidf)
print(f"Features saved at: {features_path}")

# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train Naive Bayes Classifier
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Make predictions
y_pred = nb_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save trained model
model_path = os.path.join(model_folder, "scam_detector.pkl")
with open(model_path, "wb") as f:
    pickle.dump(nb_model, f)
print(f"Model saved at: {model_path}")
