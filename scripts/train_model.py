import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load the cleaned dataset
df = pd.read_csv("datasets/cleaned_data_fixed.csv")
texts = df["cleaned_text"]
labels = df["label"]

print("Label distribution:\n", labels.value_counts())  # Debug check

# Step 2: Extract features using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(texts)

# Save vectorizer and features
joblib.dump(vectorizer, "model/vectorizer.pkl")
joblib.dump(X, "model/features.pkl")

# Step 3: Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Step 4: Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 6: Save model
joblib.dump(model, "model/scam_detector.pkl")
