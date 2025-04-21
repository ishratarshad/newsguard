from flask_cors import CORS
from flask import Flask, request, jsonify
import joblib
from lime.lime_text import LimeTextExplainer
import numpy as np

# Load vectorizer and model
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vectorizer = joblib.load(os.path.join(BASE_DIR, "model", "vectorizer.pkl"))
model = joblib.load(os.path.join(BASE_DIR, "model", "scam_detector.pkl"))

# Model outputs "Real" or "Fake", we map them to nicer labels
label_mapping = {
    "Real": "Legitimate",
    "Fake": "Fake News"
}

# LIME needs a list of class names
class_names = ["Legitimate", "Fake News"]

# Initialize Flask app
app = Flask(__name__)
from flask_cors import CORS
CORS(app)


@app.route('/')
def home():
    return "NewsGuard API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data['text']
    X = vectorizer.transform([text])
    label = model.predict(X)[0]
    proba = float(np.max(model.predict_proba(X)[0]))

    return jsonify({
        "prediction": label_mapping.get(label, "Unknown"),
        "confidence": round(proba, 4)
    })

@app.route('/explain', methods=['POST'])
def explain():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data['text']

    # Define prediction function for LIME
    def predict_proba(texts):
        return model.predict_proba(vectorizer.transform(list(texts)))

    # Run LIME
    explainer = LimeTextExplainer(class_names=class_names)
    explanation = explainer.explain_instance(text, predict_proba, num_features=10)

    label = model.predict(vectorizer.transform([text]))[0]
    confidence = float(np.max(model.predict_proba(vectorizer.transform([text]))[0]))

    # Prepare highlight data (word and weight)
    highlights = explanation.as_list()

    return jsonify({
        "prediction": label_mapping.get(label, "Unknown"),
        "confidence": round(confidence, 4),
        "original_text": text,
        "highlight_data": [
            {"word": word, "weight": round(score, 4)}
            for word, score in highlights
        ]
    })

if __name__ == '__main__':
    app.run(debug=True)
