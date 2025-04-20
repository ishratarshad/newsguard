from flask import Flask, request, jsonify
import joblib
from lime.lime_text import LimeTextExplainer
import numpy as np

# Load vectorizer and model
vectorizer = joblib.load("model/vectorizer.pkl")
model = joblib.load("model/scam_detector.pkl")

# Model outputs "Real" or "Fake", we map them to nicer labels
label_mapping = {
    "Real": "Legitimate",
    "Fake": "Fake News"
}

# LIME needs a list of class names
class_names = ["Legitimate", "Fake News"]

# Initialize Flask app
app = Flask(__name__)

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
    label = model.predict(X)[0]  # returns "Real" or "Fake"
    proba = float(np.max(model.predict_proba(X)[0]))  # max confidence

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

    def predict_proba(texts):
        X = vectorizer.transform(texts)
        return model.predict_proba(X)

    explainer = LimeTextExplainer(class_names=class_names)
    explanation = explainer.explain_instance(text, predict_proba, num_features=5)
    label = model.predict(vectorizer.transform([text]))[0]

    return jsonify({
        "prediction": label_mapping.get(label, "Unknown"),
        "explanation": explanation.as_list()
    })

if __name__ == '__main__':
    app.run(debug=True)
