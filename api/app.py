from flask import Flask, request, jsonify
import joblib

# Flask for the API server, obviously
# request is an object that represents the HTTP request
# jsonify allows for JSON response to be returned
# joblib to load in the pkl files that exist under model/


# load and store the pkl files as global variables, using exception handling justtt in case the model files can't be read or don't exist
try:
    vectorizer = joblib.load('model/vectorizer.pkl')
    model = joblib.load('model/scam_detector.pkl')
except Exception as e:
    print(f"Error loading model files: {e}")
    vectorizer = None
    model = None


app = Flask(__name__)

# route for 
@app.route('/')

# Return simple response if no route has been selected
def index():
    return jsonify({"message": "API in use"}), 200

# predict route, which ONLY accepts POST requests, any other request will lead to a 405 method (not allowed)
@app.route('/predict', methods=['POST'])

# the predict route will made a prediction using the model and the request data as input
def predict():
    # Check if either of the components of the modek are missing
    if vectorizer is None or model is None:
        return jsonify({'error': 'No Model has Been Loaded'}), 500

    data = request.get_json()

    # Check if the request doesn't contain any input
    if not data or 'text' not in data:
        return jsonify({'error': 'Unable to retrieve text'}), 400

    # Load in text from the request into a variable
    input_text = data['text']

    # Vectorize the input text
    X = vectorizer.transform([input_text])

    # Make a prediction utilzing the model
    result = model.predict(X)[0]  # Returns 0 for 'Legit', 1 for "Fake"
    confidence_score = model.predict_proba(X)[0].max()  # Confidence score will be between 0.000 and 1.000

    # Create label which holds the result of the prediction
    label = 'Fake News' if result == 1 else 'Legitimate'

    # Return a JSON response, where float() is used to avoid a JSON serialization error, and round to keep values to 3 decimals
    return jsonify({
        'prediction': label,
        'confidence': round(float(confidence_score), 3)
    }), 200 

# Run the program itself, using port 5001, although its recommended to use 5000 instead if possible, I just couldn't on my machine
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
=======
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