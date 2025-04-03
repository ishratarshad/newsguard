import pickle
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import json


with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


with open('model/features.pkl', 'rb') as f:
    features = pickle.load(f)


data_path = Path('datasets/cleaned_data_fixed.csv')
if data_path.exists():
    df = pd.read_csv(data_path)
    X = df['text']  
    labels = df['label']  
else:
    raise FileNotFoundError("Cleaned data file not found")


X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)


X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

explainer = LimeTextExplainer(class_names=['fake', 'real'])

sample_idx = 0
sample_text = df.iloc[sample_idx]['text']

def predict_fn(texts):
    text_features = vectorizer.transform(texts)
    return model.predict_proba(text_features)

explanation = explainer.explain_instance(
    sample_text,
    predict_fn,
    num_features=10,
    top_labels=1
)


exp = explanation.as_list(0)  


plt.figure(figsize=(10, 6))
plt.barh([x[0] for x in exp], [x[1] for x in exp])
plt.title(f"LIME Explanation for {['fake', 'real'][explanation.top_labels[0]]}")
plt.xlabel('Weight')
plt.tight_layout()
plt.savefig('explanation_visualization.png', bbox_inches='tight')

explanation_data = {
    'text': sample_text,
    'prediction': explanation.top_labels[0],
    'explanation': exp,
    'class_names': ['fake', 'real']
}

import os
os.makedirs('docs', exist_ok=True)

with open('docs/lime_sample_output.json', 'w') as f:
    json.dump(explanation_data, f, indent=4)

print("Explanation saved successfully!")

explanation.show_in_notebook(text=True)
