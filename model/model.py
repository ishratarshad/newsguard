import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

df1 = pd.read_csv("datasets/BuzzFeed_fake_news_content.csv")
df2 = pd.read_csv("datasets/FakeNewsNet.csv")
df3 = pd.read_csv("datasets/news_sample.csv")

df = pd.concat([df1, df2, df3,], ignore_index=True)

text_column = "text"

if text_column in df.columns:
    corpus = df[text_column].dropna().tolist()
else:
     raise ValueError(f"Column '{text_column}' not found in datasets!")

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
Transform = vectorizer.fit_transform(corpus)


joblib.dump(vectorizer, "model/vectorizer.pkl")
joblib.dump(Transform, "model/features.pkl")

print(f"TFIDFG saved to succesfully")