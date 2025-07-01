# NewsGuard  
**AI-Powered Scam Article Detector using NLP & Machine Learning**

---

## Project Overview  
NewsGuard is a machine learning system built to detect fake or misleading news articles. By leveraging Natural Language Processing (NLP), it classifies text as **Legitimate** or **Fake News**, providing a confidence score and an explanation using LIME.

---

## Folder Structure  
```
newsguard/
│
├── datasets/           # Raw and cleaned data files
├── analysis/           # EDA reports and data visualizations
├── docs/               # Documentation, test results, and API usage
├── scripts/            # Python scripts (training, testing, LIME, etc.)
├── model/              # Saved models, vectorizers, and features
├── api/                # Flask API implementation
├── ui/                 # (Optional) React front-end for user interaction
│
├── README.md           # Project overview and instructions
├── requirements.txt    # Required Python libraries
├── .gitignore          # Files/directories to ignore in version control
```

---

## Features  
- **Fake News Classifier** using Naive Bayes & TF-IDF  
- **REST API** with `/predict` and `/explain` routes  
- **LIME Explainability** for interpretable results  
- Confidence score for predictions  
- Clean documentation and test logs  
- Tested with Postman + documented test outputs

---

## Tech Stack  
- **Python** (NLP & API)  
- **Scikit-learn**, **LIME**, **Pandas**  
- **Flask** (Backend API)  
- **Postman** (API testing)  



