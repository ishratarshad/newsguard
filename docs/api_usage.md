# NewsGuard API – Usage Guide

### The NewsGuard API helps classify whether a news article is *Fake News* or *Legitimate* using a machine learning model trained on TF-IDF features and Naive Bayes. It also offers explainability via LIME.

This guide provides an overview of how to use the NewsGuard API by interacting with the `/predict` endpoint.

---

## Endpoints

**POST** `/predict`

Accepts a JSON request containing a text article and returns a prediction on whether it is "Fake News" or "Legitimate", along with a confidence score.

**POST** `/explain`

Accepts a JSON request containing a text article and returns a prediction and confidence score, alongside a copy of the text that was sent to it, and highlighted data that may be used for other purposes.

---

## Base URL

Local development server:  
`http://127.0.0.1:5000`

---

## Request Format

- **URL**: `http://127.0.0.1:5000/predict` or `http://127.0.0.1:5000/explain`
- **Method**: `POST`
- **Headers**:  `Content-Type: application/json`
- **Body of Request**

*Example Input for `/predict` and `/explain` :*
```json
{
  "text": "Government passes new healthcare policy today."
}
```

## Response Format

- **Prediction**: A string, either `Fake News` or `Legitimate`
- **Confidence**: A float number between `0 - 1` representing model connfidence

*Example Output for `/predict` :*
```json
{
  "confidence": "0.8675",
  "prediction": "Legitimate"
}
```

*Example Output for `/predict` :*
```json
{
  "confidence": 0.8675,
  "highlight_data": [
    {
      "weight": -0.0731,
      "word": "Government"
    },
    {
      "weight": 0.0583,
      "word": "passes"
    },
    {
      "weight": -0.03,
      "word": "policy"
    },
    {
      "weight": -0.02,
      "word": "healthcare"
    },
    {
      "weight": -0.0118,
      "word": "new"
    },
    {
      "weight": 0.0038,
      "word": "today"
    }
  ],
  "original_text": "Government passes new healthcare policy today.",
  "prediction": "Fake News"
}
```

## Example using `curl`

```python
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists confirm water found on Mars."}'
```