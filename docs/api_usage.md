# NewsGuard API – Usage Guide

### This guide provides an overview of how to use the NewsGuard API by interacting with the `/predict` endpoint.

---

## Endpoint

**POST** `/predict`

Accepts a JSON request containing a text article and returns a prediction on whether it is "Fake News" or "Legitimate", along with a confidence score.

---

## Request Format

- **URL**: `http://localhost:5001/predict`
- **Method**: `POST`
- **Headers**:  `Content-Type: application/json`
- **Body of Request**

*Example Input:*
```json
{
  "text": "Government passes new healthcare policy today."
}
```

## Response Format

- **Prediction**: A string, either `Fake News` or `Legitimate`
- **Confidence**: A float number between `0 - 1` representing model connfidence

*Example Output:*
```json
{
  "confidence": "0.885",
  "prediction": "Legitimate"
}
```

## Example using `curl`

```python
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists confirm water found on Mars."}'
```
=======
# NewsGuard API

The NewsGuard API helps classify whether a news article is *Fake News* or *Legitimate* using a machine learning model trained on TF-IDF features and Naive Bayes. It also offers explainability via LIME.

---

## Base URL

Local development server:  
`http://127.0.0.1:5000`

---

## Endpoints

### 1. `/predict`

Predicts whether an article is fake or legitimate.

**Method:** `POST`  
**Endpoint:** `/predict`  
**Content-Type:** `application/json`

#### Request Body

```json
{
  "text": "Shocking miracle cure discovered today!"
}