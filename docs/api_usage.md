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
