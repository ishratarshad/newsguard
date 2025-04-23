
```markdown
# /explain Route Testing Notes

### Test Case: Legitimate Article

**Input JSON:**
```json
{
  "text": "Shocking miracle cure discovered today!"
}
```

**Response:**
```json
{
  "prediction": "Legitimate",
  "explanation": [
    ["miracle", 0.3144],
    ["today", -0.0301],
    ["Shocking", -0.0224],
    ["discovered", 0.0139],
    ["cure", 0.0028]
  ]
}
```

**Status:** Passed — Explanation returned top 5 influential words with weights, and prediction format matched expectations.

---

### Test Case: Missing "text" Field

**Input JSON:**
```json
{}
```

**Response:**
```json
{
  "error": "Missing 'text' field"
}
```

**Status:**  Passed — Proper error handling for missing required fields.
```

Now paste that into `docs/explain_test_notes.md`, save it, then run:

```bash
git add docs/explain_test_notes.md
git commit -m "Added test notes for /explain route"
git push origin main
```

