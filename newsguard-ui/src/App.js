import React, { useState } from 'react';
import './App.css';

function App() {
  const [articleText, setArticleText] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleCheckArticle = async () => {
    if (!articleText.trim()) return;

    setLoading(true);
    try {
      const res = await fetch('http://127.0.0.1:5000/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: articleText }),
      });

      const data = await res.json();
      console.log("API response:", data);
      setResponse(data);
    } catch (error) {
      console.error('Error:', error);
      setResponse({ error: 'Could not connect to the server.' });
    }
    setLoading(false);
  };

  const highlightText = () => {
    if (!response?.highlight_data) return articleText;

    let highlighted = articleText;
    response.highlight_data
      .sort((a, b) => b.word.length - a.word.length)
      .forEach(({ word, weight }) => {
        const color =
          weight > 0.1
            ? '#ff3c3c' // Red = suspicious
            : weight < -0.1
            ? '#3399ff' // Blue = trusted
            : '#eeeeee'; // Neutral

        const regex = new RegExp(`\\b(${word})\\b`, 'gi');
        highlighted = highlighted.replace(
          regex,
          `<mark title="Weight: ${weight.toFixed(4)}" style="background-color:${color}; padding: 0 2px; border-radius: 3px;">$1</mark>`
        );
      });
    return highlighted;
  };

  const calculateMisinformationScore = () => {
    if (!response?.highlight_data) return 0;
    const total = response.highlight_data.length;
    const misinfoWords = response.highlight_data.filter(w => w.weight > 0.1);
    return ((misinfoWords.length / total) * 100).toFixed(2);
  };

  const getMisinformationAssessment = () => {
    if (!response?.highlight_data || typeof response.confidence !== 'number') return '';

    const misinfoScore = parseFloat(calculateMisinformationScore());
    const confidence = response.confidence * 100;

    if (misinfoScore <= 0) return 'Most Likely NOT Misinformation';
    if (misinfoScore < confidence) return 'Likely NOT Misinformation';
    return 'Highly Likely Misinformation';
  };

  return (
    <div className="App">
      <h1>🛡️ NewsGuard</h1>

      <textarea
        placeholder="Paste article text here..."
        rows={10}
        value={articleText}
        onChange={(e) => setArticleText(e.target.value)}
      />

      <div className="button-group">
        <button onClick={handleCheckArticle} disabled={loading}>
          {loading ? 'Checking...' : 'Check Article'}
        </button>
        <button
          onClick={() => {
            setArticleText('');
            setResponse(null);
          }}
          className="clear-btn"
        >
          Clear Article
        </button>
      </div>

      {response && (
        <div className="result">
          <h2>Detecting {response.prediction}</h2>
          {typeof response.confidence === 'number' && !isNaN(response.confidence) && (
            <p>Confidence Score: <strong>{(response.confidence * 100).toFixed(2)}%</strong></p>
          )}
          {response.highlight_data && (
            <>
              <p><strong>Misinformation Score:</strong> {calculateMisinformationScore()}%</p>
              <p><strong>Misinformation Assessment:</strong> {getMisinformationAssessment()}</p>
            </>
          )}

          <h3>Highlighted Article:</h3>
          <div
            className="highlighted-text"
            dangerouslySetInnerHTML={{ __html: highlightText() }}
          />

          <div className="legend">
            <p>
              <span className="legend-box red" /> Likely Misinformation &nbsp;&nbsp;
              <span className="legend-box blue" /> Likely Trustworthy &nbsp;&nbsp;
              <span className="legend-box gray" /> Neutral
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
