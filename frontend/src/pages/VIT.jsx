import React, { useState } from 'react';
import '../styles/VIT.css'; // Import the CSS file
import axios from "axios";

function VIT() {
  const [smiles, setSmiles] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
        e.preventDefault();
        if (!smiles.trim()) {
          setError('Please enter a SMILES string');
          return;
        }
    
        setLoading(true);
        setError(null);
    
        try {
          const response = await axios.post("http://localhost:5000/api/vit", {
            state: smiles,
          });
    
          // Check if the request was successful (status code in the 2xx range)
          if (response.status >= 200 && response.status < 300) {
            const data = response.data; // Access the JSON data using response.data
            setPrediction(data);
          } else {
            // Handle error responses
            const errorData = response.data;
            throw new Error(errorData.detail || `Prediction failed with status ${response.status}`);
          }
        } catch (err) {
          setError(err.message);
        } finally {
          setLoading(false);
        }
      };

  return (
    <div className="app">
      <header className="header">
        <h1>Molecular Property Classifier</h1>
        <p>Using Vision Transformer (ViT) Model</p>
      </header>

      <main className="main-content">
        <form onSubmit={handleSubmit} className="prediction-form">
          <div className="form-group">
            <label htmlFor="smiles">SMILES String:</label>
            <input
              type="text"
              id="smiles"
              value={smiles}
              onChange={(e) => setSmiles(e.target.value)}
              placeholder="e.g., CCO for ethanol"
              required
            />
          </div>
          <button type="submit" disabled={loading}>
            {loading ? 'Predicting...' : 'Predict'}
          </button>
        </form>

        {error && (
          <div className="error-message">
            <p>Error: {error}</p>
          </div>
        )}

        {prediction && !error && (
          <div className="prediction-result">
            <h2>Prediction Result</h2>
            <p><strong>SMILES:</strong> {prediction.smiles}</p>
            <p><strong>Predicted Class:</strong> {prediction.predicted_class}</p>
          </div>
        )}
      </main>

     
    </div>
  );
}

export default VIT;