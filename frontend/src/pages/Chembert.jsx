import React, { useState } from 'react';
import axios from 'axios';
import '../styles/Chembert.css'; // Import the CSS file

const Chembert = () => {
  const [selectedDisease, setSelectedDisease] = useState('');
  const [showInfo, setShowInfo] = useState(false);
  const [smiles, setSmiles] = useState('');
  const [prediction, setPrediction] = useState([]);
  const [error, setError] = useState(null);

  const diseaseInfo = {
    alzheimers: {
      name: 'Alzheimer’s Disease',
      description:
        'Alzheimer’s is a progressive neurological disorder that causes brain cells to waste away. One therapeutic strategy involves targeting JAK2, a tyrosine kinase involved in neuroinflammation and cell survival. Predicting suitable inhibitors using SMILES with masked tokens helps explore novel compounds for this target.',
    },
  };

  const handleDiseaseSelect = (e) => {
    setSelectedDisease(e.target.value);
    setShowInfo(true);
    setPrediction([]);
    setSmiles('');
    setError(null);
  };

  const handlePredict = async () => {
    if (!smiles.includes('<mask>')) {
      setError('SMILES must contain <mask>');
      return;
    }

    setError(null);
    try {
      const res = await axios.post('http://localhost:5000/api/mask', {state:  smiles });
      setPrediction(res.data.top_results.split('\n'));
    } catch (err) {
      setError('Prediction failed');
    }
  };

  return (
    <div className="container">
      {/* Animation keyframes are now in PredictPage.css */}

      <h2 className="heading">Masked SMILES Predictor</h2>

      <div className="centerBox">
        <div className="contentBox">
          <label className="label">Select Disease</label>
          <select className="dropdown" onChange={handleDiseaseSelect} value={selectedDisease}>
            <option value="">-- Select --</option>
            <option value="alzheimers">Alzheimer’s</option>
          </select>

          {showInfo && selectedDisease && (
            <div className="infoBox">
              <h3 className="infoTitle">{diseaseInfo[selectedDisease].name}</h3>
              <p className="infoText">{diseaseInfo[selectedDisease].description}</p>
            </div>
          )}

          {selectedDisease && (
            <>
              <label className="label">Enter SMILES (with &lt;mask&gt;)</label>
              <input
                type="text"
                value={smiles}
                onChange={(e) => setSmiles(e.target.value)}
                placeholder="Example: CC<mask>OC"
                className="input"
              />
              <button onClick={handlePredict} className="button">Predict</button>
              {error && <p className="error">{error}</p>}

              {prediction.length > 0 && (
                <div className="resultBox">
                  <h4 className="resultTitle">Predictions:</h4>
                  <ul style={{ paddingLeft: '20px' }}>
                    {prediction.map((p, idx) => (
                      <li
                        key={idx}
                        className="resultItem"
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = '#37474f';
                          e.currentTarget.style.transform = 'scale(1.02)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = '#263238';
                          e.currentTarget.style.transform = 'scale(1)';
                        }}
                      >
                        {p}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default Chembert;