import React, { useState } from 'react';
import ProteinViewer from '../components/ProteinViewer';
import '../styles/ProteinStructure.css';

// Extracted Input Sequence Component
const SequenceInput = ({ sequence, setSequence, onPredict, loading, error }) => (
  <div className="sequence-input-area">
    <label htmlFor="sequenceInput" className="form-label">
      Input sequence
    </label>
    <textarea
      className="form-control"
      id="sequenceInput"
      rows={15}
      value={sequence}
      onChange={(e) => setSequence(e.target.value)}
      placeholder="Enter protein sequence..."
    />
    <button className="predict-button" onClick={onPredict} disabled={loading}>
      {loading ? 'Predicting...' : 'Predict'}
    </button>
    {error && <div className="error-message">{error}</div>}
  </div>
);

function ProteinStructure() {
  const [sequence, setSequence] = useState(
    "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"
  );
  const [pdbData, setPdbData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const calculatePlDDT = (pdbString) => {
    if (!pdbString) return 0;
    const lines = pdbString.split('\n');
    let sum = 0;
    let count = 0;

    lines.forEach((line) => {
      if (line.startsWith('ATOM')) {
        const bFactor = parseFloat(line.substring(60, 66).trim());
        sum += bFactor;
        count++;
      }
    });

    return count > 0 ? (sum / count).toFixed(4) : 0;
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);

    try {
      if (!sequence || sequence.length < 10) {
        throw new Error('Please enter a valid protein sequence (minimum 10 characters)');
      }

      const response = await fetch('http://localhost:5000/api/fold', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ sequence }),
      });

      if (!response.ok) {
        throw new Error('Prediction failed - server error');
      }

      const data = await response.json();
      if (!data.pdb) {
        throw new Error('Invalid PDB data received');
      }
      setPdbData(data.pdb);
    } catch (err) {
      setError(err.message);
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const downloadPdb = () => {
    if (!pdbData) return; // Prevent download if no PDB data
    const element = document.createElement('a');
    const file = new Blob([pdbData], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = 'predicted.pdb';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  return (
    <div className="esmfold-page">
      {!pdbData ? (
        <div className="sidebar">
          <div className="esmfold-branding">
            <h1>ESMFold</h1>
            <p>An end-to-end single sequence protein structure predictor.</p>
          </div>
          <SequenceInput
            sequence={sequence}
            setSequence={setSequence}
            onPredict={handlePredict}
            loading={loading}
            error={error}
          />
        </div>
      ) : (
        <div className="prediction-layout">
          <div className="input-area-wrapper"> {/* New wrapper for input area */}
            <h2>Input Sequence</h2>
            <SequenceInput
              sequence={sequence}
              setSequence={setSequence}
              onPredict={handlePredict}
              loading={loading}
              error={error}
            />
          </div>

          <div className="viewer-plddt-container"> {/* Container for viewer and pLDDT */}
            <div className="viewer-side">
              <h2>Predicted Structure</h2>
              {loading ? (
                <div className="spinner"></div>
              ) : (
                <div className="protein-viewer-container">
                  <ProteinViewer pdbData={pdbData} />
                </div>
              )}
              <button className="reset-button" onClick={() => setPdbData(null)}>
                🔁 Reset View
              </button>
            </div>

            <div className="plddt-side">
              <h2>pLDDT</h2>
              <div className="plddt-box">
                <p>plDDT is a confidence estimate (0-100).</p>
                <div className="plddt-value">plDDT: {calculatePlDDT(pdbData)}</div>
                <button className="download-pdb-button" onClick={downloadPdb}>
                  Download PDB
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ProteinStructure;