import React, { useState } from "react";
import Navbar from "../components/Navbar";
import axios from "axios";
import RDKitMolecule from "./RDKitMolecule";
import "../styles/UseModel.css";

const Protein2Smiles = () => {
  const [inputData, setInputData] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handlePrediction = async () => {
    try {
      setError(null);
      const response = await axios.post("http://localhost:5000/api/protein2smiles", {
        state: inputData,
      });
      setResult(response.data);
    } catch (err) {
      console.error("Error fetching prediction:", err);
      setError("Failed to get prediction. Please try again.");
    }
  };

  return (
    <>
      
      <div className="model-page-container">
        <div className="input-section">
          <h2>PROTEIN2SMILES</h2>
          <p>Enter PROTEIN sequence to let the model predict it's SMILES sequence .</p>
          <textarea
            className="textarea"
            value={inputData}
            onChange={(e) => setInputData(e.target.value)}
            placeholder="Enter PROTEIN (e.g., MM)"
          />
          <button className="button" onClick={handlePrediction}>
            Predict Smiles
          </button>
          {error && <div className="error">{error}</div>}
        </div>

        <div className="results-section">
          {result && result.top_results && Array.isArray(result.top_results) && (
            <>
              <h3>Top Results</h3>
              <div className="boxes-container">
                {result.top_results.slice(0, 5).map((item, index) => (
                  <div key={index} className="box">
                    <div className="left-side">
                      <RDKitMolecule smiles={item.SMILES} width={200} height={200} />
                    </div>
                    <div className="right-side">
                      <p>
                        <strong>SMILE:</strong> {item.SMILES}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}

          {result && result.top_results && !Array.isArray(result.top_results) && (
            <>
              <h3>Prediction Result</h3>
              <div className="box">
                <div className="left-side">
                  <RDKitMolecule smiles={result.top_results.SMILES} width={200} height={200} />
                </div>
                <div className="right-side">
                  <p>
                    <strong>SMILE:</strong> {result.top_results.SMILES}
                  </p>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
};

export default Protein2Smiles;