import React, { useState } from "react";
import Navbar from "../components/Navbar";
import axios from "axios";
import RDKitMolecule from "./RDKitMolecule";
import "../styles/UseModel.css";

const predefinedSmiles = [
  { name: "Aspirin", smiles: "CC(=O)OC1=CC=CC=C1C(=O)O" },
  { name: "Caffeine", smiles: "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" },
  { name: "Ibuprofen", smiles: "CC(C)Cc1ccc(cc1)C(C)C(=O)O" },
  { name: "Paracetamol", smiles: "CC(=O)Nc1ccc(cc1)O" },
  { name: "Ethanol", smiles: "CCO" },
];

const LogP = () => {
  const [selectedSmiles, setSelectedSmiles] = useState("");
  const [customSmiles, setCustomSmiles] = useState("");
  const [usePredefined, setUsePredefined] = useState(true);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handlePredefinedChange = (event) => {
    setSelectedSmiles(event.target.value);
    setCustomSmiles(""); // Clear custom input when a predefined option is selected
  };

  const handleCustomChange = (event) => {
    setCustomSmiles(event.target.value);
    setSelectedSmiles(""); // Clear predefined selection when custom input changes
    setUsePredefined(false);
  };

  const handleToggleInput = (event) => {
    setUsePredefined(event.target.value === "predefined");
    setCustomSmiles("");
    setSelectedSmiles("");
  };

  const getCurrentInput = () => {
    return usePredefined ? selectedSmiles : customSmiles;
  };

  const handlePrediction = async () => {
    const currentInput = getCurrentInput();
    if (!currentInput) {
      setError("Please select a predefined SMILES or enter a custom one.");
      setResult(null);
      return;
    }
    try {
      setError(null);
      const response = await axios.post("http://localhost:5000/api/reinforcement/logp", {
        state: currentInput,
      });
      console.log(response.data);
      setResult(response.data);
    } catch (err) {
      console.error("Error fetching prediction:", err);
      setError("Failed to get prediction. Please try again.");
    }
  };

  return (
    <>
      
      <div className="model-page-container">
        <div className="info-and-input-section">
          <div className="logp-info">
            <h3>Understanding LogP</h3>
            <p>
              <strong>What is LogP?</strong> LogP (Octanol-Water Partition Coefficient) is a measure of the lipophilicity (fat-loving) of a chemical compound. It quantifies the differential solubility of a compound between two immiscible phases, octanol (representing a lipid environment) and water (representing an aqueous environment).
            </p>
            <p>
              Mathematically, LogP = log<sub>10</sub> ([Solute]<sub>octanol</sub> / [Solute]<sub>water</sub>)
            </p>
            <p>
              <strong>Why is LogP Important in Drug Discovery?</strong>
              <br />
              LogP is a crucial physicochemical property in drug discovery for several reasons:
              <ul>
                <li>... (rest of the LogP explanation) ...</li>
              </ul>
              In this model, we aim to generate molecules with predicted LogP values that fall within a desirable range for drug-like properties, balancing lipophilicity and hydrophilicity.
            </p>
          </div>

          <div className="input-section">
            <h2>Use the DRL Model for LogP Optimization</h2>
            <p>Select a predefined SMILES or enter your own.</p>

            <div className="input-toggle">
              <label>
                <input
                  type="radio"
                  value="predefined"
                  checked={usePredefined}
                  onChange={handleToggleInput}
                />
                Predefined SMILES
              </label>
              <label>
                <input
                  type="radio"
                  value="custom"
                  checked={!usePredefined}
                  onChange={handleToggleInput}
                />
                Custom SMILES
              </label>
            </div>

            {usePredefined ? (
              <select
                className="select-smiles"
                value={selectedSmiles}
                onChange={handlePredefinedChange}
              >
                <option value="">Select a SMILES</option>
                {predefinedSmiles.map((item, index) => (
                  <option key={index} value={item.smiles}>
                    {item.name} ({item.smiles})
                  </option>
                ))}
              </select>
            ) : (
              <textarea
                className="textarea"
                value={customSmiles}
                onChange={handleCustomChange}
                placeholder="Enter SMILES (e.g., CCO)"
              />
            )}

            <button className="button" onClick={handlePrediction}>
              Predict Molecule
            </button>
            {error && <div className="error">{error}</div>}
          </div>
        </div>

        <div className="results-section">
          {result && result.top_results && (
            <>
              <h3>Top Predicted Molecules (Based on LogP)</h3>
              <div className="boxes-container">
                {result.top_results.slice(0, 5).map((item, index) => (
                  <div key={index} className="box">
                    <div className="left-side">
                      <RDKitMolecule smiles={item.SMILES} width={200} height={200} />
                    </div>
                    <div className="right-side">
                      <p>
                        <strong>SMILE:</strong> {item.SMILES}
                        <br />
                        <strong>LogP:</strong> {parseFloat(item.Reward).toFixed(2)}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
};

export default LogP;