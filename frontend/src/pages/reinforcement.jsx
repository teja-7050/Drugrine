// import React, { useState } from "react";
// import Navbar from "../components/Navbar";
// import axios from "axios";
// import RDKitMolecule from "./RDKitMolecule";
// import "../styles/UseModel.css";

// const reinforcement = () => {
//   const [inputData, setInputData] = useState("");
//   const [result, setResult] = useState(null);
//   const [error, setError] = useState(null);

//   const handlePrediction = async () => {
//     try {
//       setError(null);
//       const response = await axios.post("http://localhost:5000/api/reinforcement", {
//         state: inputData,
//       });
//       console.log(response.data)
//       setResult(response.data);
//     } catch (err) {
//       console.error("Error fetching prediction:", err);
//       setError("Failed to get prediction. Please try again.");
//     }
//   };

//   return (
//     <>
      
//       <div className="model-page-container">
//         <div className="input-section">
//           <h2>Use the DRL Model</h2>
//           <p>Enter SMILES string to let the model predict potential drug molecules.</p>
//           <textarea
//             className="textarea"
//             value={inputData}
//             onChange={(e) => setInputData(e.target.value)}
//             placeholder="Enter SMILES (e.g., CCO)"
//           />
//           <button className="button" onClick={handlePrediction}>
//             Predict Molecule
//           </button>
//           {error && <div className="error">{error}</div>}
//         </div>

//         <div className="results-section">
//           {result && result.top_results && (
//             <>
//               <h3>Top Results</h3>
//               <div className="boxes-container">
//                 {result.top_results.slice(0, 5).map((item, index) => (
//                   <div key={index} className="box">
//                     <div className="left-side">
//                       <RDKitMolecule smiles={item.SMILES} width={200} height={200} />
//                     </div>
//                     <div className="right-side">
//                       <p>
                        
//                         <strong>SMILE:</strong> {item.SMILES}
//                         <strong> PiC50:</strong> {item.Reward  }
//                       </p>
//                     </div>
//                   </div>
//                 ))}
//               </div>
//             </>
//           )}
//         </div>
//       </div>
//     </>
//   );
// };

// export default reinforcement;
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

const Reinforcement = () => {
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
      const response = await axios.post("http://localhost:5000/api/reinforcement", {
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
          <div className="pic50-info">
            <h3>Understanding pIC50</h3>
            <p>
              <strong>What is IC50?</strong> IC50 (Half Maximal Inhibitory Concentration) is a measure of the potency of a substance in inhibiting a specific biological or biochemical function. It represents the concentration of an inhibitor that is required to reduce the biological activity of a target molecule (like an enzyme, receptor, or cell) by 50%.
            </p>
            <p>
              <strong>What is pIC50?</strong> pIC50 is the negative logarithm (base 10) of the IC50 value. It is expressed as:
              <br />
              pIC<sub>50</sub> = -log<sub>10</sub>(IC<sub>50</sub>)
              <br />
              IC50 is typically expressed in molar (M) units.
            </p>
            <h3>Why is pIC50 Important in Drug Discovery?</h3>
            <p>
              pIC50 is a more convenient and intuitive scale for comparing the potency of different compounds:
            </p>
            <ul>
              <li>
                <strong>Higher Value = Higher Potency:</strong> A higher pIC50 value indicates that a lower concentration of the drug is needed to achieve 50% inhibition, meaning the drug is more potent.
              </li>
              <li>
                <strong>Linear Scale:</strong> pIC50 values provide a more linear scale for comparing potencies. For example, a drug with a pIC50 of 7 is 10 times more potent than a drug with a pIC50 of 6.
              </li>
              <li>
                <strong>Easier Comparison:</strong> It simplifies the comparison of drug candidates during the lead optimization phase of drug discovery. Researchers can easily identify and prioritize compounds with higher potency.
              </li>
              <li>
                <strong>Correlation with Binding Affinity:</strong> pIC50 values often correlate with the binding affinity of a drug to its target. Higher binding affinity generally leads to higher potency.
              </li>
            </ul>
            <p>
              In this model, we aim to generate molecules with higher predicted pIC50 values, indicating potentially more effective drug candidates.
            </p>
          </div>

          <div className="input-section">
            <h2>Use the DRL Model for pIC50 Optimization</h2>
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
              <h3>Top Predicted Molecules (Based on pIC50)</h3>
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
                        <strong>pIC50:</strong> {parseFloat(item.Reward).toFixed(2)}
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

export default Reinforcement;