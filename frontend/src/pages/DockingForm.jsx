import React, { useState, useEffect, useRef } from "react";
import "../styles/DockingForm.css"; // Assuming you have the CSS file
import dockingLogo from "../images/docking_logo.png"; // Import your image

function DockingForm() {
  const [ecNumber, setEcNumber] = useState("");
  const [ligandId, setLigandId] = useState("");
  const [predictionResultHTML, setPredictionResultHTML] = useState("");
  const [energyTable, setEnergyTable] = useState([]); // Store the Energy Table DataFrame as an array
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const outputDivRef = useRef(null);

  useEffect(() => {
    if (outputDivRef.current && predictionResultHTML) {
      outputDivRef.current.innerHTML = predictionResultHTML;
      // Manually execute the script tags within the loaded HTML
      const scripts = outputDivRef.current.getElementsByTagName("script");
      Array.from(scripts).forEach((script) => {
        const scriptTag = document.createElement("script");
        Array.from(script.attributes).forEach((attr) => {
          scriptTag.setAttribute(attr.name, attr.value);
        });
        scriptTag.textContent = script.textContent;
        // To safely re-execute scripts, remove the old one and append a new one
        if (script.parentNode) {
          script.parentNode.removeChild(script); // Remove existing script tag
        }
        outputDivRef.current.appendChild(scriptTag); // Append new script tag
      });
    }
  }, [predictionResultHTML]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setPredictionResultHTML("");
    setEnergyTable([]); // Reset the energy table on new submission

    try {
      const response = await fetch("http://localhost:5000/api/docking", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ ec_number: ecNumber, ligand_id: ligandId }),
      });

      if (!response.ok) {
        const errorMessage = await response.json();
        throw new Error(
          `HTTP error! status: ${response.status}, message: ${
            errorMessage.error || "Failed to fetch prediction"
          }`
        );
      }

      const data = await response.json(); // Expecting JSON with visualization_html and energy_table
      setPredictionResultHTML(data.visualization_html);
      setEnergyTable(data.energy_table || []); // Store the energy table
    } catch (err) {
      console.error("Error fetching prediction:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="docking-page-wrapper">
      <div className="top-section">
        {/* The image tag uses the imported dockingLogo */}
        <img src={dockingLogo} alt="Docking Logo" className="docking-logo" />
        <h2>Docking Prediction</h2>
      </div>

      <div
        className={`main-content-wrapper ${
          predictionResultHTML ? "has-results" : ""
        }`}
      >
        {/* Prediction Output (3D Visualization) - Appears on the left after prediction */}
        {predictionResultHTML && (
          <div className="prediction-output" ref={outputDivRef}>
            <h3>Prediction Result:</h3>
            {/* The HTML will be inserted here by the useEffect hook */}
          </div>
        )}

        {/* Docking Form - Always visible, centered */}
        <form onSubmit={handleSubmit} className="docking-form">
          <div className="form-group">
            <label htmlFor="ecNumber">EC Number:</label>
            <input
              type="text"
              id="ecNumber"
              value={ecNumber}
              onChange={(e) => setEcNumber(e.target.value)}
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="ligandId">Ligand ID:</label>
            <input
              type="text"
              id="ligandId"
              value={ligandId}
              onChange={(e) => setLigandId(e.target.value)}
              required
            />
          </div>
          <button type="submit" disabled={loading}>
            {loading ? "Predicting..." : "Predict Docking"}
          </button>
          {error && <p className="error-message">{error}</p>}
        </form>

        {/* Energy Table - Appears on the right after prediction */}
        {energyTable.length > 0 && (
          <div className="energy-table-container">
            <h3>Energy Scores:</h3>
            <table className="energy-table">
              <thead>
                <tr>
                  {Object.keys(energyTable[0]).map((key) => (
                    <th key={key}>{key}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {energyTable.map((row, index) => (
                  <tr key={index}>
                    {Object.values(row).map((value, i) => (
                      <td key={i}>{value}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default DockingForm;
