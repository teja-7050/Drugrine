import React, { useEffect, useRef, useState } from "react";
import "../styles/RDKitMolecule.css";

const RDKitMolecule = ({ smiles, width = 200, height = 200 }) => {
  const [rdkitLoaded, setRdkitLoaded] = useState(false);
  const canvasRef = useRef(null);

  useEffect(() => {
    // Only load RDKit once if it's not already on the window
    if (!window.RDKit) {
      const script = document.createElement("script");
      // Use the official @rdkit/rdkit package on unpkg (pinned version 2023.09.2)
      script.src =
        "https://cdn.jsdelivr.net/npm/@rdkit/rdkit@latest/Code/MinimalLib/dist/RDKit_minimal.js";
      script.async = true;

      script.onload = async () => {
        try {
          // Initialize RDKit, telling it where to find the .wasm file
          window.RDKit = await window.initRDKitModule({
            locateFile: (file) =>
              `https://cdn.jsdelivr.net/npm/@rdkit/rdkit@latest/Code/MinimalLib/dist/${file}`,
          });
          setRdkitLoaded(true);
        } catch (error) {
          console.error("Error initializing RDKit:", error);
        }
      };

      document.body.appendChild(script);
    } else {
      // If already loaded, just set the flag
      setRdkitLoaded(true);
    }
  }, []);

  useEffect(() => {
    if (rdkitLoaded && canvasRef.current) {
      try {
        const mol = window.RDKit.get_mol(smiles);
        console.log(smiles);
        const svg = mol.get_svg(width, height);

        canvasRef.current.innerHTML = svg;
        mol.delete();
      } catch (err) {
        console.error("Error generating molecule:", err);
        canvasRef.current.innerHTML = `<div style="color:red;">Invalid SMILES</div>`;
      }
    }
  }, [rdkitLoaded, smiles, width, height]);

  return (
    <div
      ref={canvasRef}
      className="rdkit-canvas"
      style={{ "--canvas-width": `${width}px`, "--canvas-height": `${height}px` }}
    />
  );
};

export default RDKitMolecule;
