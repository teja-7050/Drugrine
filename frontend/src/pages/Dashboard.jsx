// import React, { useState, useEffect } from 'react';
// import { useNavigate } from 'react-router-dom';
// import '../styles/Dashboard.css';
// import { FaFlask, FaHeartbeat, FaAtom, FaCode, FaLightbulb, FaChartBar } from 'react-icons/fa';

// const Dashboard = () => {
//     const navigate = useNavigate();
//     const [welcomeText, setWelcomeText] = useState('');
//     const fullWelcomeText = 'DRUGRIN.';
//     const typingSpeed = 150;
//     const [showCursor, setShowCursor] = useState(true);

//     useEffect(() => {
//         let charIndex = 0;
//         const typingInterval = setInterval(() => {
//             if (charIndex < fullWelcomeText.length) {
//                 setWelcomeText(fullWelcomeText.substring(0, charIndex + 1));
//                 charIndex++;
//             } else {
//                 clearInterval(typingInterval);
//                 setTimeout(() => setShowCursor(false), 500);
//             }
//         }, typingSpeed);

//         return () => clearInterval(typingInterval);
//     }, []);

//     return (
//         <div className="dashboard-container">
//             <header className="dashboard-header">
//                 <h1>Welcome to {welcomeText}{showCursor && <span className="cursor"></span>}</h1>
//                 <p>Your central point for exploring cutting-edge AI tools in pharmaceutical research.</p>
//             </header>

//             <main className="dashboard-main">
//                 <section className="feature-card" onClick={() => navigate('/reinforcement')}>
//                     <div className="feature-icon rl-icon"><FaFlask size={30} color="#fff" /></div>
//                     <h3>AI-Powered Drug Design</h3>
//                     <p>Generate novel drug candidates and optimize their properties using advanced Reinforcement Learning algorithms.</p>
//                     <button className="explore-button">Explore Drug Design</button>
//                 </section>

//                 <section className="feature-card" onClick={() => navigate('/diagnosis')}>
//                     <div className="feature-icon diagnosis-icon"><FaHeartbeat size={30} color="#fff" /></div>
//                     <h3>AI-Enhanced Medical Diagnosis</h3>
//                     <p>Analyze medical images and patient data with cutting-edge AI for rapid and accurate diagnoses.</p>
//                     <button className="explore-button">Explore Diagnosis Tools</button>
//                 </section>

//                 <section className="feature-card" onClick={() => navigate('/protein2smiles')}>
//                     <div className="feature-icon protein2smiles-icon"><FaCode size={30} color="#fff" /></div>
//                     <h3>Protein to SMILES Prediction</h3>
//                     <p>Utilize AI models to predict the SMILES representation of a molecule based on its protein sequence input.</p>
//                     <button className="explore-button">Predict from Protein</button>
//                 </section>

//                 <section className="feature-card" onClick={() => navigate('/docking')}>
//                     <div className="feature-icon model-icon"><FaAtom size={30} color="#fff" /></div>
//                     <h3>Molecular Docking</h3>
//                     <p>Simulate the interaction between small molecules and protein structures to predict binding affinities.</p>
//                     <button className="explore-button">Explore Docking</button>
//                 </section>

//                 <section className="feature-card" onClick={() => navigate('/masking')}>
//                     <div className="feature-icon research-icon"><FaLightbulb size={30} color="#fff" /></div>
//                     <h3>Chembert Analysis</h3>
//                     <p>Explore the chemical language model for various downstream tasks, including masked molecule prediction.</p>
//                     <button className="explore-button">Explore ChemBert</button>
//                 </section>

//                 <section className="feature-card" onClick={() => navigate('/protein')}>
//                     <div className="feature-icon analytics-icon"><FaChartBar size={30} color="#fff" /></div>
//                     <h3>Protein Structure Visualization</h3>
//                     <p>Visualize and analyze 3D protein structures to understand their properties and interactions.</p>
//                     <button className="explore-button">View Protein Structures</button>
//                 </section>
//                 <section className="feature-card" onClick={() => navigate('/papaya')}>
//                 <div className="feature-icon" style={{ backgroundColor: '#f48fb1' }}>
//                     <span role="img" aria-label="viewer" style={{ fontSize: '30px', color: '#fff' }}>🧠</span>
//                 </div>
//                 <h3>Medical Image Viewer</h3>
//                 <p>Visualize medical images like MRI, CT scans directly in your browser using Papaya Viewer.</p>
//                 <button className="explore-button">Open Viewer</button>
//                 </section>

//             </main>
//         </div>
//     );
// };

// export default Dashboard;

import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/Dashboard.css";
import {
  FaFlask,
  FaHeartbeat,
  FaAtom,
  FaCode,
  FaLightbulb,
  FaChartBar,
} from "react-icons/fa";

const Dashboard = () => {
  const navigate = useNavigate();
  const [welcomeText, setWelcomeText] = useState("");
  const fullWelcomeText = "DRUGRIN.";
  const typingSpeed = 150;
  const [showCursor, setShowCursor] = useState(true);

  useEffect(() => {
    let charIndex = 0;
    const typingInterval = setInterval(() => {
      if (charIndex < fullWelcomeText.length) {
        setWelcomeText(fullWelcomeText.substring(0, charIndex + 1));
        charIndex++;
      } else {
        clearInterval(typingInterval);
        setTimeout(() => setShowCursor(false), 500);
      }
    }, typingSpeed);

    return () => clearInterval(typingInterval);
  }, []);

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <h1>
          Welcome to {welcomeText}
          {showCursor && <span className="cursor"></span>}
        </h1>
        <p>
          Your central point for exploring cutting-edge AI tools in
          pharmaceutical research.
        </p>
      </header>

      <main className="dashboard-main">
        <section
          className="feature-card"
          onClick={() => navigate("/reinforcement")}
        >
          <div className="feature-icon rl-icon">
            <FaFlask size={30} color="#fff" />
          </div>
          <h3>AI-Powered Drug Design</h3>
          <p>
            Generate novel drug candidates and optimize their properties using
            advanced Reinforcement Learning algorithms.
          </p>
          <button className="explore-button">Explore Drug Design</button>
        </section>

        <section
          className="feature-card"
          onClick={() => navigate("/diagnosis")}
        >
          <div className="feature-icon diagnosis-icon">
            <FaHeartbeat size={30} color="#fff" />
          </div>
          <h3>AI-Enhanced Medical Diagnosis</h3>
          <p>
            Analyze medical images and patient data with cutting-edge AI for
            rapid and accurate diagnoses.
          </p>
          <button className="explore-button">Explore Diagnosis Tools</button>
        </section>

        <section
          className="feature-card"
          onClick={() => navigate("/protein2smiles")}
        >
          <div className="feature-icon protein2smiles-icon">
            <FaCode size={30} color="#fff" />
          </div>
          <h3>Protein to SMILES Prediction</h3>
          <p>
            Utilize AI models to predict the SMILES representation of a molecule
            based on its protein sequence input.
          </p>
          <button className="explore-button">Predict from Protein</button>
        </section>

        <section className="feature-card" onClick={() => navigate("/docking")}>
          <div className="feature-icon model-icon">
            <FaAtom size={30} color="#fff" />
          </div>
          <h3>Molecular Docking</h3>
          <p>
            Simulate the interaction between small molecules and protein
            structures to predict binding affinities.
          </p>
          <button className="explore-button">Explore Docking</button>
        </section>

        <section className="feature-card" onClick={() => navigate("/masking")}>
          <div className="feature-icon research-icon">
            <FaLightbulb size={30} color="#fff" />
          </div>
          <h3>Chembert Analysis</h3>
          <p>
            Explore the chemical language model for various downstream tasks,
            including masked molecule prediction.
          </p>
          <button className="explore-button">Explore ChemBert</button>
        </section>

        <section className="feature-card" onClick={() => navigate("/protein")}>
          <div className="feature-icon analytics-icon">
            <FaChartBar size={30} color="#fff" />
          </div>
          <h3>Protein Structure Visualization</h3>
          <p>
            Visualize and analyze 3D protein structures to understand their
            properties and interactions.
          </p>
          <button className="explore-button">View Protein Structures</button>
        </section>

        <section className="feature-card" onClick={() => navigate("/papaya")}>
          <div className="feature-icon" style={{ backgroundColor: "#f48fb1" }}>
            <span
              role="img"
              aria-label="viewer"
              style={{ fontSize: "30px", color: "#fff" }}
            >
              🧠
            </span>
          </div>
          <h3>Active OR Inactive Ligand Identification</h3>
          <p>
            Identify Active or Inactive ligand for Alzheimer disease using Gen
            AI vit technology
          </p>
          <button className="explore-button">Open Viewer</button>
        </section>

        {/* 🔥 New 3D Diagnosis Section */}
        <section
          className="feature-card"
          onClick={() => navigate("/visualization")}
        >
          <div className="feature-icon" style={{ backgroundColor: "#81d4fa" }}>
            <span
              role="img"
              aria-label="3d-diagnosis"
              style={{ fontSize: "30px", color: "#fff" }}
            >
              🧬
            </span>
          </div>
          <h3>3D Diagnosis</h3>
          <p>
            Experience immersive 3D visualization of diagnostic medical data for
            deeper analysis.
          </p>
          <button className="explore-button">Start 3D Diagnosis</button>
        </section>
        {/* 🔥 End of new section */}
      </main>
    </div>
  );
};

export default Dashboard;
