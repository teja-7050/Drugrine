import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import LoginPage from './pages/LoginPage';
import SignupPage from './pages/SignupPage';
import Dashboard from './pages/Dashboard';
import DockingForm from './pages/DockingForm';
import OTPVerificationPage from './pages/OTPVerificationPage';
import MobileVerificationPage from './pages/MobileVerificationPage';
import Protein2Smiles from "./pages/Protein2Smiles";
import Chembert from './pages/Chembert';
import Reinforcement from "./pages/reinforcement";
import Diagnosis from "./pages/Diagnosis";
import LogP from "./pages/LogP";
import "./styles/App.css";
import Navbar from './components/Navbar';
import VIT from './pages/VIT';
import PrivateRoute from './components/PrivateRoute';
import ProfilePage from './pages/ProfilePage';
import Loading from './components/Loading';
import ProteinViewer from './pages/ProteinStructure';
import Visualization from './pages/Visualization'; // ✅

const App = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(null);

  useEffect(() => {
    const token = localStorage.getItem('authToken');
    setIsAuthenticated(!!token);
  }, []);

  const handleLoginSuccess = (token) => {
    localStorage.setItem('authToken', token);
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    localStorage.removeItem('authToken');
    localStorage.removeItem('userData');
    setIsAuthenticated(false);
  };

  if (isAuthenticated === null) {
    return <Loading />;
  }

  return (
    <Router>
      <div className="app-container">
        <Routes>
          {/* Public Routes */}
          <Route path="/" element={<LandingPage />} />
          <Route path="/login" element={<LoginPage onLoginSuccess={handleLoginSuccess} />} />
          <Route path="/signup" element={<SignupPage />} />
          <Route path="/verify-otp" element={<OTPVerificationPage />} />
          <Route path="/verify-mobile" element={<MobileVerificationPage />} />

          {/* Protected Routes */}
          <Route path="/" element={<PrivateRoute isAuthenticated={isAuthenticated} />}>
            <Route path="/dashboard" element={<><Navbar isAuthenticated={isAuthenticated} onLogout={handleLogout} /><Dashboard /></>} />
            <Route path="/protein2smiles" element={<><Navbar isAuthenticated={isAuthenticated} onLogout={handleLogout} /><Protein2Smiles /></>} />
            <Route path="/reinforcement" element={<><Navbar isAuthenticated={isAuthenticated} onLogout={handleLogout} /><Reinforcement /></>} />
            <Route path="/reinforcement2" element={<><Navbar isAuthenticated={isAuthenticated} onLogout={handleLogout} /><LogP /></>} />
            <Route path="/diagnosis" element={<><Navbar isAuthenticated={isAuthenticated} onLogout={handleLogout} /><Diagnosis /></>} />
            <Route path="/docking" element={<><Navbar isAuthenticated={isAuthenticated} onLogout={handleLogout} /><DockingForm /></>} />
            <Route path="/masking" element={<><Navbar isAuthenticated={isAuthenticated} onLogout={handleLogout} /><Chembert /></>} />
            <Route path="/protein" element={<><Navbar isAuthenticated={isAuthenticated} onLogout={handleLogout} /><ProteinViewer /></>} />
            <Route path="/vit" element={<><Navbar isAuthenticated={isAuthenticated} onLogout={handleLogout} /><VIT /></>} />
            <Route path="/profile" element={<><Navbar isAuthenticated={isAuthenticated} onLogout={handleLogout} /><ProfilePage /></>} />
            <Route path="/visualization" element={<><Navbar isAuthenticated={isAuthenticated} onLogout={handleLogout} /><Visualization /></>} />
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
          </Route>

          {/* Catch-all Route */}
          <Route path="*" element={<Navigate to="/login" />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
