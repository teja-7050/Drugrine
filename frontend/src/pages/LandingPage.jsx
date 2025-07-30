import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/LandingPage.css';

const LandingPage = () => {
  return (
    <div className="landing-container">
      <div className="landing-overlay"></div>
      <div className="landing-content">
        <h1 className="landing-title">WELCOME TO DRUGRIN AI</h1>
        <p className="landing-paragraph">Discover new treatments and innovations.</p>
        <div className="landing-button-container">
          <Link to="/login" className="landing-button">
            Login
          </Link>
          <Link to="/signup" className="landing-button">
            Sign Up
          </Link>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
