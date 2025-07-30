import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import '../styles/SignupPage.css';

const SignupPage = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [message, setMessage] = useState('');
  const navigate = useNavigate();

  const handleSignup = async (e) => {
    e.preventDefault();
    if (username && email && password) {
      try {
        // Make POST request to backend API
        const response = await axios.post(
          'http://localhost:5000/api/auth/register/send-otp',
          { username, email },
          { withCredentials: true }
        );
        console.log("Signup Response:", response.data);
        console.log("Navigating to OTP Verification with:", { email, password, username });
        // If the signup is successful, navigate and show message
        if (response.status === 200) {
          setMessage(response.data.message);
          localStorage.setItem("signupData", JSON.stringify({ email, password, username }));
          navigate('/verify-otp', { state: { email, password, username } });
        } else {
          setMessage(response.data.message || "Signup failed. Please try again.");
        }
      } catch (error) {
        console.error('Error during signup:', error);
        setMessage('An error occurred during signup. Please try again.');
      }
    } else {
      setMessage('All fields are required.');
    }
  };

  const navigateToLogin = () => {
    navigate('/login');
  };

  return (
    <div className="signup-container">
      <div className="signup-overlay"></div>
      <div className="signup-form-container">
        <h2 className="signup-title">Create an Account</h2>
        <form onSubmit={handleSignup}>
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="signup-input"
            required
          />
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="signup-input"
            required
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="signup-input"
            required
          />
          <button type="submit" className="signup-button">
            Sign Up
          </button>
        </form>
        {message && <p className="signup-error-message">{message}</p>}
        <p className="signup-link-text" onClick={navigateToLogin}>
          Already have an account? Log in
        </p>
      </div>
    </div>
  );
};

export default SignupPage;
