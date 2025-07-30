import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import '../styles/MobileVerificationPage.css';

const MobileVerificationPage = () => {
  const [phoneNumber, setPhoneNumber] = useState('');
  const [otp, setOtp] = useState('');
  const [message, setMessage] = useState('');
  const location = useLocation();
  const navigate = useNavigate();
  const { email } = location.state || {};

  const handleSendOTP = async () => {
    try {
      await axios.post('http://localhost:5000/api/auth/register/send-mobile-otp', { phoneNumber, email });
      setMessage('OTP sent to your mobile number.');
    } catch (error) {
      setMessage('Failed to send OTP. Please try again.');
    }
  };

  const handleVerifyOTP = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/auth/register/verify-mobile-otp', { phoneNumber, otp, email });
      if (response.status === 201) {
        setMessage('Mobile verification successful! Redirecting to login...');
        setTimeout(() => navigate('/login'), 2000);
      }
    } catch (error) {
      setMessage('Invalid OTP. Please try again.');
    }
  };

  return (
    <div className="mobile-container">
      <div className="mobile-overlay"></div>
      <div className="mobile-form-container">
        <h2 className="mobile-title">Verify Mobile Number</h2>
        <input
          type="text"
          placeholder="Enter Phone Number"
          value={phoneNumber}
          onChange={(e) => setPhoneNumber(e.target.value)}
          className="mobile-input"
        />
        <button onClick={handleSendOTP} className="mobile-button">
          Send OTP
        </button>
        <input
          type="text"
          placeholder="Enter OTP"
          value={otp}
          onChange={(e) => setOtp(e.target.value)}
          className="mobile-input"
        />
        <button onClick={handleVerifyOTP} className="mobile-button">
          Verify OTP
        </button>
        {message && <p className="mobile-message">{message}</p>}
      </div>
    </div>
  );
};

export default MobileVerificationPage;
