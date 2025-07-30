import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import '../styles/OTPVerificationPage.css';

const OTPVerificationPage = () => {
  const [otp, setOtp] = useState('');
  const [message, setMessage] = useState('');
  const location = useLocation();
  const navigate = useNavigate();
  const email = location.state?.email;

  useEffect(() => {
    if (!email) {
      navigate('/signup');
    }
  }, [email, navigate]);

  const handleOtpVerification = async (e) => {
    e.preventDefault();
    if (!otp) {
      setMessage('Please enter the OTP.');
      return;
    }

    try {
      const storedData = JSON.parse(localStorage.getItem('signupData') || '{}');
      const response = await axios.post(
        'http://localhost:5000/api/auth/register/verify-otp',
        { email, otp, password: storedData.password, username: storedData.username },
        { withCredentials: true }
      );

      if (response.status === 200) {
        setMessage('OTP verified! Redirecting to mobile verification page...');
        navigate('/verify-mobile', { state: { email } });
      } else {
        setMessage('Invalid OTP. Please try again.');
      }
    } catch (error) {
      setMessage('OTP verification failed. Try again.');
    }
  };

  const handleResendOTP = async () => {
    try {
      await axios.post('http://localhost:5000/api/auth/register/send-otp', { email });
      setMessage('New OTP sent to your email.');
    } catch (error) {
      setMessage('Failed to resend OTP. Try again.');
    }
  };

  return (
    <div className="otp-container">
      <div className="otp-overlay"></div>
      <div className="otp-form-container">
        <h2 className="otp-title">Verify OTP</h2>
        <form onSubmit={handleOtpVerification}>
          <input
            type="text"
            placeholder="Enter OTP"
            value={otp}
            onChange={(e) => setOtp(e.target.value)}
            className="otp-input"
            required
          />
          <button type="submit" className="otp-button">
            Verify OTP
          </button>
        </form>
        {message && <p className="otp-message">{message}</p>}
        <p className="otp-link" onClick={handleResendOTP}>
          Resend OTP
        </p>
      </div>
    </div>
  );
};

export default OTPVerificationPage;
