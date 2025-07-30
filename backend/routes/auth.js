const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const nodemailer = require('nodemailer');
const twilio = require('twilio');
//const mongoose = require('mongoose');
const User = require('../models/User');
const OTP = require('../models/OTP'); // Ensure you have an OTP model
const router = express.Router();
require('dotenv').config();

const jwtSecret = process.env.JWT_SECRET;

// Nodemailer transporter setup
const transporter = nodemailer.createTransport({
    service: "gmail",
    auth: {
        user: process.env.EMAIL,
        pass: process.env.EMAIL_PASS
    },
    logger: true,
    debug: true,
});
// Twilio configuration
const twilioClient = twilio(
    process.env.TWILIO_ACCOUNT_SID,
    process.env.TWILIO_AUTH_TOKEN
);

// Function to send OTP via Twilio
const sendMobileOTP = async (phoneNumber, otp) => {
    try {
        await twilioClient.messages.create({
            body: `Your OTP for mobile verification is: ${otp}`,
            from: process.env.TWILIO_PHONE_NUMBER,
            to: phoneNumber,//phno
        });
        console.log("📩 Mobile OTP sent successfully");
    } catch (err) {
        console.error("❌ Error sending mobile OTP:", err);
        throw err;
    }
};
// Generate JWT Token
const generateToken = (userId) => {
    return jwt.sign({ userId }, jwtSecret, { expiresIn: '7d' });
};

// Generate OTP Function
const generateOTP = () => {
    return Math.floor(100000 + Math.random() * 900000); // 6-digit OTP
};

// **Step 1: Send OTP for Registration**
router.post('/register/send-otp', async (req, res) => {
    console.log(req.body   );
    let { username, email } = req.body;

    try {
        username = username.trim().toLowerCase();
        email = email.trim().toLowerCase();

        // Check if user already exists
        const existingUser = await User.findOne({ $or: [{ email }, { username }] });

        if (existingUser) {
            console.log("1");
            return res.status(400).json({ message: 'Username or Email already registered' });
        }

        // Generate OTP
        const otp = generateOTP();
        console.log("📩 Generated OTP:", otp);
        console.log(process.env.EMAIL_PASS);
        // Send OTP via email
        await transporter.sendMail({
            from: process.env.EMAIL,
            to: email,
            subject: "Your OTP for Signup Verification",
            text: `Your OTP is: ${otp}. It will expire in 10 minutes.`
        });

        // Store OTP in OTP collection (not User collection)
        await OTP.updateOne(
            { email },
            { otp, otpExpires: Date.now() + 10 * 60 * 1000 },
            { upsert: true }
        );

        res.status(200).json({ message: 'OTP sent successfully' });
    } catch (err) {
        console.log("1");
        console.error('❌ Error in sending OTP:', err);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// **Step 2: Verify OTP and Register User**
router.post('/register/verify-otp', async (req, res) => {
    let { email, otp, password, username } = req.body;

    console.log("🔍 Received OTP Verification Request:", { email, otp });

    if (!password) {
        return res.status(400).json({ message: "Password is missing." });
    }

    try {
        email = email.trim().toLowerCase();
        otp = otp.trim();

        // Check if OTP exists
        const otpRecord = await OTP.findOne({ email });

        if (!otpRecord) {
            console.log("❌ No OTP record found for email:", email);
            return res.status(400).json({ message: 'Invalid or expired OTP' });
        }

        console.log("🔹 Stored OTP:", otpRecord.otp);
        console.log("🔹 Entered OTP:", otp);

        // Compare OTP (Ensuring both are trimmed and string)
        if (String(otpRecord.otp).trim() !== String(otp).trim()) {
            console.log("❌ OTP Mismatch!");
            return res.status(400).json({ message: 'Invalid OTP' });
        }

        // Check if OTP is expired
        if (otpRecord.otpExpires < Date.now()) {
            console.log("❌ OTP Expired!");
            return res.status(400).json({ message: 'Expired OTP' });
        }

        await OTP.updateOne(
            { email },
            { email, username, password, isEmailVerified: true },
            { upsert: true }
        );

        // Redirect to mobile verification
        res.status(200).json({ message: 'Email verified. Proceed to mobile verification.' });
    } catch (err) {
        console.error('❌ Error in OTP verification:', err);
        res.status(500).json({ message: 'Internal server error' });
    }
});
// Send Mobile OTP
router.post('/register/send-mobile-otp', async (req, res) => {
    const { phoneNumber, email } = req.body;
    
    try {
        const otp = generateOTP();
        await sendMobileOTP(phoneNumber, otp); // Use Twilio function

        await OTP.updateOne(
            { email },
            { 
                phoneNumber,
                mobileOTP: otp,
                mobileOTPExpires: Date.now() + 10 * 60 * 1000 
            }
        );

        res.status(200).json({ message: 'Mobile OTP sent' });
    } catch (err) {
        res.status(500).json({ message: 'Failed to send mobile OTP' });
    }
});

// Verify Mobile OTP
router.post('/register/verify-mobile-otp', async (req, res) => {
    const { otp, email } = req.body;
    
    try {
        const otpRecord = await OTP.findOne({ email });
        console.log(otpRecord);
        console.log(otp);
        if (!otpRecord || 
            otpRecord.mobileOTP !== otp.trim() || 
            otpRecord.mobileOTPExpires < Date.now()
        ) {
            return res.status(400).json({ message: 'Invalid OTP' });
        }
        console.log("hi");
        // Create final user
        const newUser = new User({
            username: otpRecord.username,
            email: otpRecord.email,
            password: otpRecord.password,
            phoneNumber: otpRecord.phoneNumber
        });

        await newUser.save();
        await OTP.deleteOne({ email });

        res.status(201).json({ message: 'Registration complete!' });
    } catch (err) {
        console.log(err);
        res.status(500).json({ message: 'Registration failed' });
    }
});
// **Login Route**
router.post('/login', async (req, res) => {
    let { username, password } = req.body;
    console.log("🔑 Login request received:", req.body);

    try {
        username = username.trim().toLowerCase();

        const user = await User.findOne({ username });

        if (!user) {
            console.log("❌ User not found");
            return res.status(400).json({ message: 'Invalid credentials' });
        }

        // Compare hashed password
        const isMatch = password== user.password;
        if (!isMatch) {
            console.log("❌ Password mismatch");
            return res.status(400).json({ message: 'Invalid credentials' });
        }

        // Generate JWT token
        const token = generateToken(user._id);
        console.log("✅ Generated Token:", token);

        // Set token as HTTP-only cookie
        res.cookie('token', token, {
            httpOnly: true,
            secure: process.env.NODE_ENV === 'production',
            sameSite: 'lax',
            maxAge: 3600000, // 1 hour
        });

        return res.json({ message: 'Login successful', success: true, token });
    } catch (err) {
        console.error('❌ Error in login:', err);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// **Logout Route**
router.post('/logout', (req, res) => {
    res.clearCookie('token');
    res.json({ message: 'Logout successful' });
});


// New Route: DRL Model Prediction
router.post('/drl-action', async (req, res) => {
    try {
        const { state } = req.body;
        if (!state) {
            return res.status(400).json({ error: 'State is required' });
        }
        const response = await axios.post('http://localhost:5001/predict', { state });
        res.json(response.data);
    } catch (error) {
        console.error("❌ Error connecting to RL model:", error);
        res.status(500).json({ error: 'Failed to get action from RL model' });
    }
});

module.exports = router;
