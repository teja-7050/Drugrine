const mongoose = require('mongoose');

const OTPSchema = new mongoose.Schema({
    email: { type: String, required: true },
    otp: { type: String },
    otpExpires: { type: Date },
    mobileOTP: { type: String },
    mobileOTPExpires: { type: Date },
    phoneNumber: { type: String },
    username: { type: String },
    password: { type: String }, // Store hashed password temporarily
    isEmailVerified: { type: Boolean, default: false }
});

module.exports = mongoose.model('OTP', OTPSchema);