const mongoose = require('mongoose');

const UserSchema = new mongoose.Schema({
    username: { type: String, required: true, unique: true, lowercase: true },
    email: { type: String, required: true, unique: true, lowercase: true },
    password: { type: String, required: true }, // Removed conditional
    phoneNumber: { type: String, required: true, unique: true }
}, { timestamps: true });

module.exports = mongoose.model('User', UserSchema);