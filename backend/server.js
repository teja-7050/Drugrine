const express = require('express');
const cors = require('cors');
const multer = require("multer");
const path = require("path");
const bodyParser = require('body-parser');
const cookieParser = require('cookie-parser');
const axios = require('axios'); // Added axios import
const connectDB = require('./config/db');
const authRoutes = require('./routes/auth');
const mongoose = require('mongoose');
const twilio = require('twilio');
const FormData = require('form-data'); // Make sure this line is at the top of your server.js
const fs = require('fs'); // Import the 'fs' module
const apiRoutes = require('./routes/apiRoutes');



mongoose.connect(process.env.MONGODB_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true
}).then(() => console.log("✅ MongoDB Connected"))
  .catch(err => console.error("❌ MongoDB Connection Error:", err));

require('dotenv').config();

const app = express();
connectDB();

// CORS Configuration to allow frontend & cookies
app.use(
    cors({
        origin: "http://localhost:3000",
        credentials: true, // Allow cookies & authentication headers
        methods: "GET, POST, PUT, DELETE",
        allowedHeaders: "Content-Type, Authorization",
    })
);

// Middleware for JSON, URL encoding & cookies
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cookieParser());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// API Routes
app.use('/api/auth', authRoutes);
app.use('/api', apiRoutes);

app.use(express.static("frontend"));
app.use(express.json());

// Start server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
