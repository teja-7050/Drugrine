import nodemailer from "nodemailer";
import dotenv from "dotenv";

dotenv.config();

// Configure transporter
const transporter = nodemailer.createTransport({
    service: "gmail", // You can use SMTP or another service
    auth: {
        user: process.env.EMAIL,
        pass: process.env.EMAIL_PASS
    }
});

// Function to send OTP email
export const sendOtpEmail = async (to, otp) => {
    try {
        await transporter.sendMail({
            from: process.env.EMAIL,
            to,
            subject: "Your OTP for Signup Verification",
            text: `Your OTP is: ${otp}. It will expire in 10 minutes.`
        });

        console.log("📨 OTP email sent successfully to:", to);
    } catch (error) {
        console.error("❌ Error sending OTP email:", error);
    }
};
