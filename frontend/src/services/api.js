// import axios from 'axios';

// // Hardcoded BASE_URL for image upload - REPLACE WITH YOUR ACTUAL UPLOAD API URL
// const BASE_URL = 'http://localhost:5000/api'; // Adjust if your upload endpoint is different

// export const uploadImage = async (imageFile, modelType, onUploadProgress) => {
//     const formData = new FormData();
//     formData.append("image", imageFile);
//     formData.append("modelType", modelType);

//     try {
//         console.log("HI");
//         const response = await axios.post(`${BASE_URL}/upload`, formData, {
//             headers: {
//                 'Content-Type': 'multipart/form-data',
//             },
//             onUploadProgress: onUploadProgress, // This is where you pass the progress callback
//         });
//         console.log(response);
//         return response.data; // axios returns response.data for the JSON body
//     } catch (error) {
//         console.error("API Error:", error);
//         return { success: false, error: error.message };
//     }
// };

// // The other authentication-related functions (sendOtp, verifyOtp, loginUser, fetchAPI)
// // remain unchanged as per your request.
// const API_URL = 'http://localhost:5000/api/auth';

// // Generic API request function
// const fetchAPI = async (endpoint, method, body = null) => {
//     try {
//         const options = {
//             method,
//             headers: {
//                 'Content-Type': 'application/json',
//             },
//             body: body ? JSON.stringify(body) : null,
//         };

//         const response = await fetch(`${API_URL}${endpoint}`, options);

//         if (!response.ok) {
//             // Handle both JSON and text error responses
//             const errorText = await response.text();
//             try {
//                 const errorData = JSON.parse(errorText);
//                 throw new Error(errorData.message || 'An error occurred');
//             } catch {
//                 throw new Error(errorText || 'An unknown error occurred');
//             }
//         }

//         return await response.json();
//     } catch (error) {
//         console.error("API Error:", error.message);
//         throw error;
//     }
// };

// // Send OTP for registration
// export const sendOtp = async (username, email) => {
//     return await fetchAPI('/register/send-otp', 'POST', { username, email });
// };

// // Verify OTP and complete registration
// export const verifyOtp = async (username, email, otp, password) => {
//     return await fetchAPI('/register/verify-otp', 'POST', { username, email, otp, password });
// };

// // Login user (Stores token in localStorage)
// export const loginUser = async (username, password) => {
//     const data = await fetchAPI('/login', 'POST', { username, password });

//     if (data.token) {
//         localStorage.setItem('authToken', data.token);
//     }

//     return data;
// };
import axios from 'axios';

// Base URL for your server
const BASE_URL = 'http://localhost:5000/api'; // Adjust if your upload endpoint is different

// Upload function for generic image files (like for model-related uploads)
export const uploadImage = async (imageFile, modelType, onUploadProgress) => {
    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("modelType", modelType);

    try {
        const response = await axios.post(`${BASE_URL}/upload`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
            onUploadProgress: onUploadProgress,
        });
        console.log("Image upload response:", response.data);
        return response.data;
    } catch (error) {
        console.error("Error uploading image:", error);
        return { success: false, error: error.message };
    }
};

// Upload function for .nii.gz files (specific to your use case)
export const uploadNiiFile = async (file, onUploadProgress) => {
    const formData = new FormData();
    formData.append("file", file); // Append the .nii.gz file

    try {
        console.log("Hello");
        const response = await axios.post(`${BASE_URL}/uploadnii`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
            onUploadProgress: onUploadProgress, // Optional: track upload progress
        });
        console.log("NII file upload response:", response.data);
        return response.data;
    } catch (error) {
        console.error("Error uploading .nii.gz file:", error);
        return { success: false, error: error.message };
    }
};

// Generic API request function
const fetchAPI = async (endpoint, method, body = null) => {
    try {
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            },
            body: body ? JSON.stringify(body) : null,
        };

        const response = await fetch(`${BASE_URL}${endpoint}`, options);

        if (!response.ok) {
            const errorText = await response.text();
            try {
                const errorData = JSON.parse(errorText);
                throw new Error(errorData.message || 'An error occurred');
            } catch {
                throw new Error(errorText || 'An unknown error occurred');
            }
        }

        return await response.json();
    } catch (error) {
        console.error("API Error:", error.message);
        throw error;
    }
};

// Send OTP for registration
export const sendOtp = async (username, email) => {
    return await fetchAPI('/register/send-otp', 'POST', { username, email });
};

// Verify OTP and complete registration
export const verifyOtp = async (username, email, otp, password) => {
    return await fetchAPI('/register/verify-otp', 'POST', { username, email, otp, password });
};

// Login user (Stores token in localStorage)
export const loginUser = async (username, password) => {
    const data = await fetchAPI('/login', 'POST', { username, password });

    if (data.token) {
        localStorage.setItem('authToken', data.token);
    }

    return data;
};
