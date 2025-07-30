// const express = require('express');
// const cors = require('cors');
// const multer = require("multer");
// const path = require("path");
// const axios = require('axios'); 
// const FormData = require('form-data'); // Make sure this line is at the top of your server.js
// const fs = require('fs'); // Import the 'fs' module
// const router = express.Router();



// const uploadFolder = path.join(__dirname, "uploads");

// // Ensure upload folder exists
// if (!fs.existsSync(uploadFolder)) {
//     fs.mkdirSync(uploadFolder, { recursive: true });
// }

// // Configure Multer for file uploads
// const storage = multer.diskStorage({
//     destination: (req, file, cb) => cb(null, uploadFolder),
//     filename: (req, file, cb) => cb(null, `${Date.now()}${path.extname(file.originalname)}`)
// });
// const upload = multer({ storage });



// router.post('/reinforcement', async (req, res) => {
//     try {
        
//         const { state } = req.body;
//         console.log("OUTPUT :"+ req.body.state);
        
//         const payload = { SMILES: state };
//         console.log(payload);
//         const fastApiUrl = 'http://localhost:8000/predict/reinforcement';
//         const response = await axios.post(fastApiUrl, payload);
//         console.log(response);
//         res.json(response.data);
//     } catch (error) {
//         console.error("Error in /api/reinforcement:", error);
//         res.status(500).json({ error: "Failed to get prediction from FastAPI." });
//     }
// });
// router.post('/vit', async (req, res) => {
//     try {
        
//         const { state } = req.body;
//         console.log("OUTPUT :"+ req.body.state);
        
//         const payload = { SMILES: state };
//         console.log(payload);
//         const fastApiUrl = 'http://localhost:8000/predict/vit';
//         const response = await axios.post(fastApiUrl, payload);
//         console.log(response);
//         res.json(response.data);
//     } catch (error) {
//         console.error("Error in /api/reinforcement:", error);
//         res.status(500).json({ error: "Failed to get prediction from FastAPI." });
//     }
// });

// router.post('/reinforcement/logp', async (req, res) => {
//     try {
        
//         const { state } = req.body;
//         console.log("OUTPUT :"+ req.body.state);
        
//         const payload = { SMILES: state };
//         console.log(payload);
//         const fastApiUrl = 'http://localhost:8000/predict/reinforcement/logp';
//         const response = await axios.post(fastApiUrl, payload);
//         console.log(response);
//         res.json(response.data);
//     } catch (error) {
//         console.error("Error in /api/reinforcement:", error);
//         res.status(500).json({ error: "Failed to get prediction from FastAPI." });
//     }
// });


// router.post("/upload", upload.single("image"), async (req, res) => {
//     const { modelType } = req.body;

//     if (!req.file) {
//         return res.status(400).json({ error: "No file uploaded" });
//     }
//     if (!modelType || !["brain", "lung"].includes(modelType)) {
//         return res.status(400).json({ error: "Invalid model type" });
//     }

//     const fastApiUrl = 'http://localhost:8000/predict/segmentation';

//     console.log("req.file:", req.file);
//     console.log(`File uploaded at path: ${req.file.path}`);
//     console.log(`Model selected: ${modelType}`);

//     try {
//         let fileBuffer;
//         try {
//             fileBuffer = fs.readFileSync(req.file.path);
//             console.log(`Successfully read file into buffer: ${fileBuffer.length} bytes`);
//         } catch (readError) {
//             console.error("Error reading file with readFileSync:", readError);
//             if (!res.headersSent) {
//                 return res.status(500).json({ error: "Failed to read the uploaded file." });
//             }
//             return;
//         }

//         const formData = new FormData();
//         formData.append("image_file", fileBuffer, {
//             filename: req.file.originalname,
//             contentType: req.file.mimetype,
//         });
//         formData.append("model_type", modelType);

//         const response = await axios.post(fastApiUrl, formData, {
//             headers: {
//                 ...formData.getHeaders(), // Crucial: Get headers from form-data
//             },
//         });

//         console.log("FastAPI Response:", response.data);

//         if (fs.existsSync(req.file.path)) {
//             fs.unlinkSync(req.file.path);
//             console.log(`Successfully unlinked (deleted) file after FastAPI call.`);
//         } else {
//             console.log(`File not found during unlink after FastAPI call.`);
//         }

//         if (response.data.success) {
//             return res.json({ success: true, predictedImage: response.data.mask_path });
//         } else {
//             return res.status(500).json({ error: "Model failed", details: response.data.message });
//         }

//     } catch (error) {
//         console.error("Error communicating with FastAPI:", error);
//         if (req.file && fs.existsSync(req.file.path)) {
//             fs.unlinkSync(req.file.path);
//             console.log(`Successfully unlinked (deleted) file after FastAPI error.`);
//         } else {
//             console.log(`File not found during unlink after FastAPI error.`);
//         }
//         if (!res.headersSent) {
//             return res.status(500).json({ error: "Failed to get prediction from FastAPI.", details: error.message });
//         }
//     }
// });

// router.post('/protein2smiles', async (req, res) => {
//     try {
        
//         const { state } = req.body;
//         console.log("OUTPUT :"+ req.body.state);
        
//         const payload = { PROTEIN: state };
//         console.log(payload);
//         const fastApiUrl = 'http://localhost:8000/predict/protein2smiles';
//         const response = await axios.post(fastApiUrl, payload);
//         console.log(response);
//         res.json(response.data);
//     } catch (error) {
//         console.error("Error in /api/protein2smiles:", error);
//         res.status(500).json({ error: "Failed to get prediction from FastAPI." });
//     }
// });

// router.post('/mask', async (req, res) => {
//     try {
        
//         const { state } = req.body;
//         console.log("OUTPUT :"+ req.body.state);
        
//         const payload = { SMILES: state };
//         console.log(payload);
//         const fastApiUrl = 'http://localhost:8000/predict/mask';
//         const response = await axios.post(fastApiUrl, payload);
//         console.log(response);
//         res.json(response.data);
//     } catch (error) {
//         console.error("Error in /api/nask:", error);
//         res.status(500).json({ error: "Failed to get prediction from FastAPI." });
//     }
// });
// router.post('/fold', async (req, res) => {
//     try {
//       const { sequence } = req.body;
      
//       const response = await axios.post(
//         'https://api.esmatlas.com/foldSequence/v1/pdb/',
//         sequence,
//         {
//           headers: {
//             'Content-Type': 'application/x-www-form-urlencoded',
//           }
//         }
//       );
      
//       const pdbString = response.data;
//       res.json({ pdb: pdbString });
//     } catch (error) {
//       console.error('Error:', error);
//       res.status(500).json({ error: 'Failed to predict protein structure' });
//     }
//   });
//   router.post('/docking', async (req, res) => {
//     try {
//         const { ec_number, ligand_id } = req.body;
//         console.log("Received data for docking:", { ec_number, ligand_id });

//         const fastApiUrl = 'http://localhost:8000/predict/docking';
//         const payload = { ec_number: ec_number, ligand_id: ligand_id };

//         const response = await axios.post(fastApiUrl, payload);
//         console.log("Response from FastAPI:", response.data);
//         res.json(response.data); // Send the FastAPI response back to the frontend

//     } catch (error) {
//         console.error("Error in /api/docking:", error);
//         res.status(500).json({ error: "Failed to send data to FastAPI or receive response." });
//     }
// });
// module.exports = router;

const express = require('express');
const cors = require('cors');
const multer = require("multer");
const path = require("path");
const axios = require('axios'); 
const FormData = require('form-data'); // Make sure this line is at the top of your server.js
const fs = require('fs'); // Import the 'fs' module
const { } = require('console');
const router = express.Router();


const uploadFolder = path.join(__dirname, "uploads");


// Ensure upload folder exists
if (!fs.existsSync(uploadFolder)) {
    fs.mkdirSync(uploadFolder, { recursive: true });
}

// Configure Multer for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, uploadFolder),
    // filename: (req, file, cb) => cb(null, `${Date.now()}${path.extname(file.originalname)}`)
    filename: (req, file, cb) => cb(null, `${Date.now()}-${file.originalname}`)

});
const upload = multer({ storage });


// Existing routes for reinforcement, vit, etc.
router.post('/reinforcement', async (req, res) => {
    try {
        
        const { state } = req.body;
        console.log("OUTPUT :"+ req.body.state);
        
        const payload = { SMILES: state };
        console.log(payload);
        const fastApiUrl = 'http://localhost:8000/predict/reinforcement';
        const response = await axios.post(fastApiUrl, payload);
        console.log(response);
        res.json(response.data);
    } catch (error) {
        console.error("Error in /api/reinforcement:", error);
        res.status(500).json({ error: "Failed to get prediction from FastAPI." });
    }
});

router.post('/vit', async (req, res) => {
    try {
        
        const { state } = req.body;
        console.log("OUTPUT :"+ req.body.state);
        
        const payload = { SMILES: state };
        console.log(payload);
        const fastApiUrl = 'http://localhost:8000/predict/vit';
        const response = await axios.post(fastApiUrl, payload);
        console.log(response);
        res.json(response.data);
    } catch (error) {
        console.error("Error in /api/reinforcement:", error);
        res.status(500).json({ error: "Failed to get prediction from FastAPI." });
    }
});

router.post('/reinforcement/logp', async (req, res) => {
    try {
        
        const { state } = req.body;
        console.log("OUTPUT :"+ req.body.state);
        
        const payload = { SMILES: state };
        console.log(payload);
        const fastApiUrl = 'http://localhost:8000/predict/reinforcement/logp';
        const response = await axios.post(fastApiUrl, payload);
        console.log(response);
        res.json(response.data);
    } catch (error) {
        console.error("Error in /api/reinforcement:", error);
        res.status(500).json({ error: "Failed to get prediction from FastAPI." });
    }
});

router.post("/upload", upload.single("image"), async (req, res) => {
    const { modelType } = req.body;

    if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
    }
    if (!modelType || !["brain", "lung"].includes(modelType)) {
        return res.status(400).json({ error: "Invalid model type" });
    }

    const fastApiUrl = 'http://localhost:8000/predict/segmentation';

    console.log("req.file:", req.file);
    console.log(`File uploaded at path: ${req.file.path}`);
    console.log(`Model selected: ${modelType}`);

    try {
        let fileBuffer;
        try {
            fileBuffer = fs.readFileSync(req.file.path);
            console.log(`Successfully read file into buffer: ${fileBuffer.length} bytes`);
        } catch (readError) {
            console.error("Error reading file with readFileSync:", readError);
            if (!res.headersSent) {
                return res.status(500).json({ error: "Failed to read the uploaded file." });
            }
            return;
        }

        const formData = new FormData();
        formData.append("image_file", fileBuffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype,
        });
        formData.append("model_type", modelType);

        const response = await axios.post(fastApiUrl, formData, {
            headers: {
                ...formData.getHeaders(), // Crucial: Get headers from form-data
            },
        });

        console.log("FastAPI Response:", response.data);

        if (fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
            console.log(`Successfully unlinked (deleted) file after FastAPI call.`);
        } else {
            console.log(`File not found during unlink after FastAPI call.`);
        }

        if (response.data.success) {
            return res.json({ success: true, predictedImage: response.data.mask_path });
        } else {
            return res.status(500).json({ error: "Model failed", details: response.data.message });
        }

    } catch (error) {
        console.error("Error communicating with FastAPI:", error);
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
            console.log(`Successfully unlinked (deleted) file after FastAPI error.`);
        } else {
            console.log(`File not found during unlink after FastAPI error.`);
        }
        if (!res.headersSent) {
            return res.status(500).json({ error: "Failed to get prediction from FastAPI.", details: error.message });
        }
    }
});

router.post('/protein2smiles', async (req, res) => {
    try {
        
        const { state } = req.body;
        console.log("OUTPUT :"+ req.body.state);
        
        const payload = { PROTEIN: state };
        console.log(payload);
        const fastApiUrl = 'http://localhost:8000/predict/protein2smiles';
        const response = await axios.post(fastApiUrl, payload);
        console.log(response);
        res.json(response.data);
    } catch (error) {
        console.error("Error in /api/protein2smiles:", error);
        res.status(500).json({ error: "Failed to get prediction from FastAPI." });
    }
});

router.post('/mask', async (req, res) => {
    try {
        
        const { state } = req.body;
        console.log("OUTPUT :"+ req.body.state);
        
        const payload = { SMILES: state };
        console.log(payload);
        const fastApiUrl = 'http://localhost:8000/predict/mask';
        const response = await axios.post(fastApiUrl, payload);
        console.log(response);
        res.json(response.data);
    } catch (error) {
        console.error("Error in /api/nask:", error);
        res.status(500).json({ error: "Failed to get prediction from FastAPI." });
    }
});

router.post('/fold', async (req, res) => {
    try {
      const { sequence } = req.body;
      
      const response = await axios.post(
        'https://api.esmatlas.com/foldSequence/v1/pdb/',
        sequence,
        {
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          }
        }
      );
    
      const pdbString = response.data;
      res.json({ pdb: pdbString });
    } catch (error) {
      console.error('Error:', error);
      res.status(500).json({ error: 'Failed to predict protein structure' });
    }
});

router.post('/docking', async (req, res) => {
    try {
        const { ec_number, ligand_id } = req.body;
        console.log("Received data for docking:", { ec_number, ligand_id });

        const fastApiUrl = 'http://localhost:8000/predict/docking';
        const payload = { ec_number: ec_number, ligand_id: ligand_id };

        const response = await axios.post(fastApiUrl, payload);
        console.log("Response from FastAPI:", response.data);
        res.json(response.data); // Send the FastAPI response back to the frontend

    } catch (error) {
        console.error("Error in /api/docking:", error);
        res.status(500).json({ error: "Failed to send data to FastAPI or receive response." });
    }
});

// New route for handling .nii.gz file uploads
// router.post("/uploadnii", upload.single("file"), async (req, res) => {
//     if (!req.file) {
//         return res.status(400).json({ error: "No file uploaded" });
//     }

//     // Here you can add additional logic for processing the .nii.gz file
//     const filePath = req.file.path;
//     console.log(`File uploaded at path: ${filePath}`);
    
//     try {
        
//         // For example, we might process the .nii.gz file here, or forward it to another service
//         // For now, just send the file path back as part of the response
//         return res.json({ success: true, filePath });
//     } catch (error) {
//         console.error("Error processing .nii.gz file:", error);
//         return res.status(500).json({ error: "Failed to process .nii.gz file." });
//     }
// });
// const upload = multer({ dest: 'uploads/' });

router.post("/uploadnii", upload.single("file"), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
    }

    const filePath = req.file.path; // Temporary saved file
    const originalName = req.file.originalname;

    try {
        const fastApiUrl = 'http://localhost:8000/predict/uploadnii';

        // Create a FormData object
        const formData = new FormData();
        // formData.append('file', fs.createReadStream(filePath), originalName);
        formData.append('image_file', fs.createReadStream(filePath), originalName);


        // Send file to FastAPI server
        const response = await axios.post(fastApiUrl, formData, {
            headers: {
                ...formData.getHeaders()
            },
            maxBodyLength: Infinity,
            maxContentLength: Infinity,
        });

        console.log("Response from FastAPI:", response.data);

        // Cleanup: Remove the temp file
        fs.unlinkSync(filePath);

        // Send back FastAPI's response to frontend
        return res.json(response.data);

    } catch (error) {
        console.error("Error sending file to FastAPI:", error);

        // Cleanup even if failed
        if (fs.existsSync(filePath)) {
            fs.unlinkSync(filePath);
        }

        return res.status(500).json({ error: "Failed to process .nii.gz file through FastAPI." });
    }
});

module.exports = router;
