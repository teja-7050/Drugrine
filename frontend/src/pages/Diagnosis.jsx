import React, { useState, useRef, useEffect } from "react";
import { uploadImage } from "../services/api";
import '../styles/diagnosis.css';
import Navbar from "../components/Navbar";

const Diagnosis = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [originalImage, setOriginalImage] = useState(null);
    const [predictedMask, setPredictedMask] = useState(null);
    const [overlayImage, setOverlayImage] = useState(null);
    const [loading, setLoading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [uploadError, setUploadError] = useState("");
    const [modelType, setModelType] = useState("brain");
    const fileInputRef = useRef(null);
    const [uploadInitiated, setUploadInitiated] = useState(false);

    const handleDragOver = (event) => {
        event.preventDefault();
    };

    const handleDrop = (event) => {
        event.preventDefault();
        const file = event.dataTransfer.files[0];
        handleFile(file);
    };

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        handleFile(file);
    };

    const handleFile = (file) => {
        if (file) {
            if (!file.type.startsWith('image/')) {
                setUploadError("Please upload an image file.");
                setSelectedFile(null);
                setOriginalImage(null);
                return;
            }

            const maxSizeMB = 5;
            if (file.size > maxSizeMB * 1024 * 1024) {
                setUploadError(`File size exceeds the limit of ${maxSizeMB}MB.`);
                setSelectedFile(null);
                setOriginalImage(null);
                return;
            }

            setSelectedFile(file);
            setOriginalImage(URL.createObjectURL(file));
            setPredictedMask(null);
            setOverlayImage(null);
            setUploadError("");
        }
    };

    const createOverlayImage = async () => {
        if (!originalImage || !predictedMask) return;

        const original = new Image();
        const mask = new Image();

        original.src = originalImage;
        mask.src = predictedMask;

        await Promise.all([
            new Promise((resolve) => (original.onload = resolve)),
            new Promise((resolve) => (mask.onload = resolve)),
        ]);

        const canvas = document.createElement("canvas");
        canvas.width = original.width;
        canvas.height = original.height;
        const ctx = canvas.getContext("2d");

        ctx.drawImage(original, 0, 0, canvas.width, canvas.height);

        const maskCanvas = document.createElement("canvas");
        maskCanvas.width = mask.width;
        maskCanvas.height = mask.height;
        const maskCtx = maskCanvas.getContext("2d");
        maskCtx.drawImage(mask, 0, 0, maskCanvas.width, maskCanvas.height);
        const imageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
        const data = imageData.data;

        for (let i = 0; i < data.length; i += 4) {
            if (data[i] > 200 && data[i + 1] > 200 && data[i + 2] > 200) {
                data[i] = 255;
                data[i + 1] = 0;
                data[i + 2] = 0;
                data[i + 3] = 150;
            } else {
                data[i + 3] = 0;
            }
        }

        maskCtx.putImageData(imageData, 0, 0);
        ctx.drawImage(maskCanvas, 0, 0, canvas.width, canvas.height);
        setOverlayImage(canvas.toDataURL());
    };

    useEffect(() => {
        createOverlayImage();
    }, [predictedMask]);

    const handleUpload = async () => {
        if (!selectedFile) {
            setUploadError("Please select an image.");
            return;
        }

        setLoading(true);
        setUploadProgress(0);
        setUploadError("");
        setUploadInitiated(true); // Set upload initiated state

        try {
            const data = await uploadImage(selectedFile, modelType, (progressEvent) => {
                const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                setUploadProgress(progress);
            });

            if (data.success) {
                const filePath = data.predictedImage;
                const normalizedPath = filePath.replace(/\\/g, "/");
                const filename = normalizedPath.split("/").pop();
                setPredictedMask(`/${filename}`);
            } else {
                setUploadError("Prediction failed.");
            }
        } catch (error) {
            console.error("Upload error:", error);
            setUploadError("An error occurred during upload.");
        } finally {
            setLoading(false);
        }
    };

    const handleRetry = () => {
        setSelectedFile(null);
        setOriginalImage(null);
        setPredictedMask(null);
        setOverlayImage(null);
        setLoading(false);
        setUploadProgress(0);
        setUploadError("");
        setUploadInitiated(false); // Reset upload initiated state
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };

    return (
        <>
            
            
            <div className="container"> {/* Add the container div here */}
                <div className="card">
                    <h2>Upload MRI/CT Scan</h2>
                    <label>Select Model: </label>
                    <select onChange={(e) => setModelType(e.target.value)} value={modelType}>
                        <option value="brain">Brain Tumor</option>
                        <option value="lung">Lung Tumor</option>
                    </select>
                    <br />

                    {!uploadInitiated && (
                        <div
                            className="upload-area"
                            onDragOver={handleDragOver}
                            onDrop={handleDrop}
                            style={{ border: '2px dashed #ccc', padding: '20px', textAlign: 'center', cursor: 'pointer' }}
                        >
                            <p>Drag and drop your image here</p>
                            <input
                                type="file"
                                accept="image/*"
                                onChange={handleFileChange}
                                style={{ display: 'none' }}
                                ref={fileInputRef}
                                multiple={false}
                            />
                            <button type="button" onClick={() => fileInputRef.current.click()}>
                                Browse Files
                            </button>
                        </div>
                    )}
                    <br />

                    {selectedFile && (
                        <p>Selected file: {selectedFile.name} ({Math.round(selectedFile.size / 1024)} KB)</p>
                    )}

                    {uploadError && <p className="error-message" style={{ color: 'red' }}>{uploadError}</p>}

                    <button onClick={handleUpload} disabled={loading || !selectedFile}>
                        {loading ? `Processing... ${uploadProgress}%` : "Upload"}
                    </button>

                    {loading && uploadProgress > 0 && uploadProgress < 100 && (
                        <div className="progress-bar-container" style={{ backgroundColor: '#f0f0f0', borderRadius: '5px', height: '10px', margin: '10px 0' }}>
                            <div
                                className="progress-bar"
                                style={{
                                    backgroundColor: '#4CAF50',
                                    height: '100%',
                                    width: `${uploadProgress}%`,
                                    borderRadius: '5px',
                                }}
                            ></div>
                        </div>
                    )}

                    {uploadError && (
                        <button onClick={handleRetry}>Retry Upload</button>
                    )}

                    {/* Display Images */}
                    <div className="image-grid">
                        {originalImage && (
                            <div className="image-container">
                                <h3>Original Image</h3>
                                <img src={originalImage} alt="Original" style={{ maxWidth: '100%', height: 'auto' }} />
                            </div>
                        )}
                        {predictedMask && (
                            <div className="image-container">
                                <h3>Predicted Mask</h3>
                                <img src={predictedMask} alt="Tumor Mask" style={{ maxWidth: '100%', height: 'auto' }} />
                            </div>
                        )}
                        {overlayImage && (
                            <div className="image-container">
                                <h3>Overlay Visualization</h3>
                                <img src={overlayImage} alt="Overlayed Tumor" style={{ maxWidth: '100%', height: 'auto' }} />
                            </div>
                        )}
                    </div>
                </div>
            </div>
            
            
        </>
    );
};

export default Diagnosis;