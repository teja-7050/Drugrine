// src/components/Visualization.jsx
import React, { useState } from 'react';
import { uploadNiiFile, uploadImage } from '../services/api';
import '../styles/Visualization.css'; // Ensure this path is correct

const Visualization = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [dragActive, setDragActive] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [uploadError, setUploadError] = useState(null);

    const handleFileChange = (e) => {
        setSelectedFile(e.target.files[0]);
        setUploadError(null);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setDragActive(false);
        const file = e.dataTransfer.files[0];
        setSelectedFile(file);
        setUploadError(null);
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setDragActive(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        setDragActive(false);
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            setUploadError('No file selected.');
            return;
        }

        setUploading(true);

        try {
            let response;
            if (selectedFile.name.endsWith('.nii.gz')) {
                response = await uploadNiiFile(selectedFile);
            } else {
                response = await uploadImage(selectedFile, 'lung');
            }

            if (response && response.success) {
                console.log("hiii");
                const originalPath = encodeURIComponent(response.originalImage);
                const predictedPath = encodeURIComponent(response.predictedImage || '');

                window.location.href = "http://localhost:8000/papaya";
            } else {
                setUploadError(response.error || 'Upload failed');
            }
        } catch (error) {
            setUploadError(error.message || 'Upload error');
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="visualization-container">
            <header className="visualization-header">
                <h1>3D Diagnosis Viewer</h1>
                <p>Visualize and interact with 3D medical imaging data.</p>
            </header>

            <main className="visualization-main">
                <div
                    className={`upload-area ${dragActive ? 'active' : ''}`}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                >
                    <input
                        type="file"
                        accept=".nii.gz,image/*"
                        hidden
                        id="fileUpload"
                        onChange={handleFileChange}
                    />
                    <label htmlFor="fileUpload" className="browse-button">
                        {uploading ? 'Uploading...' : 'Drag & drop or Browse files'}
                    </label>
                    {selectedFile && <p>Selected: {selectedFile.name}</p>}
                    {uploadError && <p className="error-message">{uploadError}</p>}
                </div>

                <button
                    className="predict-button"
                    onClick={handleUpload}
                    disabled={uploading || !selectedFile}
                >
                    {selectedFile?.name.endsWith('.nii.gz') ? 'Predict & Visualize' : 'Upload Image'}
                </button>

                {uploading && <p>Processing and redirecting...</p>}
                {uploadError && <p className="error-message">{uploadError}</p>}
            </main>
        </div>
    );
};

export default Visualization;
