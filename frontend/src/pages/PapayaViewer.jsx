// import React, { useState } from 'react';

// const PapayaViewer = () => {
//     const [selectedFile, setSelectedFile] = useState(null);

//     const handleFileChange = (event) => {
//         const file = event.target.files[0];
//         if (file && file.name.endsWith('.nii.gz')) {
//             setSelectedFile(file);
//         } else {
//             alert('Please upload a valid .nii.gz file.');
//             setSelectedFile(null);
//         }
//     };

//     const handleSubmit = (event) => {
//         event.preventDefault();
//         if (!selectedFile) {
//             alert('Please select a .nii.gz file before submitting.');
//             return;
//         }

//         // For now, just log the file (or later you can load it into the Papaya viewer)
//         console.log('Uploaded file:', selectedFile);

//         // If you plan to display it with Papaya, you would integrate Papaya here.
//     };

//     return (
//         <div style={{ padding: '20px', maxWidth: '600px', margin: '0 auto' }}>
//             <h1 style={{ marginBottom: '20px' }}>Upload .nii.gz File</h1>
//             <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
//                 <input
//                     type="file"
//                     accept=".nii.gz"
//                     onChange={handleFileChange}
//                     style={{ padding: '10px', border: '1px solid #ccc', borderRadius: '5px' }}
//                 />
//                 <button
//                     type="submit"
//                     style={{
//                         padding: '10px 20px',
//                         backgroundColor: '#4caf50',
//                         color: 'white',
//                         border: 'none',
//                         borderRadius: '5px',
//                         cursor: 'pointer',
//                         fontSize: '16px',
//                     }}
//                 >
//                     Upload and View
//                 </button>
//             </form>

//             {selectedFile && (
//                 <div style={{ marginTop: '20px' }}>
//                     <h3>Selected File:</h3>
//                     <p>{selectedFile.name}</p>
//                 </div>
//             )}
//         </div>
//     );
// };

// export default PapayaViewer;
import React, { useState } from 'react';
import { uploadNiiFile } from '../services/api'; // Ensure correct path

const PapayaViewer = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploading, setUploading] = useState(false); // State to manage upload progress

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file && file.name.endsWith('.nii.gz')) {
            setSelectedFile(file);
        } else {
            alert('Please upload a valid .nii.gz file.');
            setSelectedFile(null);
        }
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        if (!selectedFile) {
            alert('Please select a .nii.gz file before submitting.');
            return;
        }

        setUploading(true); // Start uploading process

        try {
            const data = await uploadNiiFile(selectedFile, (progressEvent) => {
                const progress = Math.round((progressEvent.loaded / progressEvent.total) * 100);
                console.log(`Upload progress: ${progress}%`);
                // Optionally, update progress state here to show a progress bar
            });

            console.log('Upload response:', data);

            if (data.success) {
                alert('Upload successful!');
                // You can load the file into the Papaya viewer here
            } else {
                alert('Upload failed!');
            }
        } catch (error) {
            console.error('Error uploading file:', error);
            alert('Error uploading file.');
        } finally {
            setUploading(false); // Stop the uploading process
        }
    };

    return (
        <div style={{ padding: '20px', maxWidth: '600px', margin: '0 auto' }}>
            <h1 style={{ marginBottom: '20px' }}>Upload .nii.gz File</h1>
            <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <input
                    type="file"
                    accept=".nii.gz"
                    onChange={handleFileChange}
                    style={{ padding: '10px', border: '1px solid #ccc', borderRadius: '5px' }}
                />
                <button
                    type="submit"
                    disabled={uploading} // Disable button during upload
                    style={{
                        padding: '10px 20px',
                        backgroundColor: '#4caf50',
                        color: 'white',
                        border: 'none',
                        borderRadius: '5px',
                        cursor: uploading ? 'not-allowed' : 'pointer',
                        fontSize: '16px',
                    }}
                >
                    {uploading ? 'Uploading...' : 'Upload and View'}
                </button>
            </form>

            {selectedFile && (
                <div style={{ marginTop: '20px' }}>
                    <h3>Selected File:</h3>
                    <p>{selectedFile.name}</p>
                </div>
            )}
        </div>
    );
};

export default PapayaViewer;
