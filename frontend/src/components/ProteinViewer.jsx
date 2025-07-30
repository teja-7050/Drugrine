import React, { useEffect, useRef } from 'react';
import '../styles/ProteinViewer.css';

const ProteinViewer = ({ pdbData }) => {
  const viewerRef = useRef(null);

  useEffect(() => {
    if (!pdbData || !window.$3Dmol || !viewerRef.current) return;

    const viewer = window.$3Dmol.createViewer(viewerRef.current, {
      backgroundColor: '#1e1e1e',
    });

    try {
      viewer.addModel(pdbData, 'pdb');
      viewer.setStyle({}, { cartoon: { color: 'spectrum' } });
      viewer.zoomTo();
      viewer.spin(true);
      viewer.render();
    } catch (error) {
      console.error('Error rendering 3Dmol protein:', error);
    }

    // Optional cleanup if needed
    return () => {
      if (viewerRef.current) {
        viewerRef.current.innerHTML = '';
      }
    };
  }, [pdbData]);

  return (
    <div className="viewer-wrapper">
      <div ref={viewerRef} className="viewer-box" />
    </div>
  );
};

export default ProteinViewer;
