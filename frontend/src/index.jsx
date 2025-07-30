import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';
import * as $3Dmol from '3dmol';
window.$3Dmol = $3Dmol;


const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
