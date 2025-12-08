import React from 'react';
import ReactDOM from 'react-dom/client'; // Import the new root API
import './index.css';
import App from './App';

// Create a root for the app using createRoot
const root = ReactDOM.createRoot(document.getElementById('root'));

// Render the app inside the root
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
