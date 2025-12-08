import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import HomePage from './components/Homepage';
import ServicePage from './components/Servicepage';
import ASPServicePage from './components/ASP-Service';
import OPPServicePage from './components/OPP-Service.js';
import SIMServicePage from './components/SIM-Service.js';
import AboutUs from './components/AboutUs.js';
import ContactUs from './components/ContactUs.js';
import Signup from './components/Signup.js';
import Login from './components/Login.js';
import { authAPI } from './utils/auth';
import './App.css';

// Protected Route Component
const ProtectedRoute = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkAuth = async () => {
      const authenticated = await authAPI.verifyToken();
      setIsAuthenticated(authenticated);
      setLoading(false);
    };
    
    checkAuth();
  }, []);

  if (loading) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh',
        fontSize: '18px',
        color: '#333'
      }}>
        Loading...
      </div>
    );
  }

  return isAuthenticated ? children : <Navigate to="/login" replace />;
};

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          {/* Public routes */}
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/about" element={<AboutUs />} />
          <Route path="/contact" element={<ContactUs />} />
          
          {/* Protected routes - require authentication */}
          <Route path="/" element={
            <ProtectedRoute>
              <HomePage />
            </ProtectedRoute>
          } />
          <Route path="/services" element={
            <ProtectedRoute>
              <ServicePage />
            </ProtectedRoute>
          } />
          <Route path="/analyze-set-piece" element={
            <ProtectedRoute>
              <ASPServicePage />
            </ProtectedRoute>
          } />
          <Route path="/optimize-player-positioning" element={
            <ProtectedRoute>
              <OPPServicePage />
            </ProtectedRoute>
          } />
          <Route path="/simulate-strategies" element={
            <ProtectedRoute>
              <SIMServicePage />
            </ProtectedRoute>
          } />
          
          {/* Redirect any unknown routes to login */}
          <Route path="*" element={<Navigate to="/login" replace />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;