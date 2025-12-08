// Authentication utility functions
import React from 'react';

const AUTH_TOKEN_KEY = 'authToken';
const USER_DATA_KEY = 'userData';

export const authAPI = {
  baseURL: 'http://localhost:5002/api/auth',
  
  // Get stored token
  getToken: () => {
    return localStorage.getItem(AUTH_TOKEN_KEY);
  },
  
  // Get stored user data
  getUserData: () => {
    const userData = localStorage.getItem(USER_DATA_KEY);
    return userData ? JSON.parse(userData) : null;
  },
  
  // Check if user is authenticated
  isAuthenticated: () => {
    const token = localStorage.getItem(AUTH_TOKEN_KEY);
    return !!token;
  },
  
  // Store authentication data
  setAuthData: (token, userData) => {
    localStorage.setItem(AUTH_TOKEN_KEY, token);
    localStorage.setItem(USER_DATA_KEY, JSON.stringify(userData));
  },
  
  // Clear authentication data
  clearAuthData: () => {
    localStorage.removeItem(AUTH_TOKEN_KEY);
    localStorage.removeItem(USER_DATA_KEY);
  },
  
  // Verify token with server
  verifyToken: async () => {
    const token = authAPI.getToken();
    if (!token) return false;
    
    try {
      const response = await fetch(`${authAPI.baseURL}/verify`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          // Update user data with fresh data from server
          authAPI.setAuthData(token, data.user);
          return true;
        }
      }
      
      // Token is invalid, clear auth data
      authAPI.clearAuthData();
      return false;
    } catch (error) {
      console.error('Token verification failed:', error);
      authAPI.clearAuthData();
      return false;
    }
  },
  
  // Logout user
  logout: async () => {
    const token = authAPI.getToken();
    
    try {
      // Call logout endpoint
      if (token) {
        await fetch(`${authAPI.baseURL}/logout`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          }
        });
      }
    } catch (error) {
      console.error('Logout API call failed:', error);
    } finally {
      // Always clear local auth data
      authAPI.clearAuthData();
    }
  }
};

// Higher-order component for protected routes
export const withAuth = (WrappedComponent) => {
  return (props) => {
    const [isAuthenticated, setIsAuthenticated] = React.useState(false);
    const [loading, setLoading] = React.useState(true);
    
    React.useEffect(() => {
      const checkAuth = async () => {
        const authenticated = await authAPI.verifyToken();
        setIsAuthenticated(authenticated);
        setLoading(false);
      };
      
      checkAuth();
    }, []);
    
    if (loading) {
      return <div>Loading...</div>;
    }
    
    if (!isAuthenticated) {
      // Redirect to login page
      window.location.href = '/login';
      return null;
    }
    
    return <WrappedComponent {...props} />;
  };
};

export default authAPI;
