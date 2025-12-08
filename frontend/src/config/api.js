// API Configuration - Uses environment variables with fallback to localhost for development
const getApiUrl = (envVar, defaultUrl) => {
  return process.env[envVar] || defaultUrl;
};

export const API_CONFIG = {
  AUTH_API: getApiUrl('REACT_APP_AUTH_API', 'http://localhost:5002/api/auth'),
  CORNER_API: getApiUrl('REACT_APP_CORNER_API', 'http://localhost:5000/api'),
  FREEKICK_API: getApiUrl('REACT_APP_FK_API', 'http://localhost:5001/api/freekick'),
};

export default API_CONFIG;

