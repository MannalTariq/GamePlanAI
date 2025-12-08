# GamePlan AI - API Connections Summary

## Overview
All API endpoints have been connected and the project is fully integrated. This document summarizes all API connections and how to run the complete system.

## API Servers

### 1. Authentication API (Port 5002)
**Location:** `database/auth_api.py`

**Endpoints:**
- `GET /api/auth/health` - Health check
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `GET /api/auth/verify` - Verify JWT token
- `GET /api/auth/profile` - Get user profile
- `POST /api/auth/logout` - User logout
- `POST /api/auth/change-password` - Change password

**Connected Components:**
- `frontend/src/components/Login.js` - Login functionality
- `frontend/src/components/Signup.js` - Registration functionality
- `frontend/src/utils/auth.js` - Authentication utilities

### 2. Corner Kick API (Port 5000)
**Location:** `backend/hassaa/data/api_server.py`

**Endpoints:**
- `GET /api/health` - Health check
- `POST /api/optimize` - Optimize player positioning strategy
- `POST /api/simulate` - Generate corner kick simulation
- `POST /api/corner/left` - Set corner to left side
- `POST /api/corner/right` - Set corner to right side
- `GET /api/strategies` - List all saved strategies
- `GET /api/strategy/<filename>` - Get specific strategy details

**Connected Components:**
- `frontend/src/components/SIM-Service.js` - Simulation service (uses `/api/simulate`, `/api/corner/left`, `/api/corner/right`)
- `frontend/src/components/OPP-Service.js` - Optimize Player Positioning (uses `/api/optimize`)
- `frontend/src/components/ASP-Service.js` - Analyze Set Piece (uses `/api/strategies`, `/api/strategy/<filename>`)

### 3. Free Kick API (Port 5001)
**Location:** `backend/hassaa/data/freekick_api.py`

**Endpoints:**
- `GET /api/freekick/health` - Health check
- `POST /api/freekick/position` - Set freekick position
- `POST /api/freekick/simulate` - Generate freekick simulation

**Connected Components:**
- `frontend/src/components/SIM-Service.js` - Simulation service (uses `/api/freekick/position`, `/api/freekick/simulate`)

## Frontend Application (Port 3000)
**Location:** `frontend/`

**Routes:**
- `/login` - Login page
- `/signup` - Registration page
- `/` - Homepage (protected)
- `/services` - Services page (protected)
- `/analyze-set-piece` - Analyze Set Piece service (protected)
- `/optimize-player-positioning` - Optimize Player Positioning service (protected)
- `/simulate-strategies` - Simulate Strategies service (protected)

## How to Run the Complete System

### Option 1: Using the Startup Script (Recommended)
**Windows:**
```bash
.\start_all_servers.bat
```

**Linux/Mac:**
```bash
chmod +x start_all_servers.sh
./start_all_servers.sh
```

This will start all servers in separate terminal windows.

### Option 2: Manual Startup

1. **Start Authentication API:**
   ```bash
   cd database
   python auth_api.py
   ```

2. **Start Free Kick API:**
   ```bash
   cd backend/hassaa/data
   python freekick_api.py
   ```

3. **Start Corner Kick API:**
   ```bash
   cd backend/hassaa/data
   python api_server.py
   ```

4. **Start React Frontend:**
   ```bash
   cd frontend
   npm start
   ```

### Testing API Connections

Run the test script to verify all APIs are running:
```bash
python test_api_connections.py
```

## API Integration Details

### ASP Service (Analyze Set Piece)
- Fetches available strategies from `/api/strategies` when a set piece is selected
- Loads strategy details from `/api/strategy/<filename>` when analyzing
- Displays strategy analysis including:
  - Primary receiver information
  - Shot confidence
  - Tactical decision
  - Decision reasoning

### OPP Service (Optimize Player Positioning)
- Calls `/api/optimize` with default player positions
- Supports both Corner Kick and Free Kick set pieces
- Displays optimization results including:
  - Optimized player positions
  - Primary receiver
  - Shot confidence
  - Tactical insights

### SIM Service (Simulate Strategies)
- Uses `/api/simulate` for corner kicks
- Uses `/api/freekick/simulate` for free kicks
- Supports interactive player placement
- Generates animated simulations with ball trajectory and player movements

## Dependencies

### Backend APIs
- Flask
- Flask-CORS
- PyTorch
- PyTorch Geometric
- Other ML libraries (see `backend/hassaa/data/requirements.txt`)

### Authentication API
- Flask
- Flask-CORS
- bcrypt
- PyJWT
- (see `database/requirements.txt`)

### Frontend
- React
- React Router
- Chart.js
- Other UI libraries (see `frontend/package.json`)

## Notes

- All API servers have CORS enabled for React frontend
- Authentication is required for all protected routes
- JWT tokens are stored in localStorage
- API responses include detailed error messages for debugging
- All endpoints return JSON responses

## Troubleshooting

1. **Port already in use:** Make sure no other services are using ports 5000, 5001, 5002, or 3000
2. **API not responding:** Check that all Python dependencies are installed
3. **Frontend can't connect:** Verify all API servers are running and CORS is enabled
4. **Authentication issues:** Check that the database file exists and is accessible

