# GamePlan AI - Complete API Documentation

## Table of Contents
1. [Database Schema](#database-schema)
2. [Authentication API](#authentication-api)
3. [Corner Kick API](#corner-kick-api)
4. [Free Kick API](#free-kick-api)
5. [Frontend API Integration](#frontend-api-integration)
6. [Data Flow](#data-flow)

---

## Database Schema

### Location
- **Database File**: `database/database/users.db`
- **Schema File**: `database/user_schema.py`
- **Type**: SQLite Database

### Users Table Structure

```sql
CREATE TABLE users (
    id TEXT PRIMARY KEY,                    -- UUID string
    name TEXT NOT NULL,                      -- User's full name
    email TEXT UNIQUE NOT NULL,              -- Unique email address
    password_hash TEXT NOT NULL,             -- Bcrypt hashed password
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);
```

### Database Operations

#### UserSchema Class Methods

1. **`create_user(name, email, password_hash)`**
   - Creates a new user in the database
   - Returns: User dict with `id`, `name`, `email`, `created_at`
   - Raises: `IntegrityError` if email already exists

2. **`get_user_by_email(email)`**
   - Retrieves user by email address
   - Returns: User dict including `password_hash` or `None`

3. **`get_user_by_id(user_id)`**
   - Retrieves user by UUID
   - Returns: User dict or `None`

4. **`update_user(user_id, **kwargs)`**
   - Updates user fields (name, email, password_hash)
   - Returns: `True` if successful, `False` otherwise

5. **`delete_user(user_id)`**
   - Deletes user from database
   - Returns: `True` if successful, `False` otherwise

6. **`user_exists(email)`**
   - Checks if user exists
   - Returns: `True` or `False`

7. **`get_all_users()`**
   - Retrieves all users (admin function)
   - Returns: List of user dicts (without password_hash)

---

## Authentication API

### Base URL
```
http://localhost:5002/api/auth
```

### Server Details
- **Port**: 5002
- **File**: `database/auth_api.py`
- **Framework**: Flask with CORS enabled
- **Authentication**: JWT (JSON Web Tokens)
- **Password Hashing**: bcrypt

### Endpoints

#### 1. Health Check
```http
GET /api/auth/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Authentication API",
  "timestamp": "2025-01-15T10:30:00"
}
```

---

#### 2. User Registration
```http
POST /api/auth/register
Content-Type: application/json
```

**Request Body:**
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "securepassword123"
}
```

**Validation Rules:**
- Name: Required, non-empty string
- Email: Required, must contain '@' and '.', converted to lowercase
- Password: Required, minimum 6 characters

**Success Response (201):**
```json
{
  "success": true,
  "message": "User registered successfully",
  "user": {
    "id": "uuid-string",
    "name": "John Doe",
    "email": "john@example.com",
    "created_at": "2025-01-15T10:30:00"
  },
  "token": "jwt-token-string"
}
```

**Error Responses:**
- `400`: Missing fields or validation failed
- `409`: Email already exists
- `500`: Server error

---

#### 3. User Login
```http
POST /api/auth/login
Content-Type: application/json
```

**Request Body:**
```json
{
  "email": "john@example.com",
  "password": "securepassword123"
}
```

**Success Response (200):**
```json
{
  "success": true,
  "message": "Login successful",
  "user": {
    "id": "uuid-string",
    "name": "John Doe",
    "email": "john@example.com",
    "created_at": "2025-01-15T10:30:00"
  },
  "token": "jwt-token-string"
}
```

**Error Responses:**
- `400`: Missing email or password
- `401`: Invalid email or password
- `500`: Server error

---

#### 4. Verify Token
```http
GET /api/auth/verify
Authorization: Bearer <token>
```

**Success Response (200):**
```json
{
  "success": true,
  "user": {
    "id": "uuid-string",
    "name": "John Doe",
    "email": "john@example.com",
    "created_at": "2025-01-15T10:30:00"
  }
}
```

**Error Responses:**
- `401`: Missing, invalid, or expired token

---

#### 5. Get User Profile
```http
GET /api/auth/profile
Authorization: Bearer <token>
```

**Response:** Same as `/verify` endpoint

---

#### 6. Logout
```http
POST /api/auth/logout
Authorization: Bearer <token>
```

**Success Response (200):**
```json
{
  "success": true,
  "message": "Logout successful"
}
```

**Note:** Token is removed client-side. Server logs the logout event.

---

#### 7. Change Password
```http
POST /api/auth/change-password
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "current_password": "oldpassword",
  "new_password": "newpassword123"
}
```

**Success Response (200):**
```json
{
  "success": true,
  "message": "Password changed successfully"
}
```

**Error Responses:**
- `400`: Missing fields or new password too short
- `401`: Current password incorrect
- `404`: User not found
- `500`: Server error

---

### JWT Token Details

**Token Payload:**
```json
{
  "user_id": "uuid-string",
  "email": "john@example.com",
  "name": "John Doe",
  "exp": 1234567890,  // Expiration timestamp
  "iat": 1234567890   // Issued at timestamp
}
```

**Token Expiration:** 24 hours
**Algorithm:** HS256
**Secret Key:** Configured in `auth_api.py` (change in production!)

---

## Corner Kick API

### Base URL
```
http://localhost:5000/api
```

### Server Details
- **Port**: 5000
- **File**: `backend/hassaa/data/api_server.py`
- **Framework**: Flask with CORS enabled
- **ML Model**: GNN-based Strategy Maker (PyTorch)

### Endpoints

#### 1. Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "strategy_maker_ready": true,
  "timestamp": "2025-01-15T10:30:00"
}
```

---

#### 2. Optimize Strategy
```http
POST /api/optimize
Content-Type: application/json
```

**Request Body:**
```json
{
  "team": "Team A",
  "setPiece": "Corner Kick",
  "players": [
    {
      "id": 1,
      "x": 95,
      "y": 30,
      "team": "attacker"
    },
    {
      "id": 2,
      "x": 90,
      "y": 34,
      "team": "defender"
    }
  ],
  "cornerPosition": {
    "x": 105,
    "y": 0
  }
}
```

**Player Format:**
- `id`: Unique player identifier (integer)
- `x`: X position in meters (0-105)
- `y`: Y position in meters (0-68)
- `team`: `"attacker"`, `"defender"`, or `"keeper"`

**Success Response (200):**
```json
{
  "success": true,
  "strategy": {
    "primaryReceiver": {
      "playerId": 1,
      "score": 0.85,
      "position": {
        "x": 92.5,
        "y": 32.0
      }
    },
    "alternateReceivers": [
      {
        "playerId": 2,
        "score": 0.72,
        "position": {
          "x": 88.0,
          "y": 35.0
        }
      }
    ],
    "shotConfidence": 0.78,
    "tacticalDecision": "Near Post Attack",
    "successRate": 78
  },
  "players": [
    {
      "id": 1,
      "name": "Player 1",
      "role": "Attacker",
      "position": {
        "x": 95,
        "y": 30
      },
      "isPrimary": true,
      "isAlternate": false
    }
  ],
  "optimizationInsights": "Strategy optimized based on GNN predictions",
  "strategyFile": "generated_strategy_20250115_103000.json",
  "timestamp": "2025-01-15T10:30:00"
}
```

---

#### 3. Simulate Corner Kick
```http
POST /api/simulate
Content-Type: application/json
```

**Request Body:**
```json
{
  "players": [
    {
      "id": 1,
      "xPct": 90.5,
      "yPct": 44.1,
      "role": "red",
      "label": "Player 1"
    }
  ],
  "cornerPosition": {
    "x": 95.0,
    "y": 5.0
  },
  "goalPosition": {
    "x": 95.0,
    "y": 50.0
  },
  "setPiece": "Corner Kick"
}
```

**Note:** Frontend uses percentage coordinates (0-100), backend converts to meters (0-105, 0-68)

**Success Response (200):**
```json
{
  "success": true,
  "setPiece": "Corner Kick",
  "team": "Selected Team",
  "totalPlayers": 8,
  "positions": [...],
  "prediction": {
    "primaryPlayer": "Player 1 (85%)",
    "shotConfidence": 78,
    "tacticalDecision": "Near Post Attack",
    "decisionReason": "Primary receiver positioned optimally..."
  },
  "ballTrajectory": {
    "start": {
      "x": 90.5,
      "y": 7.4
    },
    "end": {
      "x": 88.2,
      "y": 47.1
    },
    "control": {
      "x": 89.3,
      "y": 20.0
    },
    "points": [
      {
        "x": 90.5,
        "y": 7.4,
        "progress": 0.0
      }
    ]
  },
  "playerMovements": [
    {
      "playerId": 1,
      "startPos": {
        "x": 90.5,
        "y": 44.1
      },
      "targetPos": {
        "x": 88.2,
        "y": 47.1
      },
      "role": "attacker",
      "movementSpeed": 0.8
    }
  ],
  "simulationData": {
    "shotAction": true,
    "shotTarget": {
      "x": 95.0,
      "y": 50.0
    }
  }
}
```

---

#### 4. Set Corner to Left Side
```http
POST /api/corner/left
Content-Type: application/json
```

**Request Body:** (Optional - can be empty)
```json
{}
```

**Success Response (200):**
```json
{
  "success": true,
  "message": "Corner set to left side",
  "cornerPosition": {
    "x": 0,
    "y": 0
  }
}
```

---

#### 5. Set Corner to Right Side
```http
POST /api/corner/right
Content-Type: application/json
```

**Request Body:** (Optional - can be empty)
```json
{}
```

**Success Response (200):**
```json
{
  "success": true,
  "message": "Corner set to right side",
  "cornerPosition": {
    "x": 105,
    "y": 0
  }
}
```

---

#### 6. List All Strategies
```http
GET /api/strategies
```

**Success Response (200):**
```json
{
  "success": true,
  "strategies": [
    {
      "filename": "generated_strategy_20250115_103000.json",
      "timestamp": "2025-01-15T10:30:00",
      "team": "Team A",
      "setPiece": "Corner Kick"
    }
  ]
}
```

---

#### 7. Get Specific Strategy
```http
GET /api/strategy/<filename>
```

**Example:**
```
GET /api/strategy/generated_strategy_20250115_103000.json
```

**Success Response (200):**
```json
{
  "success": true,
  "strategy": {
    "timestamp": "2025-01-15T10:30:00",
    "team": "Team A",
    "setPiece": "Corner Kick",
    "predictions": {
      "primary_receiver": {
        "player_id": 1,
        "score": 0.85,
        "position": {
          "x": 92.5,
          "y": 32.0
        }
      },
      "shot_confidence": 0.78,
      "tactical_decision": "Near Post Attack"
    }
  }
}
```

---

## Free Kick API

### Base URL
```
http://localhost:5001/api/freekick
```

### Server Details
- **Port**: 5001
- **File**: `backend/hassaa/data/freekick_api.py` (Note: File may not exist in current repo)
- **Framework**: Flask with CORS enabled

### Endpoints

#### 1. Health Check
```http
GET /api/freekick/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Free Kick API",
  "timestamp": "2025-01-15T10:30:00"
}
```

---

#### 2. Set Free Kick Position
```http
POST /api/freekick/position
Content-Type: application/json
```

**Request Body:**
```json
{
  "freekickPosition": {
    "x": 50.0,
    "y": 30.0
  }
}
```

**Success Response (200):**
```json
{
  "success": true,
  "message": "Free kick position set successfully",
  "freekickDecision": {
    "position": {
      "x": 50.0,
      "y": 30.0
    }
  }
}
```

---

#### 3. Simulate Free Kick
```http
POST /api/freekick/simulate
Content-Type: application/json
```

**Request Body:** (Similar to corner kick simulation)
```json
{
  "players": [...],
  "freekickPosition": {
    "x": 50.0,
    "y": 30.0
  },
  "goalPosition": {
    "x": 95.0,
    "y": 50.0
  }
}
```

**Response:** Similar structure to corner kick simulation

---

## Frontend API Integration

### Authentication Utilities
**File**: `frontend/src/utils/auth.js`

**Base URL**: `http://localhost:5002/api/auth`

**Methods:**

1. **`authAPI.getToken()`**
   - Retrieves JWT token from localStorage
   - Returns: Token string or `null`

2. **`authAPI.getUserData()`**
   - Retrieves user data from localStorage
   - Returns: User object or `null`

3. **`authAPI.isAuthenticated()`**
   - Checks if user has valid token
   - Returns: `true` or `false`

4. **`authAPI.setAuthData(token, userData)`**
   - Stores token and user data in localStorage

5. **`authAPI.clearAuthData()`**
   - Removes token and user data from localStorage

6. **`authAPI.verifyToken()`**
   - Verifies token with server
   - Returns: `Promise<boolean>`

7. **`authAPI.logout()`**
   - Calls logout endpoint and clears local data
   - Returns: `Promise<void>`

---

### Frontend Components & API Usage

#### 1. Login Component
**File**: `frontend/src/js files/Login/login-body.js`

**API Call:**
```javascript
POST http://localhost:5002/api/auth/login
Body: { email, password }
```

**On Success:**
- Stores token in `localStorage.authToken`
- Stores user data in `localStorage.userData`
- Redirects to homepage (`/`)

---

#### 2. Signup Component
**File**: `frontend/src/components/Signup.js`

**API Call:**
```javascript
POST http://localhost:5002/api/auth/register
Body: { name, email, password }
```

**On Success:**
- Stores token and user data
- Redirects to homepage

---

#### 3. OPP Service (Optimize Player Positioning)
**File**: `frontend/src/js files/OPP-Service/Obody.js`

**API Calls:**

**For Corner Kick:**
```javascript
POST http://localhost:5000/api/optimize
Body: {
  team: selectedTeam,
  setPiece: selectedSetPiece,
  players: defaultPlayers,
  cornerPosition: cornerPosition
}
```

**For Free Kick:**
```javascript
POST http://localhost:5001/api/freekick/position
Body: { freekickPosition: { x, y } }
```

---

#### 4. SIM Service (Simulate Strategies)
**File**: `frontend/src/js files/Sim-Service/Simbody.js`

**API Calls:**

**Corner Kick Simulation:**
```javascript
POST http://localhost:5000/api/simulate
Body: {
  players: [...],
  cornerPosition: { x, y },
  goalPosition: { x, y },
  setPiece: "Corner Kick"
}
```

**Set Corner Side:**
```javascript
POST http://localhost:5000/api/corner/left
POST http://localhost:5000/api/corner/right
```

**Free Kick Position:**
```javascript
POST http://localhost:5001/api/freekick/position
Body: { freekickPosition: { x, y } }
```

**Free Kick Simulation:**
```javascript
POST http://localhost:5001/api/freekick/simulate
Body: {
  players: [...],
  freekickPosition: { x, y },
  goalPosition: { x, y }
}
```

---

#### 5. ASP Service (Analyze Set Piece)
**File**: `frontend/src/components/ASP-Service.js`

**API Calls:**

**List Strategies:**
```javascript
GET http://localhost:5000/api/strategies
```

**Get Strategy Details:**
```javascript
GET http://localhost:5000/api/strategy/<filename>
```

---

## Data Flow

### Authentication Flow

```
1. User submits login form
   ↓
2. Frontend → POST /api/auth/login
   ↓
3. Backend validates credentials
   ↓
4. Backend generates JWT token
   ↓
5. Frontend stores token in localStorage
   ↓
6. Frontend includes token in Authorization header for protected routes
   ↓
7. Backend validates token on each request
```

### Strategy Optimization Flow

```
1. User selects team and set piece
   ↓
2. Frontend sends player positions → POST /api/optimize
   ↓
3. Backend converts frontend coordinates (percentage) to backend (meters)
   ↓
4. Strategy Maker (GNN) processes positions
   ↓
5. Backend generates strategy predictions
   ↓
6. Strategy saved to JSON file
   ↓
7. Frontend receives optimized positions and insights
   ↓
8. Frontend displays results with charts and visualizations
```

### Simulation Flow

```
1. User places players on field
   ↓
2. User clicks "Simulate"
   ↓
3. Frontend → POST /api/simulate with player positions
   ↓
4. Backend:
   - Converts coordinates
   - Generates strategy via GNN
   - Calculates ball trajectory (Bezier curve)
   - Calculates player movements
   ↓
5. Frontend receives:
   - Ball trajectory points
   - Player movement paths
   - Tactical predictions
   ↓
6. Frontend animates simulation
```

---

## Coordinate Systems

### Frontend (Percentage)
- **X-axis**: 0-100% (left to right)
- **Y-axis**: 0-100% (top to bottom)
- **Field Dimensions**: 100% × 100%

### Backend (Meters)
- **X-axis**: 0-105 meters (left to right)
- **Y-axis**: 0-68 meters (top to bottom)
- **Field Dimensions**: 105m × 68m (standard football field)

### Conversion Formulas

**Frontend → Backend:**
```javascript
x_meters = (xPct / 100) * 105
y_meters = (yPct / 100) * 68
```

**Backend → Frontend:**
```javascript
xPct = (x_meters / 105) * 100
yPct = (y_meters / 68) * 100
```

---

## Error Handling

### Common HTTP Status Codes

- **200**: Success
- **201**: Created (registration)
- **400**: Bad Request (validation errors)
- **401**: Unauthorized (invalid token/credentials)
- **404**: Not Found
- **409**: Conflict (email already exists)
- **500**: Internal Server Error
- **503**: Service Unavailable (Strategy Maker not initialized)

### Error Response Format

```json
{
  "error": "Error message description"
}
```

---

## Security Considerations

1. **Password Security**
   - Passwords hashed with bcrypt (salt rounds: default)
   - Never stored in plain text
   - Minimum 6 characters required

2. **JWT Tokens**
   - Tokens expire after 24 hours
   - Stored in localStorage (consider httpOnly cookies for production)
   - Secret key should be changed in production

3. **CORS**
   - Enabled for React frontend (localhost:3000)
   - Configure allowed origins in production

4. **Input Validation**
   - Server-side validation for all inputs
   - Email format validation
   - Password strength requirements

---

## Running the System

### Start All Servers

**Windows:**
```bash
.\start_all_servers.bat
```

**Linux/Mac:**
```bash
chmod +x start_all_servers.sh
./start_all_servers.sh
```

### Manual Startup

1. **Authentication API** (Port 5002):
   ```bash
   cd database
   python auth_api.py
   ```

2. **Corner Kick API** (Port 5000):
   ```bash
   cd backend/hassaa/data
   python api_server.py
   ```

3. **Free Kick API** (Port 5001):
   ```bash
   cd backend/hassaa/data
   python freekick_api.py
   ```

4. **React Frontend** (Port 3000):
   ```bash
   cd frontend
   npm start
   ```

### Test API Connections

```bash
python test_api_connections.py
```

---

## Dependencies

### Authentication API
- Flask==2.3.3
- Flask-CORS==4.0.0
- bcrypt==4.0.1
- PyJWT==2.8.0

### Corner Kick API
- Flask
- Flask-CORS
- PyTorch
- PyTorch Geometric
- NumPy
- Other ML libraries

### Frontend
- React 19.1.0
- React Router DOM 7.5.3
- Chart.js 4.4.9
- jsPDF 3.0.1
- html2canvas 1.4.1
- interactjs 1.10.27

---

## Notes

- All API servers have CORS enabled for React frontend
- Authentication is required for all protected routes
- JWT tokens are stored in localStorage
- API responses include detailed error messages for debugging
- All endpoints return JSON responses
- Frontend uses percentage coordinates, backend uses meters
- Strategy files are saved as JSON in `backend/hassaa/data/`

