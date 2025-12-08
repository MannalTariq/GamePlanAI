# Authentication System

This directory contains the authentication system for GamePlan AI, including user management, JWT-based authentication, and secure password handling.

## Files

- `user_schema.py` - User database schema and operations
- `auth_api.py` - Authentication API server with registration, login, and JWT endpoints
- `requirements.txt` - Python dependencies for the authentication system
- `users.db` - SQLite database file (created automatically)

## Features

### User Schema
- **ID**: Auto-generated UUID for each user
- **Name**: User's full name (string)
- **Email**: Unique email address (string)
- **Password**: Bcrypt hashed password (string)
- **Created At**: Timestamp with default current time

### Authentication API Endpoints

#### Base URL: `http://localhost:5002/api/auth`

1. **Health Check**
   - `GET /health` - Check API status

2. **User Registration**
   - `POST /register` - Create new user account
   - Body: `{ "name": "John Doe", "email": "john@example.com", "password": "password123" }`

3. **User Login**
   - `POST /login` - Authenticate user
   - Body: `{ "email": "john@example.com", "password": "password123" }`

4. **Token Verification**
   - `GET /verify` - Verify JWT token (requires Authorization header)
   - Header: `Authorization: Bearer <token>`

5. **User Profile**
   - `GET /profile` - Get user profile (requires Authorization header)
   - Header: `Authorization: Bearer <token>`

6. **Logout**
   - `POST /logout` - Logout user (requires Authorization header)
   - Header: `Authorization: Bearer <token>`

7. **Change Password**
   - `POST /change-password` - Change user password (requires Authorization header)
   - Body: `{ "current_password": "oldpass", "new_password": "newpass" }`
   - Header: `Authorization: Bearer <token>`

## Security Features

- **Password Hashing**: Uses bcrypt for secure password storage
- **JWT Authentication**: Secure token-based authentication
- **Input Validation**: Server-side validation for all inputs
- **CORS Support**: Enabled for frontend communication
- **Error Handling**: Comprehensive error handling and logging

## Installation

1. Install Python dependencies:
   ```bash
   cd database
   pip install -r requirements.txt
   ```

2. Run the authentication API server:
   ```bash
   python auth_api.py
   ```

The server will start on `http://localhost:5002`

## Frontend Integration

The frontend components (`Login.js` and `Signup.js`) have been updated to integrate with this authentication API:

- **Login Form**: Connects to `/api/auth/login` endpoint
- **Signup Form**: Connects to `/api/auth/register` endpoint
- **Token Storage**: JWT tokens stored in localStorage
- **Error Handling**: User-friendly error messages
- **Loading States**: Visual feedback during API calls

## Database

The system uses SQLite for simplicity and includes:
- Automatic database initialization
- User table with proper indexing
- UUID-based primary keys
- Timestamp tracking

## JWT Configuration

- **Algorithm**: HS256
- **Expiration**: 24 hours
- **Secret Key**: Configurable (change in production)

## Usage Example

```javascript
// Frontend usage
import { authAPI } from './utils/auth';

// Check if user is authenticated
if (authAPI.isAuthenticated()) {
  const userData = authAPI.getUserData();
  console.log('Logged in as:', userData.name);
}

// Logout
await authAPI.logout();
```

## Production Considerations

1. **Change Secret Key**: Update `SECRET_KEY` in `auth_api.py`
2. **Use Environment Variables**: Store sensitive configuration
3. **Database Security**: Consider PostgreSQL for production
4. **HTTPS**: Use SSL/TLS in production
5. **Rate Limiting**: Implement API rate limiting
6. **Logging**: Set up proper logging and monitoring
