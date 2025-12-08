#!/usr/bin/env python3
"""
Authentication API for User Registration and Login
Handles user signup, login, and JWT-based authentication
"""

import os
import sys
import bcrypt
import jwt
import json
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
import traceback

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import user schema
from user_schema import UserSchema

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'  # Change this in production
JWT_SECRET_KEY = app.config['SECRET_KEY']
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24

# Initialize user schema
user_schema = UserSchema()

def hash_password(password: str) -> str:
    """
    Hash password using bcrypt
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password string
    """
    try:
        # Generate salt and hash password
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    except Exception as e:
        print(f"[ERROR] Password hashing failed: {e}")
        raise

def verify_password(password: str, hashed_password: str) -> bool:
    """
    Verify password against hash
    
    Args:
        password: Plain text password
        hashed_password: Hashed password from database
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception as e:
        print(f"[ERROR] Password verification failed: {e}")
        return False

def generate_jwt_token(user_data: dict) -> str:
    """
    Generate JWT token for user
    
    Args:
        user_data: User data dictionary
        
    Returns:
        JWT token string
    """
    try:
        payload = {
            'user_id': user_data['id'],
            'email': user_data['email'],
            'name': user_data['name'],
            'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
            'iat': datetime.utcnow()
        }
        
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return token
    except Exception as e:
        print(f"[ERROR] JWT token generation failed: {e}")
        raise

def verify_jwt_token(token: str) -> dict:
    """
    Verify and decode JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        print("[ERROR] JWT token has expired")
        return None
    except jwt.InvalidTokenError:
        print("[ERROR] Invalid JWT token")
        return None
    except Exception as e:
        print(f"[ERROR] JWT token verification failed: {e}")
        return None

def token_required(f):
    """
    Decorator to require JWT token for protected routes
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Check for token in Authorization header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]  # Bearer <token>
            except IndexError:
                return jsonify({'error': 'Invalid authorization header format'}), 401
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            # Verify token
            payload = verify_jwt_token(token)
            if not payload:
                return jsonify({'error': 'Invalid or expired token'}), 401
            
            # Add user info to request context
            request.current_user = payload
            return f(*args, **kwargs)
            
        except Exception as e:
            print(f"[ERROR] Token verification error: {e}")
            return jsonify({'error': 'Token verification failed'}), 401
    
    return decorated

@app.route('/api/auth/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Authentication API',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/auth/register', methods=['POST'])
def register():
    """
    User registration endpoint
    
    Expected JSON body:
    {
        "name": "John Doe",
        "email": "john@example.com",
        "password": "securepassword123"
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        name = data.get('name', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        # Validation
        if not name:
            return jsonify({'error': 'Name is required'}), 400
        
        if not email:
            return jsonify({'error': 'Email is required'}), 400
        
        if not password:
            return jsonify({'error': 'Password is required'}), 400
        
        # Basic email validation
        if '@' not in email or '.' not in email:
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Password strength validation
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters long'}), 400
        
        # Check if user already exists
        if user_schema.user_exists(email):
            return jsonify({'error': 'User with this email already exists'}), 409
        
        # Hash password
        hashed_password = hash_password(password)
        
        # Create user
        user_data = user_schema.create_user(name, email, hashed_password)
        
        if not user_data:
            return jsonify({'error': 'Failed to create user'}), 500
        
        # Generate JWT token
        token = generate_jwt_token(user_data)
        
        print(f"[AUTH] User registered successfully: {email}")
        
        return jsonify({
            'success': True,
            'message': 'User registered successfully',
            'user': {
                'id': user_data['id'],
                'name': user_data['name'],
                'email': user_data['email'],
                'created_at': user_data['created_at']
            },
            'token': token
        }), 201
        
    except Exception as e:
        print(f"[ERROR] Registration failed: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """
    User login endpoint
    
    Expected JSON body:
    {
        "email": "john@example.com",
        "password": "securepassword123"
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        # Validation
        if not email:
            return jsonify({'error': 'Email is required'}), 400
        
        if not password:
            return jsonify({'error': 'Password is required'}), 400
        
        # Get user from database
        user_data = user_schema.get_user_by_email(email)
        
        if not user_data:
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Verify password
        if not verify_password(password, user_data['password_hash']):
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Generate JWT token
        token = generate_jwt_token(user_data)
        
        print(f"[AUTH] User logged in successfully: {email}")
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'id': user_data['id'],
                'name': user_data['name'],
                'email': user_data['email'],
                'created_at': user_data['created_at']
            },
            'token': token
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Login failed: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/auth/verify', methods=['GET'])
@token_required
def verify_token():
    """
    Verify JWT token and return user information
    """
    try:
        user_info = request.current_user
        
        # Get fresh user data from database
        user_data = user_schema.get_user_by_id(user_info['user_id'])
        
        if not user_data:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'success': True,
            'user': {
                'id': user_data['id'],
                'name': user_data['name'],
                'email': user_data['email'],
                'created_at': user_data['created_at']
            }
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Token verification failed: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Token verification failed'}), 500

@app.route('/api/auth/profile', methods=['GET'])
@token_required
def get_profile():
    """
    Get user profile information
    """
    try:
        user_info = request.current_user
        
        # Get fresh user data from database
        user_data = user_schema.get_user_by_id(user_info['user_id'])
        
        if not user_data:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'success': True,
            'user': {
                'id': user_data['id'],
                'name': user_data['name'],
                'email': user_data['email'],
                'created_at': user_data['created_at']
            }
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Profile retrieval failed: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Profile retrieval failed'}), 500

@app.route('/api/auth/logout', methods=['POST'])
@token_required
def logout():
    """
    Logout endpoint (client-side token removal)
    """
    try:
        user_info = request.current_user
        
        print(f"[AUTH] User logged out: {user_info['email']}")
        
        return jsonify({
            'success': True,
            'message': 'Logout successful'
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Logout failed: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Logout failed'}), 500

@app.route('/api/auth/change-password', methods=['POST'])
@token_required
def change_password():
    """
    Change user password
    
    Expected JSON body:
    {
        "current_password": "oldpassword",
        "new_password": "newpassword123"
    }
    """
    try:
        data = request.get_json()
        user_info = request.current_user
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        current_password = data.get('current_password', '')
        new_password = data.get('new_password', '')
        
        # Validation
        if not current_password:
            return jsonify({'error': 'Current password is required'}), 400
        
        if not new_password:
            return jsonify({'error': 'New password is required'}), 400
        
        if len(new_password) < 6:
            return jsonify({'error': 'New password must be at least 6 characters long'}), 400
        
        # Get user data
        user_data = user_schema.get_user_by_id(user_info['user_id'])
        
        if not user_data:
            return jsonify({'error': 'User not found'}), 404
        
        # Verify current password
        if not verify_password(current_password, user_data['password_hash']):
            return jsonify({'error': 'Current password is incorrect'}), 401
        
        # Hash new password
        new_hashed_password = hash_password(new_password)
        
        # Update password
        success = user_schema.update_user(user_info['user_id'], password_hash=new_hashed_password)
        
        if not success:
            return jsonify({'error': 'Failed to update password'}), 500
        
        print(f"[AUTH] Password changed successfully for user: {user_info['email']}")
        
        return jsonify({
            'success': True,
            'message': 'Password changed successfully'
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Password change failed: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Password change failed'}), 500

if __name__ == '__main__':
    print("="*70)
    print("  AUTHENTICATION API SERVER")
    print("="*70)
    print("[START] Starting Flask API server on http://localhost:5002")
    print("[CORS] CORS enabled for React frontend")
    print("\nAvailable endpoints:")
    print("  GET  /api/auth/health          - Health check")
    print("  POST /api/auth/register        - User registration")
    print("  POST /api/auth/login           - User login")
    print("  GET  /api/auth/verify          - Verify JWT token")
    print("  GET  /api/auth/profile         - Get user profile")
    print("  POST /api/auth/logout          - User logout")
    print("  POST /api/auth/change-password - Change password")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5002)
