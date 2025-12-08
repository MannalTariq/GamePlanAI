#!/usr/bin/env python3
"""
User Schema/Model for Authentication System
Defines the user data structure and database operations
"""

import sqlite3
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
import os

class UserSchema:
    """
    User schema and database operations for authentication system
    """
    
    def __init__(self, db_path: str = "database/users.db"):
        """
        Initialize the user schema with database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """
        Initialize the database and create users table if it doesn't exist
        """
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create index on email for faster lookups
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)
            ''')
            
            conn.commit()
            print(f"[DATABASE] User database initialized at: {self.db_path}")
    
    def create_user(self, name: str, email: str, password_hash: str) -> Optional[Dict[str, Any]]:
        """
        Create a new user in the database
        
        Args:
            name: User's full name
            email: User's email address (must be unique)
            password_hash: Hashed password
            
        Returns:
            User data dictionary if successful, None if email already exists
        """
        try:
            user_id = str(uuid.uuid4())
            created_at = datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO users (id, name, email, password_hash, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, name, email, password_hash, created_at))
                
                conn.commit()
                
                # Return the created user data
                return {
                    'id': user_id,
                    'name': name,
                    'email': email,
                    'created_at': created_at
                }
                
        except sqlite3.IntegrityError:
            # Email already exists
            print(f"[DATABASE] User with email {email} already exists")
            return None
        except Exception as e:
            print(f"[DATABASE] Error creating user: {e}")
            return None
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Get user by email address
        
        Args:
            email: User's email address
            
        Returns:
            User data dictionary if found, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, name, email, password_hash, created_at
                    FROM users WHERE email = ?
                ''', (email,))
                
                row = cursor.fetchone()
                
                if row:
                    return {
                        'id': row[0],
                        'name': row[1],
                        'email': row[2],
                        'password_hash': row[3],
                        'created_at': row[4]
                    }
                else:
                    return None
                    
        except Exception as e:
            print(f"[DATABASE] Error getting user by email: {e}")
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user by ID
        
        Args:
            user_id: User's unique ID
            
        Returns:
            User data dictionary if found, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, name, email, password_hash, created_at
                    FROM users WHERE id = ?
                ''', (user_id,))
                
                row = cursor.fetchone()
                
                if row:
                    return {
                        'id': row[0],
                        'name': row[1],
                        'email': row[2],
                        'password_hash': row[3],
                        'created_at': row[4]
                    }
                else:
                    return None
                    
        except Exception as e:
            print(f"[DATABASE] Error getting user by ID: {e}")
            return None
    
    def update_user(self, user_id: str, **kwargs) -> bool:
        """
        Update user information
        
        Args:
            user_id: User's unique ID
            **kwargs: Fields to update (name, email, password_hash)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not kwargs:
                return False
            
            # Build dynamic update query
            set_clauses = []
            values = []
            
            for field, value in kwargs.items():
                if field in ['name', 'email', 'password_hash']:
                    set_clauses.append(f"{field} = ?")
                    values.append(value)
            
            if not set_clauses:
                return False
            
            values.append(user_id)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = f"UPDATE users SET {', '.join(set_clauses)} WHERE id = ?"
                cursor.execute(query, values)
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"[DATABASE] Error updating user: {e}")
            return False
    
    def delete_user(self, user_id: str) -> bool:
        """
        Delete user by ID
        
        Args:
            user_id: User's unique ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"[DATABASE] Error deleting user: {e}")
            return False
    
    def get_all_users(self) -> list:
        """
        Get all users (for admin purposes)
        
        Returns:
            List of user dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, name, email, created_at
                    FROM users ORDER BY created_at DESC
                ''')
                
                rows = cursor.fetchall()
                
                return [
                    {
                        'id': row[0],
                        'name': row[1],
                        'email': row[2],
                        'created_at': row[3]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            print(f"[DATABASE] Error getting all users: {e}")
            return []
    
    def user_exists(self, email: str) -> bool:
        """
        Check if user exists by email
        
        Args:
            email: User's email address
            
        Returns:
            True if user exists, False otherwise
        """
        return self.get_user_by_email(email) is not None


# Example usage and testing
if __name__ == "__main__":
    # Initialize user schema
    user_schema = UserSchema()
    
    # Test creating a user
    test_user = user_schema.create_user(
        name="Test User",
        email="test@example.com",
        password_hash="hashed_password_here"
    )
    
    if test_user:
        print(f"[TEST] User created successfully: {test_user}")
        
        # Test getting user by email
        retrieved_user = user_schema.get_user_by_email("test@example.com")
        print(f"[TEST] Retrieved user: {retrieved_user}")
        
        # Test getting user by ID
        user_by_id = user_schema.get_user_by_id(test_user['id'])
        print(f"[TEST] User by ID: {user_by_id}")
        
        # Test updating user
        updated = user_schema.update_user(test_user['id'], name="Updated Test User")
        print(f"[TEST] User updated: {updated}")
        
        # Test getting all users
        all_users = user_schema.get_all_users()
        print(f"[TEST] All users: {all_users}")
        
        # Clean up test user
        deleted = user_schema.delete_user(test_user['id'])
        print(f"[TEST] User deleted: {deleted}")
    else:
        print("[TEST] Failed to create test user")
