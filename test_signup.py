#!/usr/bin/env python3
"""
Test script to verify user signup functionality
"""

import requests
import json

def test_signup():
    """Test the signup endpoint"""
    
    print("===== TESTING USER SIGNUP =====")
    
    # Test data
    test_user = {
        "name": "Test User",
        "email": f"testuser{hash('test') % 10000}@example.com",  # Unique email
        "password": "testpass123"
    }
    
    print(f"\nAttempting to register user:")
    print(f"   Name: {test_user['name']}")
    print(f"   Email: {test_user['email']}")
    print(f"   Password: {'*' * len(test_user['password'])}")
    
    try:
        # Send signup request
        response = requests.post(
            "http://localhost:5002/api/auth/register",
            json=test_user,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"\nResponse Status: {response.status_code}")
        
        if response.status_code == 200 or response.status_code == 201:
            result = response.json()
            print("[SUCCESS] Signup successful!")
            print(f"   Success: {result.get('success', 'N/A')}")
            print(f"   User ID: {result.get('user', {}).get('id', 'N/A')}")
            print(f"   User Name: {result.get('user', {}).get('name', 'N/A')}")
            print(f"   User Email: {result.get('user', {}).get('email', 'N/A')}")
            print(f"   Token: {result.get('token', 'N/A')[:50]}..." if result.get('token') else "   Token: N/A")
            return True
        else:
            result = response.json()
            print(f"[FAILED] Signup failed!")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("[ERROR] Connection failed! Make sure the Auth API is running on port 5002")
        return False
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False

if __name__ == "__main__":
    success = test_signup()
    if success:
        print("\n[PASS] Signup test passed! The signup functionality is working.")
    else:
        print("\n[FAIL] Signup test failed! Check the error messages above.")

