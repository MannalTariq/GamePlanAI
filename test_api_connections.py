#!/usr/bin/env python3
"""
Test script to verify all API endpoints are working
"""

import requests
import time
import sys

# API endpoints
APIS = {
    'Auth API': 'http://localhost:5002/api/auth/health',
    'Free Kick API': 'http://localhost:5001/api/freekick/health',
    'Corner Kick API': 'http://localhost:5000/api/health',
}

def test_api(name, url):
    """Test a single API endpoint"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] {name}: OK - {data.get('status', 'unknown')}")
            return True
        else:
            print(f"[FAIL] {name}: Failed - Status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"[FAIL] {name}: Connection refused - Server not running")
        return False
    except requests.exceptions.Timeout:
        print(f"[FAIL] {name}: Timeout - Server not responding")
        return False
    except Exception as e:
        print(f"[FAIL] {name}: Error - {str(e)}")
        return False

def main():
    print("=" * 60)
    print("  GamePlan AI - API Connection Test")
    print("=" * 60)
    print()
    
    # Wait a bit for servers to start
    print("Waiting for servers to initialize...")
    time.sleep(3)
    print()
    
    results = []
    for name, url in APIS.items():
        results.append(test_api(name, url))
        time.sleep(1)
    
    print()
    print("=" * 60)
    if all(results):
        print("[SUCCESS] All API servers are running and responding!")
        print()
        print("Available endpoints:")
        print("  - Auth API:      http://localhost:5002")
        print("  - Free Kick API: http://localhost:5001")
        print("  - Corner API:    http://localhost:5000")
        print("  - Frontend:      http://localhost:3000")
        return 0
    else:
        print("[WARNING] Some API servers are not responding")
        print("   Please check that all servers are running")
        print("   Use start_all_servers.bat to start all servers")
        return 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)

