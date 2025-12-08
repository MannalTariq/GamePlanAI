#!/bin/bash

echo "========================================"
echo "  GamePlan AI - Starting All Servers"
echo "========================================"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "[1/4] Starting Authentication API Server (Port 5002)..."
cd database
python3 auth_api.py &
AUTH_PID=$!
cd ..
sleep 3

echo "[2/4] Starting Free Kick API Server (Port 5001)..."
cd backend/hassaa/data
python3 freekick_api.py &
FREEKICK_PID=$!
cd ../../..
sleep 3

echo "[3/4] Starting Corner Kick API Server (Port 5000)..."
cd backend/hassaa/data
python3 api_server.py &
CORNER_PID=$!
cd ../../..
sleep 5

echo "[4/4] Starting React Frontend (Port 3000)..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "========================================"
echo "  All servers are starting..."
echo "========================================"
echo ""
echo "API Servers:"
echo "  - Auth API:      http://localhost:5002"
echo "  - Free Kick API: http://localhost:5001"
echo "  - Corner API:    http://localhost:5000"
echo ""
echo "Frontend:"
echo "  - React App:     http://localhost:3000"
echo ""
echo "Server PIDs:"
echo "  - Auth API:      $AUTH_PID"
echo "  - Free Kick API: $FREEKICK_PID"
echo "  - Corner API:    $CORNER_PID"
echo "  - Frontend:      $FRONTEND_PID"
echo ""
echo "To stop all servers, run: kill $AUTH_PID $FREEKICK_PID $CORNER_PID $FRONTEND_PID"
echo ""

# Wait for user interrupt
trap "echo 'Stopping all servers...'; kill $AUTH_PID $FREEKICK_PID $CORNER_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait

