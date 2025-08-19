#!/bin/bash

# Sweep Dashboard Runner
# Starts both backend and frontend servers

echo "🚀 Starting Sweep Dashboard..."
echo ""

# Check if Python dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt
fi

# Check if Node dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node dependencies..."
    npm install
fi

# Function to kill both servers on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit
}

trap cleanup INT TERM

# Start backend server
echo "🐍 Starting backend server on http://localhost:8000..."
python backend.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start frontend server
echo "⚛️  Starting frontend server on http://localhost:3000..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Sweep Dashboard is running!"
echo ""
echo "📊 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID