#!/bin/bash

# Directory for PID files
PIDDIR="/tmp/brutus"
mkdir -p $PIDDIR

# PID file paths
BACKEND_PID="$PIDDIR/backend.pid"
FRONTEND_PID="$PIDDIR/frontend.pid"

# ...existing functions (is_running, is_port_in_use, stop_service, cleanup, wait_for_backend)...

# Set up trap for cleanup
trap cleanup SIGINT SIGTERM

# Check if services are already running
if is_port_in_use 8000; then
    echo "Backend is already running on port 8000. Please stop it first."
    exit 1
fi

if is_port_in_use 3000; then
    echo "Frontend is already running on port 3000. Please stop it first."
    exit 1
fi

# Start backend
echo "Starting backend server..."
cd backend || exit 1
poetry install
poetry run task start &
echo $! > "$BACKEND_PID"

# Wait for backend to be healthy
if ! wait_for_backend; then
    echo "Backend failed to start properly"
    cleanup
    exit 1
fi

# Start frontend
echo "Starting frontend server..."
cd ../frontend || exit 1
poetry install
poetry run task start &
echo $! > "$FRONTEND_PID"

echo "Services started successfully!"
echo "Use stop.sh to stop the services"

# Wait for signals
wait
