#!/bin/bash

# Directory for PID files
PIDDIR="/tmp/brutus"
mkdir -p $PIDDIR

BACKEND_PID="$PIDDIR/backend.pid"
FRONTEND_PID="$PIDDIR/frontend.pid"

# Function to check if a process is running
is_running() {
    [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null
}

# Function to check if a port is in use
is_port_in_use() {
    netstat -tulpn 2>/dev/null | grep "0.0.0.0:$1" >/dev/null
    return $?
}

# Function to stop a service
stop_service() {
    if [ -f "$1" ]; then
        pid=$(cat "$1")
        echo "Stopping process $pid..."
        kill "$pid" 2>/dev/null
        rm "$1"
    fi
}

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    stop_service "$BACKEND_PID"
    stop_service "$FRONTEND_PID"
    exit 0
}

# Function to wait for backend health
wait_for_backend() {
    local max_attempts=15
    local attempt=1
    local backend_url="http://localhost:8000/health"

    echo "Waiting for backend to be healthy..."

    while [ $attempt -le $max_attempts ]; do
        if curl -s "$backend_url" | grep -q '"status":"Healthy"'; then
            echo "Backend is healthy!"
            return 0
        fi
        echo "Attempt $attempt/$max_attempts: Backend not ready yet..."
        sleep 3
        ((attempt++))
    done

    echo "Backend failed to become healthy after $max_attempts attempts"
    return 1
}

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
poetry run task start &
echo $! > "$FRONTEND_PID"

echo "Services started successfully!"
echo "Use stop.sh to stop the services."

# Wait for signals
wait
