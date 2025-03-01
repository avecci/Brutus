#!/bin/bash

# Find PIDs for processes listening on ports 3000 and 8000
BACKEND_PID=$(netstat -tulpn 2>/dev/null | grep "0.0.0.0:8000" | awk '{print $7}' | cut -d'/' -f1)
FRONTEND_PID=$(netstat -tulpn 2>/dev/null | grep "0.0.0.0:3000" | awk '{print $7}' | cut -d'/' -f1)
PIDDIR="/tmp/brutus"
PI_CLIENT_PID="$PIDDIR/pi_client.pid"

# Function to stop a service by PID
stop_service() {
    if [ ! -z "$1" ]; then
        echo "Stopping process $1..."
        kill "$1" 2>/dev/null
    fi
}

echo "Stopping Brutus services..."

if [ ! -z "$BACKEND_PID" ]; then
    stop_service "$BACKEND_PID"
    echo "Backend stopped (PID: $BACKEND_PID)"
else
    echo "Backend not running."
fi

if [ ! -z "$FRONTEND_PID" ]; then
    stop_service "$FRONTEND_PID"
    echo "Frontend stopped (PID: $FRONTEND_PID)"
else
    echo "Frontend not running."
fi

if [ -f "$PI_CLIENT_PID" ]; then
    pid=$(cat "$PI_CLIENT_PID")
    echo "Stopping Raspberry Pi client (PID: $pid)..."
    kill "$pid" 2>/dev/null
    rm "$PI_CLIENT_PID"
    echo "Raspberry Pi client stopped."
fi

echo "All services stopped."
