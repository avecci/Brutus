#!/bin/bash

# Directory for PID files
PIDDIR="/tmp/brutus"
mkdir -p $PIDDIR

# PID file path
PI_CLIENT_PID="$PIDDIR/pi_client.pid"

# Function to stop the client
stop_client() {
    if [ -f "$PI_CLIENT_PID" ]; then
        pid=$(cat "$PI_CLIENT_PID")
        echo "Stopping Raspberry Pi client (PID: $pid)..."
        kill "$pid" 2>/dev/null
        rm "$PI_CLIENT_PID"
    fi
    exit 0
}

# Set up trap for cleanup
trap stop_client SIGINT SIGTERM

# Check if client is already running
if [ -f "$PI_CLIENT_PID" ] && kill -0 "$(cat "$PI_CLIENT_PID")" 2>/dev/null; then
    echo "Raspberry Pi client is already running. Please stop it first."
    exit 1
fi

# Start Raspberry Pi client
echo "Starting Raspberry Pi client..."
cd ../raspberrypi_client || exit 1
poetry install
poetry run task start &
echo $! > "$PI_CLIENT_PID"

# Wait for the client process
wait
