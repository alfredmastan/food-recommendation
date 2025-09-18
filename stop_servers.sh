#!/bin/bash
#Activate anaconda env --> ./stop_servers.sh
pkill -f gunicorn &
pkill -f model_service.py &
pkill -f streamlit &

echo "Waiting for servers to stop..."
wait
echo "All servers stopped."

