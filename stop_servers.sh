#!/bin/bash
#Activate anaconda env --> ./start_server.sh
pkill -f gunicorn &
pkill -f model_service.py &
pkill -f streamlit &

echo "All servers stopped."

