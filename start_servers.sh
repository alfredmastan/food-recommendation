#!/bin/bash
#Activate anaconda env --> ./start_server.sh
GUNICORN_CMD_ARGS="--timeout 600" mlflow server --backend-store-uri="mlflow/mlruns" \
                                                --artifacts-destination="mlflow/mlartifacts" \
                                                --host 127.0.0.1 \
                                                --port 8080 &
cd app/ 
streamlit run streamlit_web.py &
cd ../dependencies/ 
python model_service.py &

echo "Waiting for servers to start..."
wait
echo "All servers started."

