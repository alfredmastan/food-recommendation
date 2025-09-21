#!/bin/bash
#Activate anaconda env --> ./start_server.sh

# Wait for MLflow to be ready
echo "Starting MLflow server..."
GUNICORN_CMD_ARGS="--timeout 600" mlflow server --backend-store-uri="mlflow/mlruns" \
                                                --artifacts-destination="mlflow/mlartifacts" \
                                                --host 0.0.0.0 \
                                                --port 8080 &
echo "Waiting for MLflow server to be ready..."
while ! curl -s http://0.0.0.0:8080/health > /dev/null; do
    sleep 2
done
echo "MLflow server is ready!"

# Wait for model service to be ready
echo "Starting model service..." 
python service/model_service.py &
echo "Waiting for model service to be ready..."
while ! curl -s http://0.0.0.0:8000/docs > /dev/null; do
    sleep 2
done
echo "Model service is ready!"

# Wait for Streamlit to be ready
echo "Starting Streamlit app..."
streamlit run app.py &
echo "Waiting for Streamlit to be ready..."
while ! curl -s http://0.0.0.0:8501 > /dev/null; do
    sleep 2
done
echo "Streamlit app is ready!"

echo "All servers started and ready!"
wait