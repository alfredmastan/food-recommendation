#!/bin/bash
#Activate anaconda env --> ./start_server.sh
cd dependencies/
mlflow server --host 127.0.0.1 --port 8080 &
python model_service.py &
streamlit run streamlit_web.py &
echo "All servers started."

