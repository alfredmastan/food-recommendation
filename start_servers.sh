#!/bin/bash
#Activate anaconda env --> ./start_server.sh

mlflow server --host 127.0.0.1 --port 5000 &
python word2vec_service.py &
streamlit run streamlit_web.py &

echo "All servers started."

