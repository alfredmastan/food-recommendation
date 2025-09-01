from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
import uvicorn
import mlflow
import numpy as np
import yaml
import os
import pandas as pd
import json

# MLflow Load Model
# models = {}
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Load model during startup
#     global model
#     print("Loading model...")
#     mlflow.set_tracking_uri("http://127.0.0.1:8080")
#     models["fasttext_model"] = mlflow.pyfunc.load_model("models:/fasttext_model/latest")
#     print("Model loaded.")
#     yield

#     # Clean up models after shutdown
#     models.clear()
#     print("Shutting down model...")

# Preparation for Service ============================================================
# Load model metadata
with open("../registered_metadata.json", "r") as f:
    metadata = json.load(f)

# # Load params
# with open("../params.yaml", "r") as f:
#     params = yaml.safe_load(f)

# Load MLflow model
mlflow.set_tracking_uri("http://127.0.0.1:8080")
model = mlflow.pyfunc.load_model(os.path.join("../", metadata["local_uri"]))

# Create FastAPI app
app = FastAPI(title="Food-Recipe_Model-API")

# API Calls ============================================================
@app.post("/recommend/")
async def recommend_search(query: list[str] = Query(default=[],
                                              description="List of ingredients to search for",
                                              examples=["chicken", "rice"])):
    """
    Search for recipes based on a query string.
    """

    rec_idx = model.predict(query)
    return json.dumps(rec_idx.tolist())

if __name__ == "__main__":
    uvicorn.run(app="model_service:app", host="0.0.0.0", port=8000, reload=True)
    # app name: model_service.py has to be the same as filename