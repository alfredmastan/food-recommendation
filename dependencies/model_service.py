from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
import uvicorn
import mlflow
import numpy as np
import yaml
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

# Helper Functions ============================================================
def normalization(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

def load_params():
    with open("../params.yaml") as f:
        params = yaml.safe_load(f)
    return params

# Preparation for Service ============================================================
# Load MLflow models
models = {}
mlflow.set_tracking_uri("http://127.0.0.1:8080")
models["fasttext_model"] = mlflow.pyfunc.load_model("models:/fasttext_model/latest")

# Create FastAPI app
app = FastAPI(title="Food-Rec_Model-API")

# Load params
params = load_params()

# Load recipe data
data = pd.read_pickle(params["model_pipeline"]["recipe_path"])

# Prepare ingredient vectors
ingredient_vectors = []
for ingredients in data.ingredients:
    # Get the vectors for the recipe ingredients and normalize them
    embedding_vecs = models["fasttext_model"].predict(ingredients)
    unit_vecs = np.array([vec/np.linalg.norm(vec) for vec in embedding_vecs])
    ingredient_vectors.append(unit_vecs)

ingredient_vector_means = np.array([np.mean(ing_vector, axis=0) for ing_vector in ingredient_vectors])

# Prepare title vectors
title_vectors = []
for title in data.recipe_title:
    # Get the vector for the recipe title and normalize it
    title_split = title.split()
    embedding_vecs = models["fasttext_model"].predict(title_split)
    unit_vecs = np.array([vec/np.linalg.norm(vec) for vec in embedding_vecs])
    title_vectors.append(unit_vecs)

title_vector_means = np.array([np.mean(ing_vector, axis=0) for ing_vector in title_vectors])


# API Calls ============================================================

@app.post("/recommend/")
async def recommend_search(query: list[str] = Query(default=[],
                                              description="List of ingredients to search for",
                                              examples=["chicken", "rice"])):
    """
    Search for recipes based on a query string.
    """

    # Get the vectors for the search query and normalize them
    query_vecs = models["fasttext_model"].predict(query)
    unit_query_vec = np.array([vec/np.linalg.norm(vec) for vec in query_vecs])

    # Calculate MaxSim for each recipe in the data
    # Might be slow for large amount of data
    ingredient_max_sim = np.array([])
    for ing_vector in ingredient_vectors:
        tokens_sim = unit_query_vec @ ing_vector.T

        a_best = tokens_sim.max(axis=1)
        b_best = tokens_sim.max(axis=0)
        
        score = 0.5 * (a_best.mean() + b_best.mean())
        ingredient_max_sim = np.append(ingredient_max_sim, score)

    # Calculate cosine similarity between the mean query vector and the mean ingredients vector
    mean_query_vec = unit_query_vec.mean(axis=0)
    ingredient_sim = mean_query_vec.reshape(1, -1) @ ingredient_vector_means.T
    title_sim = mean_query_vec.reshape(1, -1) @ title_vector_means.T

    score = (params["model_service"]["w_cosine"] * normalization(ingredient_sim[0]) + 
             params["model_service"]["w_maxsim"] * normalization(ingredient_max_sim) + 
             params["model_service"]["w_title"] * normalization(title_sim[0]))

    rec_idx = np.argsort(score)[-params["model_service"]["n_recs"]:][::-1]
    return json.dumps(rec_idx.tolist())

if __name__ == "__main__":
    uvicorn.run(app="model_service:app", host="0.0.0.0", port=8000, reload=True)
    # app name: model_service.py has to be the same as filename