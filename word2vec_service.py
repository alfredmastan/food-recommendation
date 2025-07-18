from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
import uvicorn
import numpy as np
import pandas as pd
import json
import boto3
import streamlit as st

def process_recipe_vector_means(data: pd.DataFrame, _model: Word2Vec) -> list:
    """Takes in a list of recipe ingredients, embeds it and calculate the mean"""
    recipe_vector_means = []

    for recipe_ingredients in data.ingredients:
        embedding_vec = [model.wv[ing] for ing in recipe_ingredients if ing in model.wv]
        mean_vec = np.mean(embedding_vec, axis=0) if embedding_vec else np.zeros(model.vector_size) * -1
        recipe_vector_means.append(mean_vec)
    
    return np.array(recipe_vector_means)

def connect_database():
    """Connect to the DynamoDB database."""
    dynamodb = boto3.resource("dynamodb",
                            aws_access_key_id=st.secrets.s3.AWS_ACCESS_KEY_ID,
                            aws_secret_access_key=st.secrets.s3.AWS_SECRET_ACCESS_KEY,
                            region_name=st.secrets.s3.AWS_DEFAULT_REGION)
    table = dynamodb.Table(st.secrets.s3.DB_NAME)
    return table

app = FastAPI(title="Food-Rec_Word2Vec_API")
data = pd.read_pickle("processed_cookbook.pkl")
model = Word2Vec.load("word2vec.model")
table = connect_database()
recipe_vector_means = process_recipe_vector_means(data, model)


@app.get("/recommend/{user_id}/{n}")
def recommend(user_id, n: int):
    """
    Update the recommendation by resetting the display indices.
    Returns a random selection of indices of the top n recipes based on cosine similarity.
    """
    # Get user configuration from DynamoDB
    user_config = table.get_item(Key={"user_id": int(user_id)})["Item"]
    liked_idx = set(map(int, user_config.get("liked_idx").keys()))
    disliked_idx = set(map(int, user_config.get("disliked_idx").keys()))

    exclude_indices = set(liked_idx).union(disliked_idx)

    # If there are no liked or disliked recipes, return random indices
    if not exclude_indices:
        rand = np.random.choice(len(recipe_vector_means), n, replace=False)
        return json.dumps(rand.tolist())
    
    user_vec = np.zeros(recipe_vector_means.shape[1]) + np.sum(recipe_vector_means[list(liked_idx)], axis=0) - np.sum(recipe_vector_means[list(disliked_idx)], axis=0)
    sims = cosine_similarity(user_vec.reshape(1, -1), recipe_vector_means)[0]

    # Create a DataFrame to hold the recipe IDs and their similarity scores to prevent shifting issues
    recipe_similarity = pd.DataFrame({
        "id": data.id,
        "similarity": sims
    })
    
    # Exclude liked and disliked recipes from the similarity scores
    if exclude_indices:
       recipe_similarity = recipe_similarity[~recipe_similarity.id.isin(exclude_indices)]

    recipe_similarity = recipe_similarity.sort_values(by="similarity", ascending=False) # Sort by similarity in descending order
    sorted_ids = recipe_similarity.id[::-1]  # Sort indices by similarity in descending order

    # arg_pool = np.append(arg_sorted_sims[:n], arg_sorted_sims[n:(len(arg_sorted_sims)//2)])  # Get a pool of indices to choose from
    rand = np.random.choice(sorted_ids[:n], n, replace=False)
    return json.dumps(rand.tolist())
    
# @app.get("/liked_idx/{user_id}")
# def get_liked_idx(user_id):
#     """Get the liked indices for a user."""
#     user_config = table.get_item(Key={"user_id": int(user_id)})["Item"]
#     return json.dumps(np.array(user_config.get("liked_idx"), dtype=np.int64))

# @app.get("/disliked_idx/{user_id}")
# def get_disliked_idx(user_id):
#     """Get the disliked indices for a user."""
#     user_config = table.get_item(Key={"user_id": int(user_id)})["Item"]
#     return json.dumps(np.array(user_config.get("disliked_idx"), dtype=np.int64))

# def calculate_user_vector(user_id):
#     user_vec = np.zeros_like(recipe_vector_means)
#     return user_vec

if __name__ == "__main__":
    uvicorn.run("word2vec_service:app", host="0.0.0.0", port=8000, reload=True)