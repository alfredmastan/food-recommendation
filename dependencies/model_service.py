from fastapi import FastAPI, Query
import uvicorn
import mlflow

app = FastAPI(title="Food-Rec_Model-API")
model_path = "word2vec.model"
recipe_path = "processed_cookbook.pkl"

# MLflow Load Model
mlflow.set_tracking_uri("http://127.0.0.1:8080")
model = mlflow.pyfunc.load_model("models:/fasttext_model/latest")

# @app.get("/recommend/")
# def recommend(liked_idx: list[int] = Query(default=[],
#                                            description="List of liked recipe indices", 
#                                            example=[1, 5, 10]), 
#               disliked_idx: list[int] = Query(default=[],
#                                               description="List of disliked recipe indices", 
#                                               example=[2, 6, 11])):
#     """
#     Update the recommendation by resetting the display indices.
#     Returns a random selection of indices of the top n recipes based on cosine similarity.
#     """

#     return model.predict([liked_idx, disliked_idx])

@app.get("/recommend/")
def recommend_search(query: list[str] = Query(default=[],
                                              description="List of ingredients to search for",
                                              example=["chicken", "rice"])):
    """
    Search for recipes based on a query string.
    """
    return model.predict(query)


if __name__ == "__main__":
    uvicorn.run("word2vec_service:app", host="0.0.0.0", port=8000, reload=True)