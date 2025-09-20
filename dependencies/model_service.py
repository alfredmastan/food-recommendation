from fastapi import FastAPI, Query
import uvicorn
import mlflow
import os
import json

# Preparation for Service ============================================================
# Load MLflow model
model = mlflow.pyfunc.load_model(os.path.join("../", "dependencies/model/"))

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

    if not query:
        return json.dumps([])
    
    rec_idx = model.predict(query)
    return json.dumps(rec_idx.tolist())

if __name__ == "__main__":
    uvicorn.run(app="model_service:app", host="0.0.0.0", port=8000, reload=True)
    # app name: model_service.py has to be the same as filename