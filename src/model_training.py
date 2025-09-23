# Import required libraries
from typing import Counter
import numpy as np
import pandas as pd
from collections import Counter

# Model
import mlflow
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity

# Logging
import os 
import yaml
import json

# Helper Functions
def load_params():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    return params

# MLFlow Model Wrapper
class RecipeFastText(mlflow.pyfunc.PythonModel):
    def __init__(self, params):
        self.params = params

    def load_context(self, context):
        self.model = FastText.load(context.artifacts["model_path"])
        self.data = pd.read_pickle(context.artifacts["data_path"])

        # Prepare IDF
        self.idf = {}
        ingredients_count = Counter(self.data.ingredients.explode())
        data_length = len(self.data)
        for vocab in self.model.wv.index_to_key:
            n_recipe_contains_vocab = ingredients_count.get(vocab, 0)
            self.idf[vocab] = np.log(data_length / (n_recipe_contains_vocab + 1e-9))

        # Prepare ingredient vectors
        self.ingredient_vectors = []
        for ingredients in self.data.ingredients:
            # Get the vectors for the recipe ingredients and normalize them
            embedding_vecs = [self.model.wv.get_vector(ing, norm=True) * self.idf.get(ing, 1)  for ing in ingredients]
            self.ingredient_vectors.append(embedding_vecs)

        self.ingredient_vector_means = np.array([np.mean(ing_vector, axis=0) for ing_vector in self.ingredient_vectors])

        # Prepare title vectors
        self.title_vectors = []
        for title in self.data.recipe_title:
            # Get the vector for the recipe title and normalize it
            title_split = title.split()
            embedding_vecs = [self.model.wv.get_vector(title, norm=True) for title in title_split]
            self.title_vectors.append(embedding_vecs)

        self.title_vector_means = np.array([np.mean(ing_vector, axis=0) for ing_vector in self.title_vectors])

    def predict(self, model_input: list[str]) -> list[float]:
        """Predicts the recipe similarity score based on the input string."""
        
        # Get the vectors for the search query and normalize them
        query_vecs = np.array([self.model.wv.get_vector(input, norm=True) for input in model_input])

        # Borrow IDF from top 3 most similar ingredients
        idf_top_weights = self.params["model_scoring"]["idf_top_weights"] # Weights for top 3 similar ingredients

        query_idfs = []
        for query in query_vecs:
            similar_ings = [ing for ing, _ in self.model.wv.similar_by_vector(query, topn=3)]
            ing_idf = np.sum([weight * self.idf.get(similar_ing, 1) for similar_ing, weight in zip(similar_ings, idf_top_weights)])
            query_idfs.append(ing_idf)

        # Apply IDF 
        query_vecs = np.array([(query_vec * query_idf) for query_vec, query_idf in zip(query_vecs, query_idfs)])

        # Calculate MaxSim for each recipe in the data
        # Might be slow for large amount of data
        ingredient_max_sim = np.array([])
        for ing_vector in self.ingredient_vectors:
            ing_vector = np.array(ing_vector)
            tokens_sim = np.array([self.model.wv.cosine_similarities(vec, ing_vector) for vec in query_vecs])

            a_best = tokens_sim.max(axis=1)
            b_best = tokens_sim.max(axis=0)
            
            score = 0.5 * (a_best.mean() + b_best.mean())
            ingredient_max_sim = np.append(ingredient_max_sim, score)

        # Calculate cosine similarity between the mean query vector and the mean ingredients vector
        mean_query_vec = query_vecs.mean(axis=0)
        ingredient_sim = self.model.wv.cosine_similarities(mean_query_vec, self.ingredient_vector_means)
        title_sim = self.model.wv.cosine_similarities(mean_query_vec, self.title_vector_means)

        score = (self.params["model_scoring"]["w_cosine"] * ingredient_sim +
                self.params["model_scoring"]["w_maxsim"] * ingredient_max_sim +
                self.params["model_scoring"]["w_title"] * title_sim)

        return score
    
def main():
    # Load data and params
    params = load_params()
    data = pd.read_pickle(params["model_pipeline"]["recipe_path"])

    # Train model
    print("Training FastText model...")
    fasttext_model = FastText(data.ingredients, **params["model_pipeline"]["fast_text"])
    print("Model training complete.")

    # Save model
    print("Saving model...")
    os.makedirs(os.path.dirname(params["model_pipeline"]["model_path"]), exist_ok=True)
    fasttext_model.save(params["model_pipeline"]["model_path"])

    # Set up MLflow tracking
    print("Logging to MLflow...")
    mlflow.set_tracking_uri("http://localhost:8080/")
    mlflow.set_experiment("FastText-Model")

    with mlflow.start_run():
        # Create MLFlow FastText Model
        fasttext_model = RecipeFastText(params["model_pipeline"])

        # Log params
        mlflow.log_params(params["model_pipeline"]["fast_text"])
        mlflow.log_params(params["model_pipeline"]["model_scoring"])

        # Log model
        model_info = mlflow.pyfunc.log_model(
            name="fasttext_model",
            python_model=fasttext_model,
            artifacts={"model_path": params["model_pipeline"]["model_path"],
                    "data_path": params["model_pipeline"]["recipe_path"]},
            pip_requirements=["gensim==4.3.3"]
        )

        # Log local URI
        local_uri = os.path.join("mlflow/mlartifacts", *model_info.artifact_path.split("/")[1:])
        mlflow.log_params({"local_uri": local_uri})
        
        # Log metadata
        metadata = {
            "server_uri": model_info.model_uri,
            "local_uri": local_uri,
            "run_id": model_info.run_id,
            "time_logged": model_info.utc_time_created,
            "model_params": params["model_pipeline"]
        }

        # Save metadata locally
        print("Saving metadata locally...")
        metadata["model_params"] = params["model_pipeline"]
        with open("training_metadata.json", "w") as f:
            f.write(json.dumps(metadata, indent=4))

        mlflow.log_artifact("training_metadata.json")


if __name__ == "__main__":
    print(f"{'='*20} Starting model training pipeline. {'='*20}")
    main()
    print(f"{'='*20} Model training pipeline complete. {'='*20}")
