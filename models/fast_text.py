import numpy as np
import pandas as pd
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import json

class RecipeFastText(mlflow.pyfunc.PythonModel):
    def __init__(self, params):
        self.params = params

    def load_context(self, context):
        self.model = FastText.load(context.artifacts["model_path"])
        self.data = pd.read_pickle(context.artifacts["recipe_path"])

        # Apply IDF (Inverse Document Frequency)
        self.idf = {}
        self.ing_embeddings = {}

        for i, vocab in enumerate(self.model.wv.index_to_key):
            # Apply IDF
            n_recipe_contains_vocab = self.data.ingredients.apply(lambda x: vocab in x).sum()
            self.idf[vocab] = np.log(len(self.data.ingredients) / n_recipe_contains_vocab)

            # Apply IDF weight to Word2Vec embeddings
            self.ing_embeddings[vocab] = self.model.wv.vectors[i] * self.idf[vocab]
            self.ing_embeddings[vocab] = self.ing_embeddings[vocab] / np.linalg.norm(self.ing_embeddings[vocab])  # Normalize the vector with L2 norm

        # Calculate recipe vector means
        recipe_vector_means = []
        for recipe_ingredients in self.data.ingredients:
            embedding_vec = [self.ing_embeddings[ing] for ing in recipe_ingredients if ing in self.ing_embeddings]
            mean_vec = np.mean(embedding_vec, axis=0) if embedding_vec else np.zeros(self.model.vector_size) * -1
            recipe_vector_means.append(mean_vec)

        self.recipe_vector_means = np.array(recipe_vector_means)

    def get_score(self, query):
        """Calculates the similarity score for a given query."""
        # Get the vector for the search query
        query_vec = np.array([self.ing_embeddings[word] for word in query if word in self.ing_embeddings])
        if query_vec.size == 0:
            return json.dumps([])

        max_sim = np.array([])
        for recipe_ingredients in self.data.ingredients:
            recipe_embeddings = np.array([self.ing_embeddings[ing] for ing in recipe_ingredients if ing in self.ing_embeddings])

            if recipe_embeddings.size == 0:
                max_sim = np.append(max_sim, 0)
                continue

            tokens_sim = query_vec @ recipe_embeddings.T
            a_best = tokens_sim.max(axis=1)

            b_best = tokens_sim.max(axis=0)
            score = 0.5 * (a_best.mean() + b_best.mean())
            score = a_best.sum()
            max_sim = np.append(max_sim, score)

        mean_query_vec = query_vec.mean(axis=0)
        cosine_sim_matrix = mean_query_vec.reshape(1, -1) @ self.recipe_vector_means.T
        score = self.params.w_cosine * cosine_sim_matrix[0] + self.params.w_maxsim * max_sim
        return score

    def predict(self, query: list[str]) -> str:
        """Predicts the top N similar recipes based on a search query."""

        score = self.get_score(query)

        # Get the top N most similar recipe indices
        top_n_indices = np.argsort(score)[::-1][:self.params.n_recs]
        return json.dumps(top_n_indices.tolist())
