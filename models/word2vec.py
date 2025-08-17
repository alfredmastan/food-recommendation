import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import json

class Word2VecModel(mlflow.pyfunc.PythonModel):
    def __init__(self, n_recs=10, like_step=0.8, dislike_step=0.3):
        self.n_recs = n_recs
        self.model = None
        self.like_step = like_step
        self.dislike_step = dislike_step
        self.idf = {}
        self.ing_embeddings = {}

    def load_context(self, context):
        self.model = Word2Vec.load(context.artifacts["model_path"])
        self.data = pd.read_pickle(context.artifacts["recipe_path"])

        # Apply IDF (Inverse Document Frequency)
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

    def predict(self, query: list[str]) -> str:
        """
        Predicts the top N similar recipes based on a search query.

        args:
            query: The search query string.

        returns:
            A JSON string containing the indices of the similar recipes.
        """
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
        score = 0.2 * cosine_sim_matrix[0] + 0.8 * max_sim
        # score = max_sim

        # rec_idx = np.sort(score)[::-1][:5]

        # Get the top N most similar recipe indices
        top_n_indices = np.argsort(score)[-self.n_recs:][::-1]
        return json.dumps(top_n_indices.tolist())

    # def predict(self, model_input: list[list[int]]) -> str:
    #     """
    #     Predicts the top N recommended recipes based on user preferences.

    #     args:
    #         model_input: A list containing two sets:
    #             - The first set contains indices of liked recipes.
    #             - The second set contains indices of disliked recipes.
    #     returns:
    #         A JSON string containing the indices of the recommended recipes.
    #     """

    #     liked_idx = set(model_input[0])
    #     disliked_idx = set(model_input[1])

    #     exclude_indices = liked_idx.union(disliked_idx)

    #     # If there are no liked or disliked recipes, return random indices
    #     if not exclude_indices:
    #         rand = np.random.choice(len(self.recipe_vector_means), self.n_recs, replace=False)
    #         return json.dumps(rand.tolist())

    #     # Calculate user vector based on liked and disliked recipes
    #     user_vec = (np.zeros(self.recipe_vector_means.shape[1]) 
    #                 + (self.like_step * np.sum(self.recipe_vector_means[list(liked_idx)], axis=0))
    #                 - (self.dislike_step * np.sum(self.recipe_vector_means[list(disliked_idx)], axis=0)))

    #     # Calculate cosine similarity between user vector and recipe vectors
    #     sims = cosine_similarity(user_vec.reshape(1, -1), self.recipe_vector_means)[0]

    #     # Create a DataFrame to hold the recipe IDs and their similarity scores to prevent shifting issues
    #     recipe_similarity = pd.DataFrame({
    #         "id": self.data.id,
    #         "similarity": sims
    #     })

    #     # Exclude liked and disliked recipes from the similarity scores
    #     if exclude_indices:
    #         recipe_similarity = recipe_similarity[~recipe_similarity.id.isin(exclude_indices)]

    #     recipe_similarity = recipe_similarity.sort_values(by="similarity", ascending=False) # Sort by similarity in descending order
    #     sorted_ids = recipe_similarity.id[:self.n_recs] # Grab top N indices by similarity

    #     rand = np.random.choice(sorted_ids[:self.n_recs], self.n_recs, replace=False)
    #     return json.dumps(rand.tolist())
