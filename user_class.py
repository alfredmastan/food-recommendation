import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class User:
    def __init__(self, vector_size):
        self.vec = np.zeros(vector_size)
        self.dislike_step = 0.3
        self.like_step = 1
        self.liked_idx = set()
        self.disliked_idx = set()
        
    def like(self, recipe_idx, recipe_vector_means):
        self.vec += self.like_step * recipe_vector_means[recipe_idx]
        self.liked_idx.add(recipe_idx)

    def dislike(self, recipe_idx, recipe_vector_means):
        """Update the user vector by subtracting the dislike step from the disliked recipe vector."""
        self.vec -= self.dislike_step * recipe_vector_means[recipe_idx]
        self.disliked_idx.add(recipe_idx)

    def recommend(self, n, recipe_vector_means):
        """
        Update the recommendation by resetting the display indices.
        Returns a random selection of indices of the top n recipes based on cosine similarity.
        """
        if not self.vec.any():
            return np.random.choice(len(recipe_vector_means), n, replace=False)

        # Adjust the recipe vector means by excluding liked and disliked recipes
        exclude_indices = set(self.liked_idx).union(self.disliked_idx)
        if exclude_indices:
            adjusted_recipe_vector_means = np.delete(recipe_vector_means, list(exclude_indices), axis=0)
        
        sims = cosine_similarity(self.vec.reshape(1, -1), adjusted_recipe_vector_means)[0]

        arg_sorted_sims = sims.argsort()[::-1]  # Sort indices by similarity in descending order
        arg_pool = np.append(arg_sorted_sims[:n], arg_sorted_sims[n:(len(arg_sorted_sims)//2)])  # Get a pool of indices to choose from
        return np.random.choice(arg_sorted_sims[:n], n, replace=False)
        st.session_state["display_recipe_indices"] = np.random.choice(arg_sorted_sims[:n], n, replace=False)