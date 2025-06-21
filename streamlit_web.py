"""
Streamlit web app for Food Recipes Recommendation using Word2Vec model.
"""

#-- Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec


#-- Page Configuration
st.set_page_config(layout="wide")

#-- Custom CSS to style hyperlinks
st.markdown("""
<style>
/* Remove underline from all links and change color to white */
a {
    text-decoration: none !important;
    color: white !important; # change to black if dark mode is enabled
}

/* Add hover effect to show it's still clickable */
a:hover {
    text-decoration: none !important;
    color: #f0f0f0 !important;
    opacity: 0.8;
}
""", unsafe_allow_html=True)

#-- Load Required Data and Models
data = pd.read_pickle("processed_cookbook.pkl") # Load the main data
model = Word2Vec.load("word2vec.model") # Load the pre-trained Word2Vec model

#-- Helper Functions
def reset_recommendation(n):
    """
    Initialize or Overwrite recipes to display
    """
    # st.session_state["display_recipe_indices"] = np.random.choice(len(data), n, replace=False)
    st.session_state["display_recipe_indices"] = user.recommend(n)  # Get recommended indices from the user vector

def liked(recipe_idx, idx):
    """
    Handle the like button click event.
    """
    st.toast("Liked!")  # Display a toast message when the like button is clicked
    user.like(recipes_vector_mean[recipe_idx])  # Update user's vector with the liked recipe vector
    user.exclude_idx.add(recipe_idx)  # Add the recipe index to the exclude list
    st.session_state["display_recipe_indices"] = np.delete(st.session_state["display_recipe_indices"], idx)  # Remove the liked recipe from the displayed recommendations

def disliked(recipe_idx, idx):
    """
    Handle the dislike button click event.
    """
    st.toast("Disliked!")  # Display a toast message when the dislike button is clicked
    user.dislike(recipes_vector_mean[recipe_idx])
    user.exclude_idx.add(recipe_idx)  # Add the recipe index to the exclude list
    st.session_state["display_recipe_indices"] = np.delete(st.session_state["display_recipe_indices"], idx)  # Remove the disliked recipe from the displayed recommendations


#-- Recommendation Functionality
from sklearn.metrics.pairwise import cosine_similarity

def get_recipe_vector_mean(ingredients: list, model: Word2Vec) -> list:
    """
    Takes in a list of recipe ingredients, embeds it and calculate the mean
    """
    vector_embedding = [model.wv[ing] for ing in ingredients if ing in model.wv]
    return np.mean(vector_embedding, axis=0) if vector_embedding else np.zeros(model.vector_size) * -1
    
recipes_vector_mean = [get_recipe_vector_mean(recipe, model) for recipe in data.ingredients]
recipes_vector_mean = np.array(recipes_vector_mean)

class User:
    def __init__(self, vector_dim):
        self.vec = np.zeros(vector_dim)
        self.dislike_step = 0.3
        self.like_step = 1
        self.exclude_idx = set()

    def like(self, recipe_vector_mean):
        self.vec += self.like_step * recipe_vector_mean

    def dislike(self, recipe_vector_mean):
        self.vec -= self.dislike_step * recipe_vector_mean

    def recommend(self, n):
        if not self.vec.any():
            display_recipe_indices = np.random.choice(len(recipes_vector_mean), n, replace=False)
            return display_recipe_indices

        sims = cosine_similarity(self.vec.reshape(1, -1), recipes_vector_mean)[0]

        if self.exclude_idx:
            sims[list(self.exclude_idx)] = -1 # Exclude recipes shown

        arg_sorted_sims = sims.argsort()[::-1]  # Sort indices by similarity in descending order
        arg_pool = np.append(arg_sorted_sims[:(n*2)], arg_sorted_sims[(-n*2):])  # Get a pool of indices to choose from
        return np.random.choice(arg_pool, n, replace=False)

#-- User Session Management
if "user" not in st.session_state:
    st.toast("USER RESET")
    st.session_state["user"] = User(model.vector_size)  # Initialize user with vector dimension

user = st.session_state["user"]

#-- Content Configurations
n_recipe = 20  # Number of recipes to recommend
n_cols = 4 # Number of recipes to display in each row
n_rows = np.ceil(n_recipe/n_cols).astype(int) # Number of rows to display

if "display_recipe_indices" not in st.session_state:
    st.toast("RECOMMENDATION RESET")
    st.session_state["display_recipe_indices"] = user.recommend(n_recipe) 

display_recipe_indices = st.session_state["display_recipe_indices"]  # Get the recommended indices from the session state
preloaded_data = data.iloc[display_recipe_indices].reset_index(drop=True)  # Preloaded data for the selected recipes

#-- Main Page Content
if not st.user.is_logged_in:
    if st.button("Log in"):
        st.login()
else:
    if st.button("Log out"):
        st.logout()
    st.write(f"Hello, {st.user.name}!")
    
st.title("Food Recipes Recommendation")
st.write("This is a simple web app for recommending food recipes using Word2Vec model.")

def print_user_vector():
    """
    Print the user vector in a readable format.
    """
    st.toast(user.vec)


st.button("User Vector", key="user_vector", on_click=print_user_vector, help="Click to view your user vector")
st.button("Recommend Recipes", key="recommend_recipes", on_click=reset_recommendation, args=(n_recipe,))


st.subheader("Recommended Recipes")

## Create columns for displaying the recommended recipes
grid = st.columns(n_cols) * n_rows

for i, card in enumerate(grid):
    try:
        # Get the index of the recipe to display
        current_recipe = preloaded_data.iloc[i]

        # Create a container for the recipe card
        container = card.container(border=True)
        with container:
            # Display the recipe image
            st.image(f"{preloaded_data.image_url.iloc[i]}", use_container_width=True)
            sims = cosine_similarity(user.vec.reshape(1, -1), recipes_vector_mean)[0]
            st.markdown(sims[display_recipe_indices[i]])
            # Recipe title
            st.subheader(f"[{current_recipe.recipe_title}]({current_recipe.recipe_url})")

            # Total Time
            hours = int(current_recipe.total_time//60)
            minutes = int(current_recipe.total_time%60)
            time_str = ""
            if hours:
                time_str += f"**{hours} hours**"
            if minutes:
                time_str += f" **{minutes} minutes**"

            st.markdown(f"Total Time: {time_str}")

            # Ingredients
            st.markdown("**Ingredients:**")
            str = []
            for ingredient in current_recipe.ingredients:
                clean_str = ingredient.replace("_", " ").title()
                str.append(f":blue-badge[{clean_str}]")
            st.markdown(" ".join(str))

            # Like/Dislike
            button_cols = st.columns(2, gap="small")  # Create two columns for like/dislike buttons
            button_cols[0].button("", key=f"like_{i}",icon=":material/thumb_up:", use_container_width=True, on_click=liked, args=(display_recipe_indices[i], i))
            button_cols[1].button("", key=f"dislike_{i}", icon=":material/thumb_down:", use_container_width=True, on_click=disliked, args=(display_recipe_indices[i], i))

    except:
        # If there are not enough recipes to fill the columns, skip the remaining columns
        continue