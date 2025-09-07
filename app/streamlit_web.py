"""
Streamlit web app for Food Recipes Recommendation using Word2Vec model.
"""

#########################################################################################################################
#-- Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import yaml

#########################################################################################################################
#-- Page Configuration
st.set_page_config(layout="wide")

#########################################################################################################################
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

#########################################################################################################################
#-- Helper Functions

## Data and model loading functions
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load the main data and pre-trained Word2Vec model."""
    return pd.read_pickle(path)

## Recommendation functions
def update_recommendation(n_recipes):
    """Update the recommendation by resetting the display indices."""

    # Call the recommendation API
    response = requests.post(f"http://localhost:8000/recommend/", params={"query": st.session_state["ingredient_input"]})
    similarity_scores = np.asarray(json.loads(response.json()))
    recommended_indices = np.argsort(similarity_scores)[::-1][:n_recipes]
    
    st.toast(recommended_indices)

    if response.status_code != 200:
        st.toast("Failed to fetch recommendations. API call failed.")

    if json.loads(response.json()):
        st.session_state["displayed_recipe_indices"] = recommended_indices
        st.toast("RECOMMENDATION UPDATED")
    else:
        st.toast("Failed to fetch recommendations.")

#########################################################################################################################
#-- Database Management
def load_params():
    with open("../params.yaml") as f:
        params = yaml.safe_load(f)
    return params

#########################################################################################################################
#-- Preparing the content
params = load_params()
data = load_data(os.path.join("../", params["model_pipeline"]["recipe_path"])) # Load the main data

#########################################################################################################################
#-- Content Configurations
n_recipes = params["model_service"]["n_recs"]  # Number of recipes to recommend
n_cols = 4 # Number of recipes to display in each row
n_rows = np.ceil(n_recipes/n_cols).astype(int) # Number of rows to display

# Randomly select recipes to display initially
if "displayed_recipe_indices" not in st.session_state:
    st.session_state["displayed_recipe_indices"] = np.random.choice(len(data), n_recipes, replace=False) 

preloaded_data = data[~data.index.isin(list(st.session_state["display_excluded_indices"]))] # Preloaded data for the selected recipes

#########################################################################################################################
#-- Main Page Content
st.title("Food Recipes Recommendation")
st.write("This is a simple web app for recommending food recipes using Word2Vec model.")

st.button("Recommend Recipes", key="recommend_recipes", on_click=update_recommendation, args=(n_recipes, ), use_container_width=True)

ingredient_input = st.text_input("Enter your ingredients here...", key="chat_input")

if ingredient_input:
    ingredients = [ingredient.strip() for ingredient in ingredient_input.split(",") if ingredient]
    st.session_state["ingredient_input"] = ingredients
    update_recommendation(n_recipes)
    ingredients = [f":blue-badge[{ingredient.strip()}]" for ingredient in ingredients]
    ings_str = " ".join(ingredients)
    st.markdown("**Ingredients:**")
    st.markdown(ings_str)
    
st.subheader("Recommended Recipes")

## Create columns for displaying the recommended recipes
grid = st.columns(n_cols) * n_rows

for i, card in enumerate(grid):
    try:
        # Get the index of the recipe to display
        current_recipe = preloaded_data.iloc[st.session_state["displayed_recipe_indices"][i]]

        # Create a container for the recipe card
        container = card.container(border=True)
        with container:
            # Display the recipe image
            st.image(f"{preloaded_data.image_url.iloc[st.session_state["displayed_recipe_indices"][i]]}", use_container_width=True)
            
            # Show the similarity score
            # st.badge(f"Similarity: {sims[displayed_recipe_indices[i]]*100:.0f}%", color="red")

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
            ingredient_str = []
            for ingredient in current_recipe.ingredients:
                clean_str = ingredient.replace("_", " ").title()
                ingredient_str.append(f":blue-badge[{clean_str}]")
            st.markdown(" ".join(ingredient_str))

    except:
        # If there are not enough recipes to fill the columns, skip the remaining columns
        continue

#########################################################################################################################
#-- Development Controls
with st.sidebar:
    st.header("Development Controls")
    # Displayed Configurations
    st.header("Displayed Configurations")

    st.write(f"**Session Recipe Display Indices:**")
    st.write(f"{st.session_state["displayed_recipe_indices"]}")

    st.write("**Current Excluded Indices:**")
    if not st.session_state["display_excluded_indices"]:
        st.write("No recipes excluded yet.")
    else:
        st.write(f"{st.session_state["display_excluded_indices"]}")
    
    st.divider()