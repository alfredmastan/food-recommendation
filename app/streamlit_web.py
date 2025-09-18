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
import re 

#########################################################################################################################
#-- Page Configuration
st.set_page_config(layout="wide")
dev = False; # Development mode
#########################################################################################################################
#-- Custom CSS to style hyperlinks
# Remove underline from all links and change color to white
st.markdown("""
<style>
/* Remove underline from all links and change color to white */
a {
    text-decoration: none !important;
    color: white !important; # change to black if dark mode is enabled
}
</style>
""", unsafe_allow_html=True)

# Hover effect for links
st.markdown("""
<style>
/* Add hover effect to show it's still clickable */
a:hover {
    text-decoration: none !important;
    color: #f0f0f0 !important;
    opacity: 0.8;
}
</style>
""", unsafe_allow_html=True)

# Adjust metric font size
st.markdown("""
<style>
[data-testid="stMetricValue"] {
    font-size: 100%;
}
</style>
""",
    unsafe_allow_html=True,
)

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
    response = requests.post(f"http://localhost:8000/recommend/", params={"query": st.session_state.get("ingredient_input", [])})
    
    if response.status_code != 200:
        st.toast("Failed to fetch recommendations. API call failed.")
        return
    
    # Process the similarity scores from the response 
    similarity_scores = np.asarray(json.loads(response.json()))

    # Apply filters if any (Hard filter)
    mask = set(range(len(data))) # Indexes of all recipes
    for filter, conditions in nutrition_filter_map.items():
        if not st.session_state.get(filter, False):
            continue

        for condition in conditions:
            col, threshold, direction = condition
            if direction == "lower":
                mask = mask.intersection(data[data[col] <= threshold].index.to_numpy())
            else:
                mask = mask.intersection(data[data[col] >= threshold].index.to_numpy())

    mask = list(mask) # Convert to list for indexing
    
    # Sort the similarity scores, if none, show random recipes from filtered recipes
    if len(similarity_scores) != 0:
        recommended_indices = np.argsort(similarity_scores)[::-1]
        st.session_state["displayed_similarity_scores"] = np.sort(similarity_scores[recommended_indices])[::-1][:n_recipes] # Update the displayed similarity scores
    else:
        recommended_indices = np.random.choice(len(data), len(data), replace=False)
        st.session_state["displayed_similarity_scores"] = np.array([0]*n_recipes) # Set the displayed similarity scores to zeros

    # Apply the filters to the recommended indices
    recommended_indices = recommended_indices[np.isin(recommended_indices, mask)]

    # Select the top N recipes
    st.session_state["displayed_recipe_indices"] = recommended_indices[:n_recipes]

    if dev:
        st.toast(f"Filtered recipe length: {len(mask)}")
        st.toast(st.session_state["displayed_recipe_indices"])
        st.toast(recommended_indices)
        st.toast("RECOMMENDATION UPDATED")

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

# Define the available filters
nutrition_filter_map = {"High Protein": [["protein", 25, "higher"]],
           "High Fiber": [["fiber", 15, "higher"]],
           "Low Carb": [["carbohydrates", 50, "lower"]],
           "Low Calorie": [["calories", 500, "lower"]],
           "Low Fat": [["fat", 20, "lower"]],
           "Quick and Easy": [["num_steps", 10, "lower"], ["total_time", 30, "lower"], ["num_ingredients", 5, "lower"]]} # Special case, handled separately

# Nutrition facts display mapping
nutrition_facts_map = {
    "Calories": ["calories", "kcal"],
    "Protein": ["protein", "g"],
    "Fat": ["fat", "g"],
    "Carbs": ["carbohydrates", "g"],
    "Fiber": ["fiber", "g"]
}

#########################################################################################################################
#-- Main Page Content
st.markdown("<h1 style='text-align: center;'>Food Recipe Recommendation</h1><br>", unsafe_allow_html=True)

# Input section
input_cols = st.columns(2)
with input_cols[0]:
    st.markdown("<p style='text-align: left; font-weight: bold;'>Ingredients Input</p>", unsafe_allow_html=True)
    
    ingredient_input = st.multiselect(
        "Ingredients Input",
        [],
        accept_new_options=True,
        label_visibility="collapsed",
    )
    st.session_state["ingredient_input"] = ingredient_input

    # Clean the input ingredients for highlighting
    clean_input = [input.lower() for input in ingredient_input]

with input_cols[1]:
    st.markdown("<p style='text-align: left; font-weight: bold;'>Nutrition Filters</p>", unsafe_allow_html=True)
    nutrition_filters = st.multiselect(
        "Nutrition Filters",
        nutrition_filter_map.keys(),
        accept_new_options=False,
        label_visibility="collapsed",
    )
    
    for filter in nutrition_filter_map.keys():
        if filter in nutrition_filters:
            st.session_state[filter] = True
        else:
            st.session_state[filter] = False

# Surprise Me! button
if st.button("Surprise Me!", use_container_width=True):
    st.session_state["ingredient_input"] = []
    update_recommendation(n_recipes)

# Grid display section
## Update recommendation before displaying
update_recommendation(n_recipes)

## Create grid for displaying the recommended recipes
grid = st.columns(n_cols) * n_rows

for i, card in enumerate(grid):
    try:
        # Get the index of the recipe to display
        current_recipe = data.iloc[st.session_state["displayed_recipe_indices"][i]]

        # Create a container for the recipe card
        container = card.container(border=True)
        with container:
            # Display the recipe image
            st.markdown(f"<a href='{current_recipe.recipe_url}'><img src='{data.img_url.iloc[st.session_state['displayed_recipe_indices'][i]]}'></a>", unsafe_allow_html=True)
            
            # Recipe title
            st.markdown(f"<h3 style='font-weight: bold;'><a href='{current_recipe.recipe_url}'>{current_recipe.recipe_title}</a></h3>", unsafe_allow_html=True)
            
            # Nutrition facts
            nutrition_cols = st.columns(5)
            nutrition_size = "70%"
            value_size = "90%"

            for nutrition_col, (nutrition_display, (nutrition_name, units)) in zip(nutrition_cols, nutrition_facts_map.items()):
                if current_recipe[nutrition_name] == 0:
                    nutrition_col.markdown(f"<p style='font-size: {nutrition_size}; margin: 0 auto 0 auto;'>{nutrition_display}</p>\
                            <p style='font-size: {value_size}; font-weight: bold;'>N/A</p>", unsafe_allow_html=True)
                else:
                    nutrition_col.markdown(f"<p style='font-size: {nutrition_size}; margin: 0 auto 0 auto;'>{nutrition_display}</p>\
                            <p style='font-size: {value_size}; font-weight: bold;'>{current_recipe[nutrition_name]:.0f} {units}</p>", unsafe_allow_html=True)
                
            st.markdown("<hr style='margin: 0 auto 5% auto;'>", unsafe_allow_html=True)
        
            # Ingredients
            ing_cols = st.columns(2) * 50
            for i_col, ingredient in enumerate(current_recipe.ingredients):
                clean_str = ingredient.replace("_", " ").title()

                # Highlight the ingredient if it's in the input ingredients
                if np.isin(re.split(r"[_|\/]", ingredient), clean_input).any():
                    ing_cols[i_col].markdown(f"<li style='font-size: 80%; margin-bottom: 0; font-weight: bold; background-color:#ff4b4b; border-radius: 3.5px;'>{clean_str}</li>", unsafe_allow_html=True)
                else:
                    ing_cols[i_col].markdown(f"<li style='font-size: 80%; margin-bottom: 0'>{clean_str}</li>", unsafe_allow_html=True)
            
            st.markdown(f"<p style='font-size: 100%; margin: 5% auto 5% auto; text-align: right;'>\
                            <span style='font-size: 150%; font-weight: bold;'>{len(current_recipe.ingredients)}</span> ingredients\
                          </p>", unsafe_allow_html=True)
            
            st.markdown("<hr style='margin: 0 auto 5% auto;'>", unsafe_allow_html=True)

            # Similarity score
            cols = st.columns([0.3, 0.7])
            
            with cols[0]:
                if st.session_state['displayed_similarity_scores'][i] == 0:
                    st.markdown(f"<p style='font-size: 100%; margin: 0 auto 0 auto;'>Similarity</p>\
                            <p style='font-size: 150%; font-weight: bold;'>N/A</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='font-size: 100%; margin: 0 auto 0 auto;'>Similarity</p>\
                                <p style='font-size: 150%; font-weight: bold;'>{st.session_state['displayed_similarity_scores'][i]*100:.0f}%</p>", unsafe_allow_html=True)
            
            # Total time
            with cols[1]:
                hours = int(current_recipe.total_time//60)
                minutes = int(current_recipe.total_time%60)
                time_str = ""
                if hours:
                    time_str += f"{hours} hrs"
                if minutes:
                    time_str += f" {minutes} mins"
            
                st.markdown(f"<p style='font-size: 100%; margin: 0 auto 0 auto; text-align: right;'>Cook Time</p>\
                            <p style='font-size: 150%; font-weight: bold; text-align: right;'>{time_str}</p>", unsafe_allow_html=True)
                
    except Exception as e:
        # st.toast(e)
        # If there are not enough recipes to fill the columns, skip the remaining columns
        continue

#########################################################################################################################
#-- Development Controls
if dev:
    with st.sidebar:
        st.header("Development Controls")
        # Displayed Configurations
        st.header("Displayed Configurations")

        st.write(f"**Number of Recipes to Recommend:** {n_recipes}")
        st.write(f"**Number of Columns:** {n_cols}")
        st.write(f"**Number of Rows:** {n_rows}")

        st.write(f"**Session Recipe Display Indices:**")
        st.write(f"{st.session_state["displayed_recipe_indices"]}")

        st.write(f"**Session Recipe Display Similarities:**")
        st.write(f"{st.session_state["displayed_similarity_scores"]}")