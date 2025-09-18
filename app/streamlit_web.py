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
from streamlit_scroll_to_top import scroll_to_here

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

## Load params
@st.cache_data
def load_params():
    with open("../params.yaml") as f:
        params = yaml.safe_load(f)
    return params

## Load data
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load the main data and pre-trained Word2Vec model."""
    return pd.read_pickle(path)

## Recommendation functions
def fetch_recommendations():
    """Fetch recommendations from the API based on the input ingredients."""
    # Call the recommendation API
    response = requests.post(f"http://localhost:8000/recommend/", params={"query": st.session_state.get("input_ingredients", [])})

    if response.status_code != 200:
        st.toast("Failed to fetch recommendations. API call failed.")
        return
    
    # Process the similarity scores from the response 
    similarity_scores = np.asarray(json.loads(response.json()))
    st.session_state["raw_similarity"] = similarity_scores if similarity_scores.size != 0 else np.array([0]*len(data))

    # Sort the similarity scores, if none, show random recipes from filtered recipes
    if len(similarity_scores) != 0:
        st.session_state["raw_recommendation"] = np.argsort(similarity_scores)[::-1]
    else:
        st.session_state["raw_recommendation"] = np.random.choice(len(data), len(data), replace=False)

    st.session_state["page"] = 0 # Reset to first page
    display_recommendations() # Update the displayed recommendations

def update_filters():
    """Update the filter mask from raw data indices."""
    # Update the individual filter states
    for filter in nutrition_filter_map.keys():
        if filter in st.session_state.get("nutrition_filters", []):
            st.session_state[filter] = True
        else:
            st.session_state[filter] = False

    # Apply filters if any (Hard filter)
    mask = set(range(len(data))) # Indexes of all recipes
    for filter, conditions in nutrition_filter_map.items():
        if not st.session_state.get(filter, False):
            continue

        for condition in conditions:
            col, threshold, direction = condition
            st.toast(f"Applying filter: {filter} - {col} {direction} than {threshold}")
            if direction == "lower":
                mask = mask.intersection(data[data[col] <= threshold].index.to_numpy())
            else:
                mask = mask.intersection(data[data[col] >= threshold].index.to_numpy())

    st.session_state["filter_mask"] = list(mask) # Convert to list for indexing
    st.session_state["page"] = 0 # Reset to first page
    display_recommendations() # Update the displayed recommendations based on the new filter

def display_recommendations():
    """Display the recommended recipes based on the current filters and pagination."""
    raw_recommendation = st.session_state["raw_recommendation"]
    mask = st.session_state.get("filter_mask", raw_recommendation)

    # Apply mask to the raw recommended indices
    recommended_indices = raw_recommendation[np.isin(raw_recommendation, mask)]
 
    # Select the top N recipes and its similarity scores to display
    range_recipes_display = np.array([0, n_recipes]) + (st.session_state.get("page", 0) * n_recipes) # Range of recipes to display
    st.session_state["displayed_recipe_indices"] = recommended_indices[range_recipes_display[0]:range_recipes_display[1]]
    st.session_state["displayed_similarity_scores"] = st.session_state["raw_similarity"][recommended_indices][range_recipes_display[0]:range_recipes_display[1]] # Update the displayed similarity scores
    st.session_state["final_recommendation"] = recommended_indices # Store the final recommendation list after filtering
    

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
# Back to top management
if 'scroll_to_top' not in st.session_state:
    st.session_state.scroll_to_top = False

if st.session_state.scroll_to_top:
    scroll_to_here(key='top')  # Scroll to the top of the page
    st.session_state.scroll_to_top = False  # Reset the state after scrolling

st.markdown("<h1 style='text-align: center;'>Food Recipe Recommendation</h1><br>", unsafe_allow_html=True)

# Input section
input_cols = st.columns(2)

with input_cols[0]:
    st.markdown("<p style='text-align: left; font-weight: bold;'>Ingredients</p>", unsafe_allow_html=True)
    
    input_ingredients = st.multiselect(
        "Ingredients",
        [],
        accept_new_options=True,
        label_visibility="collapsed",
        default=[],
        key="input_ingredients",
        on_change=fetch_recommendations,
    )

    # Clean the input ingredients for highlighting
    clean_input = [input.lower() for input in input_ingredients]

with input_cols[1]:
    st.markdown("<p style='text-align: left; font-weight: bold;'>Nutrition Filters</p>", unsafe_allow_html=True)
    nutrition_filters = st.multiselect(
        "Nutrition Filters",
        nutrition_filter_map.keys(),
        accept_new_options=False,
        label_visibility="collapsed",
        key="nutrition_filters",
        on_change=update_filters,
    )
    
# Surprise Me! button
if st.button("Surprise Me!", use_container_width=True, on_click=lambda: st.session_state.update({"input_ingredients": []})):
    fetch_recommendations()

# Grid display section
# Initialize session states
if "raw_recommendation" not in st.session_state:
    fetch_recommendations()

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

st.markdown("<hr style='margin: 2% auto 2% auto;'>", unsafe_allow_html=True)

# Pagination controls
max_page = np.ceil(len(st.session_state["final_recommendation"]) / n_recipes).astype(int)
page_cols = st.columns([10] + [1]*10 + [10], vertical_alignment="bottom") # Max 10 page buttons

def scroll_to_top():
    st.session_state.scroll_to_top = True

for i in range(len(page_cols[1:-1])):
    if st.session_state.get("page", 0) == i:
        page_cols[i+1].markdown(f"<span style='font-size: 150%; font-weight: bold; text-decoration: underline;'>{i+1}</span>", unsafe_allow_html=True)
    else:
        if page_cols[i+1].button(f"{i+1}", type="tertiary", on_click=scroll_to_top):
            st.session_state["page"] = i
            display_recommendations()
            st.rerun()

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