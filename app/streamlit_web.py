"""
Streamlit web app for Food Recipes Recommendation using Word2Vec model.
"""

#########################################################################################################################
#-- Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import boto3
from datetime import datetime
import requests
import json
import time
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

@st.cache_resource
def connect_database():
    """Connect to the DynamoDB database."""
    dynamodb = boto3.resource("dynamodb",
                            aws_access_key_id=st.secrets.s3.AWS_ACCESS_KEY_ID,
                            aws_secret_access_key=st.secrets.s3.AWS_SECRET_ACCESS_KEY,
                            region_name=st.secrets.s3.AWS_DEFAULT_REGION)
    table = dynamodb.Table(st.secrets.s3.DB_NAME)
    return table

## Recommendation functions
def update_recommendation(n_recipes):
    """Update the recommendation by resetting the display indices."""

    # if not st.user.is_logged_in:
    #     st.toast("User not logged in.")
    #     st.session_state["display_recipe_indices"] = np.random.choice(len(data), n_recipes, replace=False)
    #     st.session_state["display_excluded_indices"] = []
    #     return
    
    # Fetch liked_idx and disliked_idx for current user
    # user_item = table.get_item(Key={"user_id": int(st.user.get("sub"))})["Item"]
    # liked_idx = list(map(int, user_item["liked_idx"].keys()))
    # disliked_idx = list(map(int, user_item["disliked_idx"].keys()))

    # Call the recommendation API
    # response = requests.get(f"http://localhost:8000/recommend/", json={"liked_idx": liked_idx, "disliked_idx": disliked_idx})
    if not st.session_state.get("ingredient_input", ""):
        st.session_state["display_recipe_indices"] = np.random.choice(len(data), n_recipes, replace=False)
        st.session_state["display_excluded_indices"] = []
        return

    
    response = requests.post(f"http://localhost:8000/recommend/", params={"query": st.session_state["ingredient_input"]})
    similarity_scores = np.asarray(json.loads(response.json()))
    recommended_indices = np.argsort(similarity_scores)[::-1][:n_recipes]
    st.toast(recommended_indices)

    if response.status_code != 200:
        st.toast("Failed to fetch recommendations. API call failed.")

    if json.loads(response.json()):
        st.session_state["display_recipe_indices"] = recommended_indices
        st.session_state["display_excluded_indices"] = set(st.session_state["liked_idx"].keys()).union(st.session_state["disliked_idx"].keys())
        st.toast("RECOMMENDATION UPDATED")
    else:
        st.toast("Failed to fetch recommendations.")

def liked(recipe_idx, display_idx):
    """Handle the like button click event."""
    st.toast("Liked!")  # Display a toast message when the like button is clicked
    st.session_state["liked_idx"][str(recipe_idx)] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Update the user vector by liking the recipe
    st.session_state["display_recipe_indices"] = np.delete(st.session_state["display_recipe_indices"], display_idx)  # Remove the liked recipe from the displayed recommendations
    save_user_config()

def disliked(recipe_idx, display_idx):
    """Handle the dislike button click event."""
    st.toast("Disliked!")  # Display a toast message when the dislike button is clicked
    st.session_state["disliked_idx"][str(recipe_idx)] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Update the user vector by disliking the recipe
    st.session_state["display_recipe_indices"] = np.delete(st.session_state["display_recipe_indices"], display_idx)  # Remove the disliked recipe from the displayed recommendations
    save_user_config()

#########################################################################################################################
#-- Database Management
def load_user_config():
    """Load user configurations from DynamoDB."""  
    try:
        user_config = table.get_item(Key={"user_id": int(st.user.get("sub"))})["Item"]
        st.session_state["liked_idx"] = user_config.get("liked_idx")
        st.session_state["disliked_idx"] = user_config.get("disliked_idx")
        st.toast("USER DYNAMODB LOADED")
    except:
        st.session_state["liked_idx"] = {}
        st.session_state["disliked_idx"] = {}
        st.toast("USER DYNAMODB NOT FOUND OR NO LIKED/DISLIKED RECIPES")

def save_user_config():
    """Save user configurations to DynamoDB only if the user is logged in."""
    if st.user.is_logged_in:
        table.put_item(
            Item={
                "user_id": int(st.user.get("sub")),
                "liked_idx": st.session_state["liked_idx"],
                "disliked_idx": st.session_state["disliked_idx"]
            }
        )
        st.toast("USER CONFIGURATIONS SAVED TO DYNAMODB")

def reset_user_config():
    """Reset the user configurations in the session state."""
    table.put_item(
        Item={
            "user_id": int(st.user.get("sub")),
            "liked_idx": {},
            "disliked_idx": {}
        }
    )

def load_params():
    with open("../params.yaml") as f:
        params = yaml.safe_load(f)
    return params

#########################################################################################################################
#-- Preparing the content
params = load_params()
table = connect_database()  # Connect to the DynamoDB table
data = load_data(params["model_pipeline"]["recipe_path"]) # Load the main data
load_user_config() # Load user configurations from DynamoDB if logged in

#########################################################################################################################
#-- Content Configurations
n_recipes = params["model_service"]["n_recs"]  # Number of recipes to recommend
n_cols = 4 # Number of recipes to display in each row
n_rows = np.ceil(n_recipes/n_cols).astype(int) # Number of rows to display

if "display_recipe_indices" not in st.session_state or "display_excluded_indices" not in st.session_state:
    update_recommendation(n_recipes)

# display_recipe_indices = st.session_state["display_recipe_indices"]  # Get the recommended indices from the session state
# excluded_indices = st.session_state["display_excluded_indices"]  # Get the excluded indices from the session state

preloaded_data = data[~data.id.isin(list(st.session_state["display_excluded_indices"]))] # Preloaded data for the selected recipes

#########################################################################################################################
#-- Main Page Content
st.title("Food Recipes Recommendation")
st.write("This is a simple web app for recommending food recipes using Word2Vec model.")
st.feedback(options="thumbs")

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
        current_recipe = preloaded_data.iloc[st.session_state["display_recipe_indices"][i]]

        # Create a container for the recipe card
        container = card.container(border=True)
        with container:
            # Display the recipe image
            st.image(f"{preloaded_data.image_url.iloc[st.session_state["display_recipe_indices"][i]]}", use_container_width=True)
            # sims = cosine_similarity(user.vec.reshape(1, -1), st.session_state["recipe_vector_means"])[0]

            # Show the similarity score
            # st.badge(f"Similarity: {sims[display_recipe_indices[i]]*100:.0f}%", color="red")

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

            # Like/Dislike
            button_cols = st.columns(2, gap="small")  # Create two columns for like/dislike buttons
            button_cols[0].button("", key=f"like_{i}",icon=":material/thumb_up:", use_container_width=True, on_click=liked, args=(st.session_state["display_recipe_indices"][i], i))
            button_cols[1].button("", key=f"dislike_{i}", icon=":material/thumb_down:", use_container_width=True, on_click=disliked, args=(st.session_state["display_recipe_indices"][i], i))

    except:
        # If there are not enough recipes to fill the columns, skip the remaining columns
        continue

#########################################################################################################################
#-- Development Controls
with st.sidebar:
    
    # User Information
    st.header("User Information")
    st.write(f"**User ID:** {st.user.get('sub', 'Not logged in')}")
    
    if not st.user.is_logged_in:
        if st.button("Login", use_container_width=True, icon=":material/login:"):
            st.login()
        
    else:
        if st.button("Logout", use_container_width=True, icon=":material/logout:"):
            st.logout()

        if st.button(":red[Reset User]", use_container_width=True):
            reset_user_config()
            st.toast("USER CONFIGURATIONS RESET")
            st.rerun()

    st.divider()
    # Displayed Configurations
    st.header("Displayed Configurations")

    st.write(f"**Session Recipe Display Indices:**")
    st.write(f"{st.session_state["display_recipe_indices"]}")

    st.write("**Current Excluded Indices:**")
    if not st.session_state["display_excluded_indices"]:
        st.write("No recipes excluded yet.")
    else:
        st.write(f"{st.session_state["display_excluded_indices"]}")
    

    st.write(f"**Number of Recipes Displayed:** {len(data)-len(st.session_state["display_excluded_indices"])}")
    st.divider()

    # User Configurations
    st.header("Current User Configurations")

    # Liked/Disliked recipes
    st.write("**Liked Recipes:**")
    if st.session_state.liked_idx:
        st.write(f"{list(st.session_state["liked_idx"].keys())}")
    else:
        st.write("No recipes excluded yet.")

    st.write("**Disliked Recipes:**")
    if st.session_state.disliked_idx:
        st.write(f"{list(st.session_state["disliked_idx"].keys())}")
    else:
        st.write("No recipes disliked yet.")


    # # User Cookies
    # st.write("Current Cookies:")
    # st.write(cookies)

    # # Clear the cookie
    # if st.button("Clear Cookies"):
    #     reset_cookies() 

    # User recipes vector mean
    # st.write("**User Recipes Vector Mean Shape:**")
    # st.write(f"{st.session_state["recipe_vector_means"].shape}")

    # # User vector
    # st.write("**User Vector:**")
    # st.write(f"{" ".join(np.round(user.vec, 2).astype("str"))}")
##