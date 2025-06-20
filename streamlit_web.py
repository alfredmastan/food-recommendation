"""
Streamlit web app for Food Recipes Recommendation using Word2Vec model.
"""

## Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

st.set_page_config(layout="wide")
## Test
# st.title("Food Recipes Recommendation")
# st.write("This is a simple web app for recommending food recipes using Word2Vec model.")

data = pd.read_pickle("processed_cookbook.pkl")
st.title("Food Recipes Recommendation")
st.write("This is a simple web app for recommending food recipes using Word2Vec model.")
st.button("Recommend Recipes")

model = Word2Vec.load("word2vec.model")

from sklearn.metrics.pairwise import cosine_similarity

def get_recipe_vector_mean(ingredients: list, model: Word2Vec) -> list:
    """
    Takes in a list of recipe ingredients, embeds it and calculate the mean
    """
    vector_embedding = [model.wv[ing] for ing in ingredients if ing in model.wv]
    return np.mean(vector_embedding, axis=0) if vector_embedding else np.zeros(model.vector_size) * -1
    
recipes_vector_mean = [get_recipe_vector_mean(recipe, model) for recipe in data.ingredients]
recipes_vector_mean = np.array(recipes_vector_mean)

sims = cosine_similarity(recipes_vector_mean[0].reshape(1, -1), recipes_vector_mean)[0]
# sims.argsort()[::-1][:10]

n = 10  # Number of recipes to recommend

# Randomly select n indices from the recipes_vector_mean
rand_idx = np.random.choice(len(recipes_vector_mean), n, replace=False)

# data.iloc[rand_idx]

st.subheader("Recommended Recipes")

num_cols = np.ceil(n/3).astype(int)  # Number of columns to display

n_col = 3  # Number of recipes to display in each column

# Create columns for displaying the recommended recipes
for i_col in range(num_cols):
    cols = st.columns(n_col, border=True)

    # Display the recommended recipes in the columns
    for i in range(len(cols)):
        try:
            idx = rand_idx[i_col * n_col + i]
            cols[i].subheader(f"{data.recipe_title.iloc[idx]}")
            
            # Url
            cols[i].markdown(f"*{data.recipe_url.iloc[idx]}*")

            # Ingredients
            cols[i].markdown("**Ingredients:**")
            str = []
            for ingredient in data.ingredients.iloc[idx]:
                clean_str = ingredient.replace("_", " ").title()
                str.append(f":blue-badge[{clean_str}]")
            cols[i].markdown(" ".join(str))

            # Total Time
            hours = int(data.total_time.iloc[idx]//60)
            minutes = int(data.total_time.iloc[idx]%60)
            time_str = "**Total Time:**"
            if hours:
                time_str += f" {hours} hours"
            if minutes:
                time_str += f" {minutes} minutes"

            cols[i].markdown(time_str)
        except:
            # If there are not enough recipes to fill the columns, skip the remaining columns
            continue


# st.rerun()
