import numpy as np
import pandas as pd
import re
import os
import yaml
import ast

# Importing necessary libraries for text processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import string

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Extra words to remove
PREP_WORDS = ["thinly", "finely", "softened", "crushed", "skinned", "whole", "chopped", "minced"]

TEMP_WORDS = ["hot", "warm", "cold", "room", "chilled", "frozen", "cool", "boiling", "temperature"]

TASTE_WORDS = ["sweet", "sour", "bitter", "salty", "umami"]

QUALITY_WORDS = ["fresh", "freshly", "organic", "good", "quality", "extra", "premium", "best", "finest", "high", "filtered", "light", "sashimi-grade", "pure", "fine"]

SIZE_WORDS = ["large", "mini", "small", "medium", "jumbo", "extra", "extra-large", "extra-small", "sliced", "halved", "diced", "cubed", "peeled", "grated", "shredded", "quartered", "clove", "short-grain", "heaping"]

STATE_WORDS = ["boneless", "skinless", "skin", "cooked", "ripe", "coarse", "coarsely", "filet", "fillet", "fillets", "canned",
                "homemade", "used", "dried", "reserved", "packet", "ground", "uncooked", "pasteurized", "bonein", "skinon", "preserved", "raw", "cooking"]

OTHER_WORDS = ["quality", "extra", "virgin", "extravirgin", "long", "high", "japanese", "combination", "diamond", "kosher", "stalk", "juice", "sea", "taste", "crystal", "choice"]

UNITS = ["cup", "cups", "tbsp", "tablespoon", "tsp", "teaspoon", "g", "kg", "oz", "ml", "l", "lb", "pound", "inch", "cm", "m", "handful", "bulb", "dollop", "pinch"]

EXTRA_WORDS = PREP_WORDS + TEMP_WORDS + TASTE_WORDS + QUALITY_WORDS + SIZE_WORDS + STATE_WORDS + OTHER_WORDS + UNITS

# Helper Functions
def load_params():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    return params

def clean_recipe_title(title: string) -> string:
    """
    Cleans the recipe title by removing characters that are not digits or alphabets and extra spaces.
    Also removes certain words from the title and ensure capitalization.
    """
    line = re.sub(r"\s*\([^()]*\)\s*", "", title) # Remove text within parentheses
    line = re.sub(r"\s*\{[^{}]*\}\s*", "", line) # Remove text within brackets
    line = re.sub(r"[^ \-a-zA-Z0-9()]*", "", line) # Replace unwanted characters with space
    line = re.sub(r"\s*\([^()]*", "", line) # Remove uncomplete parentheses

    # Remove certain words from recipe titles
    line = re.sub(r"(recipe|how\s*to\s*\w*\s|best|easy)", "", line, flags=re.IGNORECASE).strip()
    line = line.strip()  # Remove leading and trailing spaces
    return line.title()  # Capitalize the first letter of each word

def clean_ingredients(ingredients: list) -> list:
    """
    A single ingredient may consist of several words, thus
    it takes each string of ingredient and process it using multiple NLP methods into a clean string.
    This function also splits ingredients that are combined by commas and "and" and processes them separately.
    """
    # Initialize functions for cleaning
    translator = str.maketrans('', '', '!"#$%&\'()*+,.:;<=>?@[\\]^_`{|}~') # Punctuations remover (exclue slash "/" and strip "-")
    lemmatizer = WordNetLemmatizer() # Word lemmatizer
    
    clean_ingredients = []
    for ingredient in ingredients:
        # Split by commas and "and" to handle combined ingredients
        split_ingredients = [ing.strip() for ing in re.split(r'\s+(and|&|,)\s+', ingredient, flags=re.IGNORECASE) if ing.strip()]
        split_ingredients = list(dict.fromkeys(split_ingredients))  # Remove duplicates from split ingredients

        # Process each split ingredient separately
        for sub_ingredient in split_ingredients:
            # Normalization
            line = sub_ingredient.lower() # Make sure string are lowercase
            line = re.sub(r"\s*\([^()]*\)\s*", "", line) # Remove text between parentheses
            
            # Surface level cleaning
            line = re.sub(r"(\s+(or|/){1}\s+)", "/", line) # Remove ingredient substitutes
            line = re.sub(r"(diamond crystal|premium-quality)", "", line) # Remove brands
            line = re.sub(r"(\w*\d\w*|½|¼|¾)", "", line) # Remove numbers

            # Remove punctuations
            line = line.translate(translator)

            # In-depth cleaning
            line_tokenized = word_tokenize(line) # Tokenize the ingredient first
            line_lemmatized = [lemmatizer.lemmatize(ing) for ing in line_tokenized] # Then lemmatize the ingredient
            line_split = [ing for ing in line_lemmatized if ing not in stopwords.words("english") + EXTRA_WORDS + [""]] # Remove stop words
            
            if line_split:  # Only add if there are remaining words after cleaning
                line_split = list(dict.fromkeys(line_split)) # Remove duplicates from the words in a single ingredient
                line = "_".join(line_split) # Rejoin the ingredient as a single string
                clean_ingredients.append(line) # Add back to the ingredients list
        
    clean_ingredients = list(dict.fromkeys(clean_ingredients)) # Remove duplicates
    
    return clean_ingredients

def main():
    # Load params
    params = load_params()

    # Combine raw data
    print("Combining recipe data...")
    pickle_files = []
    for root, _, files in os.walk(params["data_preprocessing"]["raw_data_path"]):
        for f in files:
            if f.endswith(".pkl"):
                pickle_files.append(os.path.join(root, f))

    dfs = []
    for path in pickle_files:
        df = pd.read_pickle(path)
        dfs.append(df)

    complete_cookbook = pd.concat(dfs, axis=0)
    complete_cookbook = complete_cookbook.reset_index(drop=True)
    complete_cookbook.ingredients = complete_cookbook.ingredients.apply(lambda x: ast.literal_eval(x))
    print(f"Combined {len(pickle_files)} files with total {len(complete_cookbook)} recipes.")

    # Create a cleaned copy of the cookbook
    cleaned_cookbook = complete_cookbook.copy()

    # General Filtering
    print("Filtering recipes...")
    ## Clean recipes title from unwanted characters
    cleaned_cookbook["recipe_title"] = cleaned_cookbook.recipe_title.apply(clean_recipe_title)

    ## Drop compilation recipes and keep single recipes only
    all_numbers_title_idx = ~pd.isna(cleaned_cookbook.recipe_title.apply(lambda x: re.match(r"^\d+", x.lower()))) # All titles with numbers in front 
    keep_numbers_title_idx = ~pd.isna(cleaned_cookbook.recipe_title.apply(lambda x: re.match(r"^\d+( |-)+(ingredient(s)?|minute(s)?|hour(s)?|bowl(s)?|pot(s)?|layer(s)?)", x.lower()))) # Titles with numbers in front to keep

    cleaned_cookbook = cleaned_cookbook.drop(cleaned_cookbook[all_numbers_title_idx & ~keep_numbers_title_idx].index)
    cleaned_cookbook = cleaned_cookbook.reset_index(drop=True) # Reset index to avoid issues with indexing

    ## Remove non-meal recipe (sauces, dressing, store direction, etc)
    other_recipes_idx = ~pd.isna(cleaned_cookbook.recipe_title.apply(lambda x: re.search(r"(sauce(s)?|dressing(s)?|store|cut|slice|clean|what)", x.lower())))
    cleaned_cookbook = cleaned_cookbook.drop(cleaned_cookbook[other_recipes_idx].index)
    cleaned_cookbook = cleaned_cookbook.reset_index(drop=True) # Reset index to avoid issues with indexing

    ## Remove 1-2 ingredients recipes
    less_than_two_ingredients = cleaned_cookbook[cleaned_cookbook.ingredients.apply(lambda x: len(x)) <= 2]
    cleaned_cookbook = cleaned_cookbook.drop(less_than_two_ingredients.index)
    cleaned_cookbook = cleaned_cookbook.reset_index(drop=True) # Reset index to avoid issues with indexing

    # Handle NA Values
    ## Fill NA total time based on other times in case it wasn't catched when scraping. Otherwise, drop and fill the rest time variables with 0s
    cleaned_cookbook.total_time = cleaned_cookbook.total_time.fillna(cleaned_cookbook.prep_time + cleaned_cookbook.cook_time + cleaned_cookbook.custom_time)
    cleaned_cookbook = cleaned_cookbook.dropna(subset="total_time")
    cleaned_cookbook[["prep_time", "cook_time", "custom_time"]] = cleaned_cookbook[["prep_time", "cook_time", "custom_time"]].fillna(0)
    cleaned_cookbook.total_time = cleaned_cookbook.prep_time + cleaned_cookbook.cook_time + cleaned_cookbook.custom_time

    ## Fill NA macros and micros with 0s
    fill_idx = cleaned_cookbook.columns[8:-1]
    cleaned_cookbook[fill_idx] = cleaned_cookbook[fill_idx].fillna(0)

    # Handle Duplicated Values
    ## Remove exact duplicate recipes
    cleaned_cookbook = cleaned_cookbook.drop_duplicates(subset=["recipe_title"])

    # In-depth ingredients filtering
    print("Cleaning ingredient names...")
    cleaned_cookbook.ingredients = cleaned_cookbook.ingredients.apply(clean_ingredients)

    # Convert recipe_title to string to avoid type errors
    cleaned_cookbook.recipe_title = cleaned_cookbook.recipe_title.astype(str)

    # Create ingredients count variable
    cleaned_cookbook["num_ingredients"] = cleaned_cookbook.ingredients.apply(lambda x: len(x))

    # Export cleaned cookbook
    print("Saving cleaned data...")
    cleaned_cookbook = cleaned_cookbook.reset_index(drop=True)
    cleaned_cookbook.to_pickle("data/processed_cookbook.pkl", protocol=4)
    print(f"Final processed data contains total {len(cleaned_cookbook)} recipes.")

if __name__ == "__main__":
    print(f"{'='*20} Starting data preprocessing pipeline. {'='*20}")
    main()
    print(f"{'='*20} Data preprocessing pipeline complete. {'='*20}")
