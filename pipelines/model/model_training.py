# Import required libraries
import numpy as np
import pandas as pd

from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity
import mlflow

import json
import yaml

def load_params():
    with open("../params.yaml") as f:
        params = yaml.safe_load(f)
    return params