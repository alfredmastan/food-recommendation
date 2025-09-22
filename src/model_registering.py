# Import required libraries
import numpy as np
import pandas as pd

# Model
import mlflow

# Logging
import os
import shutil
import yaml
import json

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Helper Functions
def load_params():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    return params

def main():
    # Load current model metadata
    with open("validation_metadata.json", "r") as f:
        metadata = json.load(f)

    # Load params
    params = load_params()

    # Get the highest performance and oldest run
    print("Getting the best run...")
    mlflow.set_tracking_uri("http://localhost:8080/")
    
    runs = mlflow.search_runs(experiment_names=["FastText-Model"])
    sorted_runs = runs.sort_values(by=["start_time", "metrics.RMSE", "metrics.MSE", "metrics.MAE"], ascending=True)
    sorted_runs = sorted_runs.sort_values(by=["metrics.Hit-Rate", "metrics.MAP", "metrics.MRR", "metrics.NDCG",  "metrics.F1_10"], ascending=False)
    best_run = sorted_runs.iloc[0]

    # if no registered_metadata -> register best model
    # else -> check if best model is current registered model
    #       -> if not, register best model

    # Try to get current registered model metadata
    try:
        with open("registered_metadata.json", "r") as f:
            registered_metadata = json.load(f)
    except:
        print("No registered model found. Registering best model...")
        registered_metadata = {}
    
    # Check if the best run is the current registered model
    if best_run["run_id"] == registered_metadata.get("run_id", ""):
        print("Current registered model is the best.")
    else:
        # Load validation model metadata
        with open("validation_metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Write registered model metadata
        with open("registered_metadata.json", "w") as f:
            f.write(json.dumps(metadata, indent=4))
            print("Registered model metadata updated.")
        
        # Register best model to MLFlow
        with mlflow.start_run(run_id=best_run.run_id):
            mlflow.register_model(
                model_uri=metadata["server_uri"],
                name="fasttext_model"
            )

        # Copy registered model artifacts to a separate directory
        src_path = metadata["local_uri"]
        dest_path = "service/model/"
        os.makedirs(dest_path, exist_ok=True)

        if os.path.exists(src_path):
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            shutil.copytree(src_path, dest_path)
            print(f"Copied model artifacts to {dest_path}")
        else:
            print(f"Source path {src_path} does not exist. Cannot copy model artifacts.")

        print("New best model registered.")


if __name__ == "__main__":
    print(f"{'='*20} Starting model registering pipeline. {'='*20}")
    main()
    print(f"{'='*20} Model registering pipeline complete. {'='*20}")
