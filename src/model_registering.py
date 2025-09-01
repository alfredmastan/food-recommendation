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
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    # Get the highest performance and oldest run
    runs = mlflow.search_runs(experiment_names=["FastText-Model"])
    sorted_runs = runs.sort_values(by=["start_time", "metrics.RMSE", "metrics.MSE", "metrics.MAE"], ascending=True)
    sorted_runs = sorted_runs.sort_values(by=["metrics.Hit-Rate", "metrics.MAP", "metrics.MRR", "metrics.NDCG",  "metrics.F1_10"], ascending=False)
    best_run = sorted_runs.iloc[0]

    # Check if any registered model exists
    if mlflow.search_registered_models():
        # Get current registered model metadata
        try:
            with open("registered_metadata.json", "r") as f:
                registered_metadata = json.load(f)
        except:
            raise ValueError("Could not find registered metadata. Delete registered model in MLFlow and try again.")

        if best_run["run_id"] == registered_metadata["run_id"]:
            print("Current registered model is the best.")
            return

    # Load validation model metadata
    with open("validation_metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Save registered model metadata
    with open("registered_metadata.json", "w") as f:
        f.write(json.dumps(metadata, indent=4))
    
    # Register best model to MLFlow
    with mlflow.start_run(run_id=best_run.run_id):
        mlflow.register_model(
            model_uri=metadata["server_uri"],
            name="fasttext_model"
        )

    print("New best model registered.")

    print("Checking for old runs to delete...")
    
    deprecate_runs = sorted_runs.iloc[params["model_register"]["max_runs"]:]
    if deprecate_runs.empty:
        print("No old runs to delete.")
        return
    
    for index, run in deprecate_runs.iterrows():
        # Delete run from MLFlow
        id = run["run_id"]
        mlflow.delete_run(id)

        # Delete run from local file system
        run_path = os.path.join("mlflow/mlruns", run["experiment_id"], run["run_id"])
        if os.path.exists(run_path):
            print(f"Deleting local run directory: {run_path}")
            shutil.rmtree(run_path)

        # Delete model artifact from local file system
        model_artifact_path = os.path.join(*run["params.local_uri"].split("/")[:-1])
        if os.path.exists(model_artifact_path):
            print(f"Deleting local model artifacts: {model_artifact_path}")
            shutil.rmtree(model_artifact_path)

        # Delete model artifact from local file system
        artifact_path = os.path.join("mlflow/mlartifacts", run["experiment_id"], run["run_id"])
        if os.path.exists(artifact_path):
            print(f"Deleting local artifacts: {artifact_path}")
            shutil.rmtree(artifact_path)

if __name__ == "__main__":
    print(f"{'='*20} Starting model registering pipeline. {'='*20}")
    main()
    print(f"{'='*20} Model registering pipeline complete. {'='*20}")
