# Import required libraries
import random
import numpy as np
import pandas as pd

# Model
import mlflow
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Logging
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
    with open("training_metadata.json", "r") as f:
        metadata = json.load(f)

    # Load data and params
    params = load_params()
    data = pd.read_pickle(params["model_pipeline"]["recipe_path"])

    # Load mlflow model
    print("Loading model...")
    model = mlflow.pyfunc.load_model(metadata["local_uri"])

    # Train ingredients TF-IDF as ground truth
    tfidf_ings  = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), norm="l2")
    X_ings  = tfidf_ings.fit_transform(data.ingredients.apply(lambda x: ", ".join(x)))   

    # Set seed for reproducibility
    random.seed(params["model_evaluation"]["seed"])

    # Grab evaluation parameters
    k = params["model_evaluation"]["k"]
    relevant_thresh = params["model_evaluation"]["relevant_thresh"]

    scores = {"RMSE": [], "MSE": [], "MAE": [], f"Precision_{k}": [], f"Recall_{k}": [], f"F1_{k}": [], "NDCG": [], "MAP": [], "MRR": [], "Hit-Rate": []}

    # Repeat evaluation to ensure stability
    print("Evaluating model...")
    for i in range(10):
        # Get true scores for current query
        q_id = random.randint(0, len(data.ingredients))
        q_ings = X_ings[q_id]
        true_scores = cosine_similarity(q_ings, X_ings)[0]

        # Get predicted scores for current query
        pred_scores = model.predict(data.ingredients.iloc[q_id])

        # Get top recommendations and its true similarity
        recommended_idx = np.argsort(pred_scores)[::-1][:params["model_service"]["n_recs"]]
        true_preds = true_scores[recommended_idx]

        # Calculate predictive metrics
        ## RMSE
        rmse = np.sqrt(np.mean((true_scores - pred_scores)**2))
        scores["RMSE"].append(rmse)

        ## MSE
        mse = np.mean((true_scores - pred_scores)**2)
        scores["MSE"].append(mse)

        ## MAE
        mae = np.mean(np.abs(true_scores - pred_scores))
        scores["MAE"].append(mae)

        ## Precision@K
        precision_k = true_preds[:k].mean()
        scores[f"Precision_{k}"].append(precision_k)

        ## Recall@K
        recall_k = true_preds[:k].sum()/true_preds.sum() 
        scores[f"Recall_{k}"].append(recall_k)

        ## F1@K
        f1_k = (2 * precision_k * recall_k) / (precision_k + recall_k) 
        scores[f"F1_{k}"].append(f1_k)

        # Calculate ranking metrics
        ## NDCG
        sorted_recs = sorted(true_scores, reverse=True)
        dcg, idcg = 0, 0
        for i in range(len(true_scores)):
            dcg += true_scores[i]/np.log2((i+1)+1)
            idcg += sorted_recs[i]/np.log2((i+1)+1)

        idcg += 1e-10  # Avoid dividing by 0
        scores["NDCG"].append(dcg/idcg)

        ## MAP
        relevant_preds = true_preds[1:] > relevant_thresh
        relevant_count = np.cumsum(relevant_preds)[relevant_preds == 1]
        relevant_idx = np.arange(1, len(relevant_preds) + 1)[relevant_preds == 1]
        ap = np.nanmean((relevant_count / relevant_idx))
        scores["MAP"].append(ap if not np.isnan(ap) else 0)

        ## MRR
        first_relevant_idx = relevant_idx[0] if len(relevant_idx) > 0 else 0
        rr = 1/first_relevant_idx if first_relevant_idx else 0
        scores["MRR"].append(rr)

        ## Hit-Rate
        hit = 1 if True in relevant_preds[:k] else 0
        scores["Hit-Rate"].append(hit)

    # Print metrics
    print("Model evaluation metrics:")
    for metric in scores:
        print(f"{metric:>15}:", np.mean(scores[metric]))

    # Compile model metadata
    metric_scores = {}
    for metric in scores:
        metric_scores[metric] = np.mean(scores[metric])

    validation_metadata = {**metadata, "metrics": metric_scores}

    # Save validation model metadata
    with open("validation_metadata.json", "w") as f:
        f.write(json.dumps(validation_metadata, indent=4))

    # Set up MLflow tracking
    print("Logging to MLflow...")
    # Log metrics to MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("FastText-Model")

    with mlflow.start_run(run_id=metadata["run_id"]):
        mlflow.log_metrics(metric_scores)


if __name__ == "__main__":
    print(f"{'='*20} Starting model validation pipeline. {'='*20}")
    main()
    print(f"{'='*20} Model validation pipeline complete. {'='*20}")
