# Food Recipes Recommendation

A recommender system that gives you recipes based on ingredients you have on-hand! Simply type in **any ingredients** and select **available nutrition filters** to fit your diet.

![app_preview](assets/web_preview.gif)
The website is accessible through [here](https://food-rec.streamlit.app/).

## Stack
- **Frontend**: Streamlit
- **Backend**: FastAPI, FastText, AWS EC2, and AWS ECR 
- **Version Control**: MLflow, and DVC 
- **CI/CD**: GitHub Actions
- Docker

## Local Run
**Dependencies:**
Install `requirements.txt` in your virtual environment. The app only depends on the model's API, while executing the pipeline will also require *MLflow* tracking server to be running. 

There are two ways to run all the dependencies:
- **Shell script**
    `start_servers.sh` will run *MLflow*, the model API, and *Streamlit* sequentially.
    `stop_servers.sh` will ensure all the servers started properly stopped.
- **Docker Compose**
    Simply run `docker compose -f compose.yaml up` to compose all the required service.

## Pipeline 
To run the pipeline, ensure *MLflow* server is running and execute `dvc repro` to run the pipeline from data preprocessing until model registering. The pipeline itself consists of 5 different stages: 
  1. **Data Extraction** (`notebooks/data_extraction.ipynb`)
   Data extraction is excluded in the pipeline due to variability and inconsistencies in scraping different recipe websites. Thus, It is done manually from the notebook for easier debugging and modification.
  2. **Data Preprocessing**
   Extracted data from `data/raw` is combined and processed into `processed_cookbook.pkl` and `final_cookbook.pkl` for app access. The pipeline utilizes *NLTK* and custom *regex* expression to clean titles, ingredients, and remove non-recipe data.
  3. **Model Training**
   Cleaned ingredients is fed into a *FastText* model from *Gensim* library (Full parameters can be accessed on `params.yaml`). The model is then tracked and wrapped within *MLflow* pyfunc model with addition of *IDF* and *Cosine Similarity* between ingredients and title with the query.
  4. **Model Evaluation**
   The model is evaluated using several metrics, such as *RMSE, MSE, MAE, Precision@10, Recall@10, F1@10, NDCG, MAP, MRR,* and *Hit-Rate* using *TF-IDF* as the ground truth to measure how exact the input query with the recommended recipes ingredients.
  5. **Model Registering**
    The evaluated model is then compared with the previous models using all of the metrics and registered through *MLflow*.

## Model
*FastText* model from *Gensim* library is used in this case due to its robust handling on Out-Of-Vocabulary (OOV) words. The parameters set for the model training are as follows:
- `vector_size=100`
- `window=8`
- `sg=0`
- `seed=1234`
- `epochs=100`
- `min_count=1`
- `bucket=50000` 

Such parameters are set to keep the file size under 100mb (while preserving the predictions quality) due to its n-grams architecture to handle OOV that could explode the file size easily. 

**Inverse Document Frequency (IDF)**
On top of the model, techniques like *Inverse Document Frequency (IDF)* are further added to reduce common ingredients that may cause weight imbalance in the similarity. To handle OOV in the *IDF*, custom value is assigned to each query using the top 3 most similar word in the vocabulary. The following weights are assigned sequentially from the most similar to the least: `[0.8, 0.1, 0.1]`. These values are then multiplied to the vector of each word in the query.

**Model Scoring**
The similarity scored is calculated by combining two similarities:
- **Ingredient Similarity**: *cosine similarity* between averaged vectors in the input query and averaged vectors in the recipe ingredients.
- **Title Similarity**: *cosine similarity* between averaged vectors in the input query and the averaged vectors in the recipe title.

The similarities are then added as such: `similarity = 0.9 * ingredient_sim + 0.1 * title_sim`

## Deployment 
The model itself is wrapped under *FastAPI* as an endpoint, which is then containerized by *Docker*, pushed into *AWS ECR*, and deployed using *AWS EC2* server. This is performed automatically using CI/CD from *GitHub Actions* every new push on the branch.

The website is hosted through *Streamlit Community Cloud* server which automatically updates from the GitHub repository every push.

## Attribution
All recipes belong to their respective authors: **Daily Dish Recipes**, **Just One Cookbook**, **Love and Lemons**, **Minimalist Baker**, **Omnivore’s Cookbook**, **RecipeTin Eats**, **Spoon Fork Bacon**, and **The Woks of Life**. This repository is used strictly for for **educational and research purposes**.
