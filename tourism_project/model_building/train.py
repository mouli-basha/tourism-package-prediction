"""
Model Training with Experiment Tracking:
- Loads train/test from Hugging Face Datasets
- Builds a preprocessing+model pipeline and a param grid
- Tunes via GridSearchCV and logs tuned params to MLflow
- Evaluates on held-out test set and logs metrics
- Registers/pushes the best model to Hugging Face Model Hub
"""

import os, joblib, numpy as np, pandas as pd, mlflow
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Config
HF_TOKEN = os.getenv("HF_TOKEN")
SPLIT_DATASET_REPO = os.getenv("SPLIT_DATASET_REPO", "moulibasha/tourism-package-prediction-train-test")
MODEL_REPO         = os.getenv("MODEL_REPO",         "moulibasha/tourism-package-prediction-model")

RANDOM_SEED = 42  # keep runs reproducible


def main():
    assert HF_TOKEN, "HF_TOKEN is required"

    # Load train/test directly from HF datasets
    train = pd.read_csv(f"https://huggingface.co/datasets/{SPLIT_DATASET_REPO}/resolve/main/train.csv")
    test  = pd.read_csv(f"https://huggingface.co/datasets/{SPLIT_DATASET_REPO}/resolve/main/test.csv")

    ytr = train["ProdTaken"].astype(int); Xtr = train.drop(columns=["ProdTaken"])
    yte = test["ProdTaken"].astype(int);  Xte = test.drop(columns=["ProdTaken"])

    # Identify numeric/categorical columns once
    num = Xtr.select_dtypes(include=[np.number]).columns.tolist()
    cat = Xtr.select_dtypes(exclude=[np.number]).columns.tolist()

    # Define preprocessing & model pipeline and parameter grid
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])

    # Using Random Forest
    pipe = Pipeline([("pre", pre), ("model", RandomForestClassifier(random_state=RANDOM_SEED))])

    grid = {
        "model__n_estimators": [150, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)

    # Tune and log tuned params to MLflow
    mlflow.set_experiment("tourism_rf_experiment")
    with mlflow.start_run(run_name="rf_gridsearch"):
        gs.fit(Xtr, ytr)

        # Log the full best-parameter set for auditability
        for k, v in gs.best_params_.items():
            mlflow.log_param(k, v)

        # Evaluate on held-out test set and log metrics
        preds = gs.best_estimator_.predict(Xte)
        metrics = {
            "accuracy":  float(accuracy_score(yte, preds)),
            "precision": float(precision_score(yte, preds, zero_division=0)),
            "recall":    float(recall_score(yte, preds, zero_division=0)),
            "f1":        float(f1_score(yte, preds, zero_division=0)),
        }
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    # Persist the tuned best estimator locally
    os.makedirs("models", exist_ok=True)
    joblib.dump(gs.best_estimator_, "models/model.pkl")

    # Lightweight model card
    readme = f"""# Tourism Package Prediction Model

- **Data:** [tourism-package-prediction-train-test](https://huggingface.co/datasets/moulibasha/tourism-package-prediction-train-test)
- **Best params:** {gs.best_params_}
- **Metrics:** {metrics}
- **Pipeline:** preprocessing (imputer + onehot) + RandomForest
"""
    with open("models/README.md", "w") as f:
        f.write(readme)

    # Register best model in HF Model Hub
    api = HfApi(token=HF_TOKEN)
    try:
        api.repo_info(repo_id=MODEL_REPO, repo_type="model")
        print(f"Model repo '{MODEL_REPO}' exists.")
    except RepositoryNotFoundError:
        create_repo(repo_id=MODEL_REPO, repo_type="model", private=False, token=HF_TOKEN)
        print(f"Created model repo: {MODEL_REPO}")

    api.upload_file("models/model.pkl",  "model.pkl",
                    repo_id=MODEL_REPO, repo_type="model", token=HF_TOKEN)
    api.upload_file("models/README.md", "README.md",
                    repo_id=MODEL_REPO, repo_type="model", token=HF_TOKEN)

    print(f" Model pushed: https://huggingface.co/{MODEL_REPO}  |  metrics={metrics}")


if __name__ == "__main__":
    main()
