# Data Registration
import os
from huggingface_hub import HfApi, create_repo, upload_file
from huggingface_hub.utils import RepositoryNotFoundError

HF_DATASET_REPO = "moulibasha/tourism-package-prediction-dataset"
LOCAL_CSV_PATH  = "tourism_project/data/tourism.csv"

api = HfApi(token=os.environ["HF_TOKEN"])

try:
    api.repo_info(repo_id=HF_DATASET_REPO, repo_type="dataset")
    print(f"Dataset repo '{HF_DATASET_REPO}' exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", private=False, token=os.environ["HF_TOKEN"])
    print(f" Created dataset repo: {HF_DATASET_REPO}")

upload_file(
    path_or_fileobj=LOCAL_CSV_PATH,
    path_in_repo="data/tourism.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    token=os.environ["HF_TOKEN"]
)

print(f" Uploaded -> https://huggingface.co/datasets/{HF_DATASET_REPO}")
