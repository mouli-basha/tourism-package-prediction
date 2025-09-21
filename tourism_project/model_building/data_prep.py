"""
Loads the raw dataset from the Hugging Face Datasets repo, performs light/essential
cleaning, splits into train/test (stratified), saves local CSVs, and uploads the
splits back to a dedicated HF dataset repo.

This file is called by the GitHub Actions job: 'data-prep'.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")

RAW_DATASET_REPO   = os.getenv(  # where the raw CSV lives
    "RAW_DATASET_REPO",
    "moulibasha/tourism-package-prediction-dataset"
)
SPLIT_DATASET_REPO = os.getenv(  # where train.csv and test.csv will be published
    "SPLIT_DATASET_REPO",
    "moulibasha/tourism-package-prediction-train-test"
)

TARGET = os.getenv("TARGET_COLUMN", "ProdTaken")  # supervised target column


# Helpers
def _clean_minimal(df: pd.DataFrame) -> pd.DataFrame:
    # Drop non-predictive columns if present
    for col in ("Unnamed: 0", "CustomerID"):
        if col in df.columns:
            df = df.drop(columns=col)

    # Normalize object (string) columns
    for col in df.select_dtypes(include="object").columns:
        # cast to str, strip whitespace, lowercase; preserve NaN
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"nan": np.nan})
        )

    # Validate/format target
    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' not found in dataset columns: {df.columns.tolist()}")

    # Coerce target to integers (0/1); fill missing as 0 to keep pipeline simple
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    print(f"[clean] duplicates_removed={removed}")

    return df


def _split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified 80/20 split to preserve label distribution in train/test.
    """
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    train = pd.concat([Xtr.reset_index(drop=True), ytr.reset_index(drop=True)], axis=1)
    test  = pd.concat([Xte.reset_index(drop=True), yte.reset_index(drop=True)], axis=1)
    return train, test


def _publish_splits_to_hf(train_path: str, test_path: str) -> None:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set. Provide it via environment (GitHub Actions secret).")

    api = HfApi(token=HF_TOKEN)

    # Ensure the destination dataset repo exists
    try:
        api.repo_info(repo_id=SPLIT_DATASET_REPO, repo_type="dataset")
        print(f"[hf] dataset repo '{SPLIT_DATASET_REPO}' exists")
    except RepositoryNotFoundError:
        create_repo(repo_id=SPLIT_DATASET_REPO, repo_type="dataset", private=False, token=HF_TOKEN)
        print(f"[hf] created dataset repo '{SPLIT_DATASET_REPO}'")

    # Upload files
    api.upload_file(
        path_or_fileobj=train_path,
        path_in_repo="train.csv",
        repo_id=SPLIT_DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    api.upload_file(
        path_or_fileobj=test_path,
        path_in_repo="test.csv",
        repo_id=SPLIT_DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print(f"[hf] published splits -> https://huggingface.co/datasets/{SPLIT_DATASET_REPO}")


# Main entrypoint
def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required (export as env var).")

    # Load raw dataset directly from HF
    print(f"[load] loading from hf://datasets/{RAW_DATASET_REPO} (data/tourism.csv)")
    ds = load_dataset(RAW_DATASET_REPO, data_files="data/tourism.csv")
    df = ds["train"].to_pandas()
    print(f"[load] shape={df.shape}, columns={list(df.columns)}")

    # Clean
    df = _clean_minimal(df)
    print(f"[clean] post-clean shape={df.shape}")

    # Split and save locally
    train, test = _split(df)
    train_path = "train.csv"
    test_path  = "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"[save] train->{train_path} shape={train.shape}, test->{test_path} shape={test.shape}")

    # Upload train/test back to HF dataset space
    _publish_splits_to_hf(train_path, test_path)


if __name__ == "__main__":
    main()
