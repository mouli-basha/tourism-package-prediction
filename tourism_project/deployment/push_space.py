
import os
from huggingface_hub import create_repo, upload_file

SPACE_ID = "moulibasha/tourism-package-prediction"

def get_token():
    return os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN_TPP") or ""

def main():
    token = get_token()
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Add it as a GitHub Actions secret.")

    # Ensure the Space exists and is Streamlit
    try:
        create_repo(repo_id=SPACE_ID, repo_type="space", space_sdk="streamlit", private=False, token=token)
    except Exception:
        pass
    base = "tourism_project/deployment"
    # Upload only the required deployment files
    for local, remote in [("app.py","app.py"), ("requirements.txt","requirements.txt"), ("Dockerfile","Dockerfile")]:
        local_path = os.path.join(base, local)
        if os.path.exists(local_path):
            upload_file(path_or_fileobj=local_path, path_in_repo=remote,
                        repo_id=SPACE_ID, repo_type="space", token=token)
            print(f"Uploaded {local} -> {SPACE_ID}/{remote}")
    print(f" Space updated: https://huggingface.co/spaces/{SPACE_ID}")

if __name__ == "__main__":
    main()
