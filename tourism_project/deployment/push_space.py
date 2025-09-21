
SPACE_ID = "moulibasha/tourism-package-prediction"

def get_token():
    return os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN_TPP") or ""

def main():
    token = get_token()

    # Ensure the Space exists and is Streamlit
    try:
        create_repo(repo_id=SPACE_ID, repo_type="space", space_sdk="streamlit", private=False, token=token)
    except Exception:
        pass

    # Upload only the required deployment files
    base = os.path.dirname(__file__)
    files = [
        ("app.py", "app.py"),
        ("requirements.txt", "requirements.txt"),
        ("Dockerfile", "Dockerfile"),
    ]

    for local, remote in files:
        local_path = os.path.join(base, local)
        if os.path.exists(local_path):
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote,
                repo_id=SPACE_ID,
                repo_type="space",
                token=token,
            )
            print(f"Uploaded {local} -> {SPACE_ID}/{remote}")

    print(f" Space updated: https://huggingface.co/spaces/{SPACE_ID}")

if __name__ == "__main__":
    main()
