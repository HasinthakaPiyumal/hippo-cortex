"""Mirror a run's checkpoint folder to a private Hugging Face Hub repo, so a run survives the
end of a Kaggle/RunPod session. The token is read from the HF_TOKEN environment variable.

Layout in the repo: <run_name>/<file>, the same as <ckpt_dir>/<run_name>/<file> on disk.
Upload errors are printed, never raised, so a network hiccup does not kill training.
"""
import os

from huggingface_hub import HfApi, hf_hub_download


class HubMirror:
    def __init__(self, repo_id: str, run_name: str, ckpt_dir: str, squash: bool = False):
        self.api = HfApi(token=os.environ.get("HF_TOKEN"))
        self.repo_id, self.run_name, self.ckpt_dir, self.squash = repo_id, run_name, ckpt_dir, squash
        self.api.create_repo(repo_id, private=True, exist_ok=True)

    def pull(self, filenames: list[str]) -> list[str]:
        """Download the files this run has in the repo but not on disk; returns what was downloaded."""
        remote = set(self.api.list_repo_files(self.repo_id))
        pulled = []
        for fn in filenames:
            path_in_repo = f"{self.run_name}/{fn}"
            if path_in_repo in remote and not os.path.exists(os.path.join(self.ckpt_dir, path_in_repo)):
                hf_hub_download(self.repo_id, path_in_repo, local_dir=self.ckpt_dir, token=self.api.token)
                pulled.append(fn)
        return pulled

    def push(self, filename: str, message: str) -> None:
        try:
            self.api.upload_file(path_or_fileobj=os.path.join(self.ckpt_dir, self.run_name, filename),
                                 path_in_repo=f"{self.run_name}/{filename}", repo_id=self.repo_id, commit_message=message)
        except Exception as e:
            print(f":: HF upload of {filename} failed ({type(e).__name__}: {e}); the local copy is kept", flush=True)

    def squash_history(self) -> None:
        """Drop old checkpoint versions so repo storage stays at one copy of each file."""
        if not self.squash:
            return
        try:
            self.api.super_squash_history(self.repo_id)
        except Exception as e:
            print(f":: HF history squash failed ({type(e).__name__}: {e})", flush=True)
