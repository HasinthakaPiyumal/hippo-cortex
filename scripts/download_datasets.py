"""
Dataset download script.

OWNER: Induwara Mihisara

Usage:
    python scripts/download_datasets.py --root data/ --dataset cifar100
    python scripts/download_datasets.py --root data/ --dataset imagenet_r
    python scripts/download_datasets.py --root data/ --dataset all
"""
from __future__ import annotations

import argparse
from pathlib import Path


def download_cifar100(root: Path) -> None:
    cifar_dir = root / "cifar100"
    cifar_dir.mkdir(parents=True, exist_ok=True)
    tar_file = cifar_dir / "cifar-100-python.tar.gz"
    extract_dir = cifar_dir / "cifar-100-python"

    if not extract_dir.exists():
        if not tar_file.exists():
            print("Downloading CIFAR-100 from Google Drive via gdown...")
            import gdown
            file_id = "1SjQ7aL1NHX9DqeC72oqmX82lzuv7-FIM"
            gdown.download(id=file_id, output=str(tar_file), quiet=False)

        print("Extracting CIFAR-100 dataset...")
        import tarfile
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(path=str(cifar_dir))
        print("CIFAR-100 extraction complete.")
    else:
        print("CIFAR-100 dataset already extracted.")


def download_imagenet_r(root: Path) -> None:
    imagenet_dir = root / "imagenet-r"
    tar_file = root / "imagenet-r.tar"

    if not imagenet_dir.exists() or not any(imagenet_dir.glob("n*")):
        root.mkdir(parents=True, exist_ok=True)
        if not tar_file.exists():
            print("Downloading ImageNet-R from UC Berkeley (Hendrycks)...")
            import urllib.request
            from tqdm import tqdm
            
            url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
            
            class DownloadProgressBar(tqdm):
                def update_to(self, b=1, bsize=1, tsize=None):
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)

            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
                urllib.request.urlretrieve(url, filename=tar_file, reporthook=t.update_to)

        print("Extracting ImageNet-R dataset...")
        import tarfile
        with tarfile.open(tar_file, "r") as tar:
            tar.extractall(path=str(root))
        print("ImageNet-R extraction complete.")
        
        # Cleanup tar file to save disk space
        try:
            tar_file.unlink()
            print("Cleaned up tar file.")
        except Exception:
            pass
    else:
        print("ImageNet-R dataset already extracted.")




def main() -> None:
    parser = argparse.ArgumentParser(description="Download HippoCortex datasets.")
    parser.add_argument("--root", type=Path, default=Path("data"), help="Download target directory.")
    parser.add_argument(
        "--dataset",
        choices=["cifar100", "imagenet_r", "all"],
        default="all",
    )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("cifar100", "all"):
        download_cifar100(args.root)

    if args.dataset in ("imagenet_r", "all"):
        download_imagenet_r(args.root)


if __name__ == "__main__":
    main()
