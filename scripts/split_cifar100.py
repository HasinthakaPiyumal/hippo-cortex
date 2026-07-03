import os
import pickle
import numpy as np
from PIL import Image

# Resolve paths relative to the script location (scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

RAW_DIR = os.path.join(project_root, "data/cifar100/cifar-100-python")
OUT_DIR = os.path.join(project_root, "data/cifar100-images")

def unpickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f, encoding='bytes')

if not os.path.exists(RAW_DIR):
    raise FileNotFoundError(
        f"Raw CIFAR-100 data not found at {RAW_DIR}. "
        f"Please run 'python scripts/download_datasets.py --dataset cifar100' first."
    )

print("Reading raw CIFAR-100 files...")
meta = unpickle(os.path.join(RAW_DIR, "meta"))
fine_labels = [l.decode() for l in meta[b'fine_label_names']]

for split in ("train", "test"):
    fname = "train" if split == "train" else "test"
    data = unpickle(os.path.join(RAW_DIR, fname))
    images = data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = data[b'fine_labels']
    filenames = [f.decode() for f in data[b'filenames']]
    out_split = "val" if split == "test" else "train"
    
    print(f"Splitting {split} split into {out_split} class directories...")
    for img_arr, label, filename in zip(images, labels, filenames):
        # Mamba-CL dataset builder expects integer folders as class names
        folder = os.path.join(OUT_DIR, out_split, str(label))
        os.makedirs(folder, exist_ok=True)
        Image.fromarray(img_arr).save(os.path.join(folder, filename.replace(".png", "") + ".png"))

print("Done. CIFAR-100 split completed successfully!")
