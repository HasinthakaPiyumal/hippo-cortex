# Save as baselines/mamba-cl/tools/extract_cifar100.py
import os
import numpy as np
import pickle
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f, encoding='bytes')

# Resolve paths relative to the script location (robust for any instance)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../.."))

RAW_DIR = os.path.join(project_root, "data/cifar100/cifar-100-python")
OUT_DIR = os.path.join(project_root, "data/cifar100-images")


meta = unpickle(os.path.join(RAW_DIR, "meta"))
fine_labels = [l.decode() for l in meta[b'fine_label_names']]

for split in ("train", "test"):
    fname = "train" if split == "train" else "test"
    data = unpickle(os.path.join(RAW_DIR, fname))
    images = data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = data[b'fine_labels']
    filenames = [f.decode() for f in data[b'filenames']]
    out_split = "val" if split == "test" else "train"
    for img_arr, label, filename in zip(images, labels, filenames):
        folder = os.path.join(OUT_DIR, out_split, str(label))
        os.makedirs(folder, exist_ok=True)
        Image.fromarray(img_arr).save(os.path.join(folder, filename.replace(".png", "") + ".png"))


print("Done.")