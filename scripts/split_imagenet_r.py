import os
import os.path as osp
import shutil

# Resolve paths relative to the script location (scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

root_dir = os.path.join(project_root, "data/imagenet-r")
split_lists_dir = os.path.join(project_root, "baselines/mamba-cl/tools")

if not os.path.exists(root_dir):
    # Berkeley extracts to a folder named "imagenet-r"
    alt_root_dir = os.path.join(project_root, "data")
    if os.path.exists(os.path.join(alt_root_dir, "n01443537")):
        root_dir = alt_root_dir
    else:
        raise FileNotFoundError(
            f"ImageNet-R raw data not found. Please run 'python scripts/download_datasets.py --dataset imagenet_r' first."
        )

def split_func(mode):
    mode_dir = osp.join(root_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)
    
    split_file = osp.join(split_lists_dir, f"imagenet_r_{mode}.txt")
    if not osp.exists(split_file):
        raise FileNotFoundError(f"Split file list not found at {split_file}")
        
    with open(split_file, "r") as f:
        lines = f.readlines()
        
    print(f"Splitting ImageNet-R into {mode} directory...")
    for line in lines:
        rel_path = line.strip('\n')
        # Remove potential top-level "imagenet-r" from relative path if we are already inside it
        if rel_path.startswith("imagenet-r/"):
            clean_rel_path = rel_path.replace("imagenet-r/", "", 1)
        else:
            clean_rel_path = rel_path
            
        src_path = osp.realpath(osp.join(root_dir, clean_rel_path))
        if not osp.exists(src_path):
            print(f"Warning: source file {src_path} not found.")
            continue
            
        # Class directory is the directory containing the file
        class_name = osp.basename(osp.dirname(clean_rel_path))
        class_dir = osp.realpath(osp.join(mode_dir, class_name))
        os.makedirs(class_dir, exist_ok=True)
        
        file_name = osp.basename(src_path)
        dst_path = osp.join(class_dir, file_name)
        
        # Avoid recreating if it exists
        if osp.exists(dst_path) or osp.islink(dst_path):
            continue
            
        try:
            os.symlink(src_path, dst_path)
        except Exception:
            shutil.copy(src_path, dst_path)

split_func('train')
split_func('val')
print("Done. ImageNet-R split completed successfully!")
