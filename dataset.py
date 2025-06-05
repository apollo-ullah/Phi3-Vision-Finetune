#!/usr/bin/env python3
"""
Grab VQAv2 (questions + annotations) and the relevant MS-COCO images.
Total size ≈ 25 GB. Adjust DEST if you want a different target directory.
"""

import os
import requests
from tqdm import tqdm
import zipfile
import json
import glob
import random

DEST = "./vqav2_data"

# VQA v2 JSON archives
VQA_URLS = {
    "v2_Questions_Train_mscoco.zip":
        "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "v2_Questions_Val_mscoco.zip":
        "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "v2_Questions_Test_mscoco.zip":
        "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",
    "v2_Annotations_Train_mscoco.zip":
        "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "v2_Annotations_Val_mscoco.zip":
        "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
}

# COCO 2014/2015 image zips (train, val, test)
COCO_URLS = {
    "train2014.zip": "http://images.cocodataset.org/zips/train2014.zip",
    "val2014.zip":   "http://images.cocodataset.org/zips/val2014.zip",
    "test2015.zip":  "http://images.cocodataset.org/zips/test2015.zip",
}

URLS = {**VQA_URLS, **COCO_URLS}


def download(url: str, out_path: str):
    """Stream download with tqdm progress bar."""
    if os.path.exists(out_path):
        print(f"[✓] {os.path.basename(out_path)} already exists, skipping.")
        return
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True,
                   desc=f"⇩ {os.path.basename(out_path)}")
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        bar.close()


def safe_extract(zip_path: str, dest_dir: str):
    """Unzip and remove archive."""
    print(f"↪ Extracting {os.path.basename(zip_path)} …")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    os.remove(zip_path)


def sanity_check():
    """Quick sanity-check of downloaded files."""
    print("\n=== Sanity Check ===")
    json_files = glob.glob(f"{DEST}/*json")
    if json_files:
        samples = random.sample(json_files, min(3, len(json_files)))
        for f in samples:
            try:
                data = json.load(open(f))
                print(f"{os.path.basename(f)}: {len(data)} items")
            except Exception as e:
                print(f"{os.path.basename(f)}: Error loading - {e}")
    else:
        print("No JSON files found")
    
    # Check image directories
    for img_dir in ["train2014", "val2014", "test2015"]:
        dir_path = os.path.join(DEST, img_dir)
        if os.path.exists(dir_path):
            img_count = len([f for f in os.listdir(dir_path) if f.endswith('.jpg')])
            print(f"{img_dir}: {img_count} images")
        else:
            print(f"{img_dir}: Directory not found")


def main():
    os.makedirs(DEST, exist_ok=True)
    
    print("Starting VQAv2 dataset download...")
    print(f"Destination: {DEST}")
    print(f"Total files to download: {len(URLS)}")

    # Download all zips
    for fname, url in URLS.items():
        print(f"\n--- Downloading {fname} ---")
        download(url, os.path.join(DEST, fname))

    print("\n--- Extracting archives ---")
    # Unpack and clean up
    for fname in URLS.keys():
        zip_path = os.path.join(DEST, fname)
        if zip_path.endswith(".zip") and os.path.exists(zip_path):
            safe_extract(zip_path, DEST)

    print("\nDone! JSON files live under DEST/ and images under "
          "DEST/train2014, DEST/val2014, DEST/test2015.")
    
    # Run sanity check
    sanity_check()


if __name__ == "__main__":
    main()