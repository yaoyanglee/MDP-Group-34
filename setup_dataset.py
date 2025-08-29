# setup_dataset.py
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path


# This helper downloads a small prepared archive (user may replace with direct GTSRB URL)
# For the sake of testing quickly, you could prepare a small archive with a few labeled images and point DOWNLOAD_URL to it.


DOWNLOAD_URL = "https://example.com/small_gtsrb_sample.zip" # replace with your prepared sample or manual extraction
OUT_DIR = Path("data")


if __name__ == '__main__':
    OUT_DIR.mkdir(exist_ok=True)
# If you have GTSRB locally, skip download and prepare a CSV with image paths and labels
print("Dataset setup helper. Replace DOWNLOAD_URL with an accessible sample archive and rerun.")