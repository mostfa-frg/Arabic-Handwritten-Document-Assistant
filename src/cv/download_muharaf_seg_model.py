"""
Download the Kraken segmentation model trained on Muharaf.
Run once:

    python -m src.cv.download_muharaf_seg_model
"""

from pathlib import Path
import urllib.request
import sys

MODEL_URL = "https://zenodo.org/records/14295555/files/muharaf_seg_best.mlmodel"
TARGET = Path("models/muharaf_seg_best.mlmodel")


def main():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        print(f"Model already exists: {TARGET}")
        return

    print(f"Downloading Muharaf Kraken segmentation model …")
    print(f"  URL  : {MODEL_URL}")
    print(f"  Dest : {TARGET}")

    try:
        urllib.request.urlretrieve(MODEL_URL, TARGET)
        size_mb = TARGET.stat().st_size / (1024 * 1024)
        print(f"Done. Size: {size_mb:.1f} MB")
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
