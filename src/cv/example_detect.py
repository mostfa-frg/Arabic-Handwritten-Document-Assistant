"""
Quick example – detect lines on a Muharaf page with the best available model.

Usage:
    # 1. Download the Muharaf Kraken model (once)
    python -m src.cv.download_muharaf_seg_model

    # 2. Run detection
    python -m src.cv.example_detect path/to/muharaf_page.jpg
"""

import sys
from pathlib import Path
from PIL import Image, ImageDraw

# allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cv.line_detector import get_best_detector_for_muharaf


def main(image_path: str):
    detector = get_best_detector_for_muharaf(
        muharaf_model_path="models/muharaf_seg_best.mlmodel",
        device="cpu",
    )
    print(f"Using backend: {detector.backend}")

    lines = detector.detect(image_path, return_crops=True)
    print(f"Detected {len(lines)} text lines (top → bottom)")

    # visualise
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(0, y1 - 12)), str(i), fill="red")

    out = Path("detected_lines_preview.jpg")
    img.save(out)
    print(f"Preview saved → {out}")

    # show first few crops info
    for i, line in enumerate(lines[:3]):
        print(f"  line {i}: bbox={line['bbox']}  conf={line.get('confidence', 1.0):.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.cv.example_detect <image.jpg>")
        sys.exit(1)
    main(sys.argv[1])
