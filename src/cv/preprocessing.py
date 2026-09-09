from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def to_grayscale(image: Image.Image | str | Path) -> np.ndarray:
    """Return an 8-bit grayscale image without changing its resolution."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            image = opened.copy()
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image or an image path")
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _crop_margins(gray: np.ndarray) -> np.ndarray:
    """Crop only clear, mostly empty borders; retain a generous ink margin."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, foreground = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    foreground = cv2.morphologyEx(
        foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    points = cv2.findNonZero(foreground)
    if points is None:
        return gray

    x, y, width, height = cv2.boundingRect(points)
    image_height, image_width = gray.shape
    # Avoid cropping documents whose foreground fills the frame or whose
    # border detection is likely to be dominated by noise.
    if (
        x < image_width * 0.03
        and y < image_height * 0.03
        and x + width > image_width * 0.97
        and y + height > image_height * 0.97
    ):
        return gray

    pad_x = max(12, int(image_width * 0.02))
    pad_y = max(12, int(image_height * 0.02))
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(image_width, x + width + pad_x)
    bottom = min(image_height, y + height + pad_y)
    return gray[top:bottom, left:right]


def deskew(gray: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
    """Correct small page skew while leaving heavily rotated images untouched."""
    foreground = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    points = cv2.findNonZero(foreground)
    if points is None or len(points) < 20:
        return gray

    angle = cv2.minAreaRect(points)[-1]
    if angle < -45:
        angle += 90
    if abs(angle) < 0.15 or abs(angle) > max_angle:
        return gray

    height, width = gray.shape
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """Improve faint handwriting locally without hard thresholding it."""
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    return clahe.apply(gray)


def normalize_background(gray: np.ndarray) -> np.ndarray:
    """Reduce gentle paper illumination gradients without flattening ink."""
    height, width = gray.shape
    sigma = max(10.0, min(height, width) / 20.0)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    if float(background.std()) < 4.0:
        return gray
    corrected = cv2.divide(gray, background, scale=float(background.mean()))
    return cv2.addWeighted(gray, 0.65, corrected, 0.35, 0)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Suppress small camera noise while retaining thin ink strokes."""
    return cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=30)


def remove_horizontal_lines(gray: np.ndarray) -> np.ndarray:
    """Remove strong notebook rules only when horizontal structure is detected."""
    binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    height, width = gray.shape
    kernel_width = max(30, width // 18)
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_width, 1)
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    line_pixels = int(cv2.countNonZero(horizontal))
    if line_pixels < max(100, int(width * height * 0.001)):
        return gray

    cleaned = cv2.subtract(binary, horizontal)
    return cv2.bitwise_not(cleaned)


def preprocess_document(image: Image.Image | str | Path) -> Image.Image:
    """Prepare a handwritten page for Kraken and Muharaf conservatively."""
    gray = to_grayscale(image)
    gray = _crop_margins(gray)
    gray = deskew(gray)
    gray = normalize_background(gray)
    gray = remove_horizontal_lines(gray)
    gray = denoise(gray)
    gray = enhance_contrast(gray)
    return Image.fromarray(gray, mode="L").convert("RGB")


def save_debug_outputs(
    image: Image.Image,
    lines: list[dict[str, Any]] | None = None,
    output_dir: str | Path = "debug/ocr",
) -> tuple[Path, list[Path]]:
    """Save optional preprocessing and line crops for local development."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    preprocessed_path = directory / "preprocessed.png"
    image.save(preprocessed_path)
    line_paths: list[Path] = []
    for index, line in enumerate(lines or [], start=1):
        crop = line.get("crop")
        if isinstance(crop, Image.Image):
            path = directory / f"line_{index:03d}.png"
            crop.save(path)
            line_paths.append(path)
    logger.info("Saved OCR debug image %s and %d line crops", preprocessed_path, len(line_paths))
    return preprocessed_path, line_paths