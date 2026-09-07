"""
Stronger text-line detectors for Arabic handwritten pages (Muharaf-ready).

Supported backends (in priority order for Muharaf):
1. Kraken + Muharaf-trained segmentation model  (best domain match)
2. Surya                                       (strong modern multilingual)
3. YOLO (legacy / your previous model)         (kept for compatibility)

Usage:
    from src.cv.line_detector import LineDetector

    detector = LineDetector(backend="kraken", model_path="models/muharaf_seg_best.mlmodel")
    lines = detector.detect(image_path_or_pil)
    # lines = list of dicts: {"bbox": [x1,y1,x2,y2], "polygon": [...], "crop": PIL.Image, ...}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class LineDetector:
    """Unified interface for text-line detection on Arabic handwritten pages."""

    def __init__(
        self,
        backend: str = "kraken",
        model_path: Optional[str] = None,
        device: str = "cpu",
        conf_threshold: float = 0.25,
    ):
        """
        Args:
            backend: "kraken" | "surya" | "yolo"
            model_path: path to model weights (required for kraken/yolo)
            device: "cpu" or "cuda"
            conf_threshold: used by YOLO / some filters
        """
        self.backend = backend.lower()
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self._model = None
        self._load_model()

    def _load_model(self):
        if self.backend == "kraken":
            self._load_kraken()
        elif self.backend == "surya":
            self._load_surya()
        elif self.backend == "yolo":
            self._load_yolo()
        else:
            raise ValueError(f"Unknown backend: {self.backend}. Use 'kraken', 'surya' or 'yolo'.")

    # ------------------------------------------------------------------
    # Kraken (Muharaf-trained preferred)
    # ------------------------------------------------------------------
    def _load_kraken(self):
        try:
            from kraken import blla
            from kraken.lib.vgsl import TorchVGSLModel
        except ImportError as e:
            raise ImportError(
                "kraken is required for the Kraken backend.\n"
                "Install with:  pip install kraken"
            ) from e

        if self.model_path is None:
            # fallback to default BLLA if no Muharaf model given
            logger.warning("No model_path given → using Kraken default BLLA model")
            self._model = None  # blla.segment will use default
        else:
            path = Path(self.model_path)
            if not path.exists():
                raise FileNotFoundError(f"Kraken model not found: {path}")
            logger.info(f"Loading Kraken segmentation model from {path}")
            self._model = TorchVGSLModel.load_model(str(path))

        self._kraken_blla = blla

    def _detect_kraken(self, image: Image.Image) -> List[Dict[str, Any]]:
        # Kraken expects RGB or L
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        if self._model is None:
            seg = self._kraken_blla.segment(image)
        else:
            seg = self._kraken_blla.segment(image, model=self._model)

        results = []
        # sort top-to-bottom (important for Arabic documents)
        lines = sorted(seg.lines, key=lambda l: l.baseline[0][1] if l.baseline else 0)

        for line in lines:
            # baseline → approximate bounding box
            xs = [p[0] for p in line.baseline] + [p[0] for p in (line.boundary or [])]
            ys = [p[1] for p in line.baseline] + [p[1] for p in (line.boundary or [])]
            if not xs or not ys:
                continue
            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))
            # small padding
            pad = 4
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(image.width, x2 + pad), min(image.height, y2 + pad)

            crop = image.crop((x1, y1, x2, y2))
            results.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "polygon": line.boundary or line.baseline,
                    "baseline": line.baseline,
                    "crop": crop,
                    "confidence": getattr(line, "confidence", 1.0),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Surya
    # ------------------------------------------------------------------
    def _load_surya(self):
        try:
            from surya.detection import DetectionPredictor
        except ImportError as e:
            raise ImportError(
                "surya-ocr is required for the Surya backend.\n"
                "Install with:  pip install surya-ocr"
            ) from e

        logger.info("Loading Surya DetectionPredictor")
        self._model = DetectionPredictor()

    def _detect_surya(self, image: Image.Image) -> List[Dict[str, Any]]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        preds = self._model([image])
        page = preds[0]

        results = []
        # sort top-to-bottom
        boxes = sorted(page.bboxes, key=lambda b: b.bbox[1])

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.bbox)
            # clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.width, x2), min(image.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image.crop((x1, y1, x2, y2))
            results.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "polygon": getattr(box, "polygon", None),
                    "crop": crop,
                    "confidence": getattr(box, "confidence", 1.0),
                }
            )
        return results

    # ------------------------------------------------------------------
    # YOLO (legacy – your previous model)
    # ------------------------------------------------------------------
    def _load_yolo(self):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("ultralytics is required for YOLO backend") from e

        if self.model_path is None:
            raise ValueError("model_path is required for YOLO backend")
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {path}")

        logger.info(f"Loading YOLO model from {path}")
        self._model = YOLO(str(path))

    def _detect_yolo(self, image: Image.Image) -> List[Dict[str, Any]]:
        # YOLO expects BGR numpy or path; we pass PIL → numpy RGB then convert
        img_np = np.array(image.convert("RGB"))
        results = self._model.predict(
            source=img_np,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device,
        )

        detections = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        # sort top-to-bottom
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        order = np.argsort(xyxy[:, 1])  # sort by y1

        for i in order:
            x1, y1, x2, y2 = map(int, xyxy[i])
            conf = float(confs[i])
            crop = image.crop((x1, y1, x2, y2))
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "polygon": None,
                    "crop": crop,
                    "confidence": conf,
                }
            )
        return detections

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(
        self,
        image: Union[str, Path, Image.Image],
        return_crops: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Detect text lines and return sorted (top→bottom) results.

        Returns list of:
            {
                "bbox": [x1, y1, x2, y2],
                "polygon": optional list of points,
                "crop": PIL.Image (if return_crops=True),
                "confidence": float
            }
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError("image must be path or PIL.Image")

        if self.backend == "kraken":
            lines = self._detect_kraken(image)
        elif self.backend == "surya":
            lines = self._detect_surya(image)
        else:
            lines = self._detect_yolo(image)

        if not return_crops:
            for line in lines:
                line.pop("crop", None)

        logger.info(f"[{self.backend}] detected {len(lines)} text lines")
        return lines


# Convenience factory
def get_best_detector_for_muharaf(
    muharaf_model_path: str = "models/muharaf_seg_best.mlmodel",
    device: str = "cpu",
) -> LineDetector:
    """Return the recommended detector for Muharaf pages."""
    path = Path(muharaf_model_path)
    if path.exists():
        return LineDetector(backend="kraken", model_path=str(path), device=device)
    logger.warning(
        f"Muharaf Kraken model not found at {path}. Falling back to Surya."
    )
    return LineDetector(backend="surya", device=device)
