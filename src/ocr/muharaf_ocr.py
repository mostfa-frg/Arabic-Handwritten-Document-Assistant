from __future__ import annotations

import importlib
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


class MuharafOCR:
    """
    Wrapper around the Muharaf Arabic handwritten OCR model.
    """

    REPO_ID = "sdkv2/muharaf-arabic-ocr"

    def __init__(self):
        self.model_path = None
        self.ocr_model = None
        self.charset = None
        self.cfg = None

        self._load()

    def _load(self) -> None:

     repo_path = snapshot_download(
        repo_id=self.REPO_ID
     )

     self.model_path = Path(repo_path)

     repo_path_str = str(self.model_path)

     if repo_path_str not in sys.path:
        sys.path.insert(0, repo_path_str)

     import importlib

     torch_models = importlib.import_module(
        "submission_code.torch_models"
     )

     load_ctc_backbone = (
        torch_models.load_ctc_backbone
     )

     self._recognise_line = (
        torch_models.recognise_line
    )

     model_file = (
        self.model_path
        / "models"
        / "ctc_backbone"
        / "model.pt"
     )

     (
        self.ocr_model,
        self.charset,
        self.cfg,
     ) = load_ctc_backbone(
        str(model_file)
     )
     

    def recognize_line(self, image):
        """
        Recognize one handwritten Arabic line.
        """
        return self._recognise_line(
            self.ocr_model,
            self.charset,
            image,
        )

    def recognize_page(self, lines) -> str:
        """
        OCR a list of detected line dictionaries.
        """
        texts = []

        for line in lines:
            text = self.recognize_line(
                line["crop"]
            )

            text = str(text).strip()

            if text:
                texts.append(text)

        return "\n".join(texts)