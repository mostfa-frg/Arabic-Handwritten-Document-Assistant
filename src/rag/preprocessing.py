import re
import unicodedata


def clean_arabic_text(text: str) -> str:
    """
    Conservative Arabic text normalization.
    """

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFC",
        str(text)
    )

    # Remove Tatweel
    text = text.replace("ـ", "")

    # Remove Arabic diacritics
    text = re.sub(
        r"[\u0617-\u061A\u064B-\u065F\u0670]",
        "",
        text
    )

    # Normalize Arabic whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text