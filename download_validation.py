from datasets import load_dataset
from pathlib import Path

output_dir = Path("data/test_images/validation")
output_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset(
    "aamijar/muharaf-public",
    split="validation",
    streaming=True
)

for i, sample in enumerate(ds):
    if i >= 5:
        break

    image = sample["image"]
    text = sample["text"]

    image_path = output_dir / f"validation_{i+1}.png"
    text_path = output_dir / f"validation_{i+1}.txt"

    image.save(image_path)
    text_path.write_text(text, encoding="utf-8")

    print(f"Saved: {image_path}")