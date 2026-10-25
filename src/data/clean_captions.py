# Put cvpr2016_cub in the repository folder.
# After preprocessing, run: python -m src.data.clean_captions
from pathlib import Path

project = Path(__file__).resolve().parents[2]
source = project / "cvpr2016_cub" / "text_c10"
dataset = project / "classification_data"
if not (dataset / "complete.txt").is_file():
    raise ValueError("Run python -m src.data.preprocess successfully first.")

before = 0
after = 0
for split in ["train", "val", "test"]:
    image_folder = dataset / split
    for image in image_folder.rglob("*.jpg"):
        filename = image.relative_to(image_folder)
        filename = filename.with_suffix(".txt")
        with open(source / filename, encoding="utf-8") as file:
            lines = file.readlines()
        before = before + len(lines)
        cleaned = []
        for caption in lines:
            caption = caption.strip()
            if "�" in caption:
                if len(caption.split()) != 1:
                    continue  # Damage inside a word: do not invent a repair.
                caption = caption.replace("��", " ")  # Repair damaged spaces.
                if "�" in caption:
                    continue
            caption = caption.lower()
            caption = " ".join(caption.split())
            if caption == "" or caption == "this":
                continue
            if caption not in cleaned:
                cleaned.append(caption)
        if not cleaned:
            raise ValueError("No usable captions for " + str(filename))
        destination = dataset / "captions" / split / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
        after = after + len(cleaned)

summary = "Saved captions: " + str(after) + "\nRemoved captions: " + str(before - after) + "\n"
(dataset / "caption_cleaning.txt").write_text(summary, encoding="utf-8")
print(summary)
