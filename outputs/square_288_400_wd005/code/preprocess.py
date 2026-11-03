# Classification preparation based on EDA. Run: python preprocess.py
from pathlib import Path
import random
from PIL import Image

project = Path(__file__).resolve().parent
source = project / "CUB_200_2011"
##########################################################################
# SQUARE CROP: new output folder, so the older images stay as they are
output = project / "classification_data_square_288"
image_size = 288
##########################################################################
validation_per_species = 6
seed = 42

# Keep completed output. Use a new output folder to rebuild it.
if output.exists():
    if not (output / "complete.txt").is_file():
        raise ValueError("Incomplete output. Choose a new output folder.")
    print("Dataset already prepared:", output)
    raise SystemExit

# Read the official image list and split assignments.
with open(source / "images.txt") as file:
    image_lines = file.readlines()
with open(source / "train_test_split.txt") as file:
    split_lines = file.readlines()
if len(image_lines) != len(split_lines):
    raise ValueError("The metadata files have different numbers of images.")

##########################################################################
# BIRD CROP: read the box around the bird for each image
with open(source / "bounding_boxes.txt") as file:
    box_lines = file.readlines()
box_of = {}
for image_line, box_line in zip(image_lines, box_lines):
    box_of[image_line.split()[1]] = [float(n) for n in box_line.split()[1:]]
##########################################################################

training = {}
files = {"train": [], "val": [], "test": []}

# Pair each image with its official split. Unlisted images are never included.
for image_line, split_line in zip(image_lines, split_lines):
    image_id, filename = image_line.split()
    split_id, flag = split_line.split()
    if image_id != split_id:
        raise ValueError("Image IDs are in different orders.")
    if image_id == "3340":  # The training duplicate confirmed in EDA.
        if flag != "1":
            raise ValueError("The known duplicate should be in training.")
        continue
    if flag == "0":
        files["test"].append(filename)
    elif flag == "1":
        species = filename.split("/")[0]
        if species not in training:
            training[species] = []
        training[species].append(filename)
    else:
        raise ValueError("Split flags must be 0 or 1.")

# Take six validation images from each training species.
random.seed(seed)
for species in sorted(training):
    filenames = training[species]
    if validation_per_species < 1 or len(filenames) <= validation_per_species:
        raise ValueError("Each species must have training images left over.")
    random.shuffle(filenames)
    files["val"].extend(filenames[:validation_per_species])
    files["train"].extend(filenames[validation_per_species:])

# Save RGB images at one size. Species folders provide the labels.
for split, filenames in files.items():
    for filename in filenames:
        destination = output / split / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source / "images" / filename) as image:
            image = image.convert("RGB")
            ##########################################################################
            # SQUARE CROP: real pixels only, so no black bars and no squeezing
            x, y, w, h = box_of[filename]
            # 1.3x the bird's longer side, but never bigger than the photo's short side
            side = min(1.3 * max(w, h), image.width, image.height)
            # Centre the square on the bird, then slide it back inside the photo
            left = min(max(0, x + w / 2 - side / 2), image.width - side)
            top = min(max(0, y + h / 2 - side / 2), image.height - side)
            image = image.crop((left, top, left + side, top + side))
            ##########################################################################
            image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
            image.save(destination, quality=95)
    print(split, "images:", len(filenames), flush=True)

(output / "complete.txt").write_text("Preprocessing finished.\n", encoding="utf-8")
print("Ready for classification:", output)
