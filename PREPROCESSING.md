# Prepare the data

Keep the raw dataset folders beside these scripts. Python 3.10+ is required.
Install the only extra dependency once:

```powershell
python -m pip install -r requirements.txt
```

## Classification now

```powershell
python -m src.data.preprocess
```

This short script applies the image findings from EDA:

- Read only images listed in `CUB_200_2011/images.txt`, excluding the eight extras.
- Exclude image 3340, the training duplicate already confirmed in EDA.
- Convert all inputs to RGB so the eight grayscale images have three channels.
- Resize directly to 224 x 224. This is the initial baseline and changes proportions.
- Preserve the official test split. Choose six validation images per species
  from training data with seed 42.

The image and split metadata must have the same image-ID order; the script
checks each pair. It applies this specific CUB dataset's known duplicate finding.

```text
classification_data/
    train/<species>/<image>.jpg
    val/<species>/<image>.jpg
    test/<species>/<image>.jpg
    complete.txt
```

Expected counts: 4,793 training, 1,200 validation, and 5,794 test images.
The species folders provide the labels. Training code can use:

```python
from torchvision import datasets, transforms

train_data = datasets.ImageFolder(
    "classification_data/train", transform=transforms.ToTensor()
)
```

Add random augmentation to the training loader only. Tensor conversion and any
normalization required by your chosen model happen in the training code.

## Caption problems recorded in EDA

Run the separate caption script when cleaned captions are needed:

```powershell
python -m src.data.clean_captions
```

It uses the images selected by `src/data/preprocess.py`, so captions stay with the same
train/validation/test split and the removed image duplicate is excluded too.
It reads UTF-8, repairs the 55 captions with the known doubled-character separator
pattern, omits the one caption damaged inside a word and the two captions
containing only `this`, lowercases text, normalizes whitespace, and removes
repeated captions within each image. Punctuation and spelling are preserved.
It does not guess spelling corrections for bird terminology.

Cleaned captions go to `classification_data/captions/<split>/<species>/<image>.txt`.
The counts in `classification_data/caption_cleaning.txt` cover retained images
only. Expected output: 116,562 captions across 11,787 caption files.

The EDA found no unreadable images and no severe imbalance, so neither requires
removing images or balancing the dataset. A new unreadable image stops preparation
instead of silently entering the clean dataset. Build a caption vocabulary from
training captions only when you choose the captioning model.

## Rerunning

Original files stay unchanged. Completed classification output is reused;
change `output` to a new folder when changing settings or raw inputs. A failed
image run leaves no `complete.txt`; choose a new output folder to retry.
If using another classification folder, update `dataset` in the caption script
as well. Caption cleaning can be rerun and overwrites only its generated text
files. Earlier `prepared_data` outputs remain separate.
