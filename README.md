# pixels-to-language
End-to-end multimodal deep learning: a shared visual encoder for species classification (with Grad-CAM), image colorization, and Transformer-based captioning on CUB-200-2011.

## Transfer-learning classification

Install the environment, put `CUB_200_2011` in the repository folder, then prepare the images:

```
pip install -r requirements.txt
python -m src.data.preprocess
```

Train a pretrained ResNet. The config chooses `resnet18`, `resnet34`, or `resnet50`.

```
python -m src.cli train --task classification --config configs/cls_transfer.yaml
```

Predict one image:

```
python -m src.cli predict --task classification --image path\to\bird.jpg --checkpoint outputs\classification\resnet18\best.pt
```

The first training run downloads ImageNet weights. Results are written to `outputs/classification/`.
