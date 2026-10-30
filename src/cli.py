# From the repository folder:
#   python -m src.cli train --task classification --config configs/cls_transfer.yaml
#   python -m src.cli predict --task classification --image path\to\bird.jpg --checkpoint outputs\classification\resnet18\best.pt
import argparse
from pathlib import Path


def load_config(path):
    try:
        import yaml
    except ImportError as error:
        raise SystemExit("Install PyYAML: pip install pyyaml") from error
    path = Path(path)
    if not path.is_file():
        raise SystemExit("Config not found: " + str(path))
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise SystemExit("Config must be a mapping: " + str(path))
    return config


def main():
    parser = argparse.ArgumentParser(description="Train or run the species classifier.")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    train.add_argument("--task", required=True)
    train.add_argument("--config", required=True)

    predict = commands.add_parser("predict")
    predict.add_argument("--task", required=True)
    predict.add_argument("--image", required=True)
    predict.add_argument("--checkpoint", required=True)

    args = parser.parse_args()
    if args.task != "classification":
        raise SystemExit("Only classification is implemented.")
    if args.command == "train":
        from src.training.trainer import train_model

        train_model(load_config(args.config))
        return
    from src.training.trainer import predict_image

    predict_image(args.image, args.checkpoint)


if __name__ == "__main__":
    main()
