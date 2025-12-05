"""Predict activity labels for one image or a directory of images."""

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image, ImageOps

from har.data import image_transform
from har.manifest import IMAGE_EXTENSIONS
from har.models import load_classifier
from har.runtime import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Prediction CSV")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    paths = (
        sorted(path for path in args.input.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        if args.input.is_dir()
        else [args.input]
    )
    if not paths or any(not path.is_file() for path in paths):
        parser.error("No input images found")
    if args.output.resolve() in {path.resolve() for path in paths}:
        parser.error("Prediction output must not overwrite an input image")
    device = get_device(args.device)
    model, checkpoint = load_classifier(args.checkpoint, device)
    transform = image_transform(checkpoint["image_size"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle, torch.inference_mode():
        writer = csv.writer(handle)
        writer.writerow(["path", "predicted", "confidence"])
        for path in paths:
            with Image.open(path) as image:
                inputs = transform(ImageOps.exif_transpose(image).convert("RGB"))
            probabilities = model(inputs.unsqueeze(0).to(device)).softmax(1)[0]
            confidence, prediction = probabilities.max(0)
            label = checkpoint["classes"][prediction.item()]
            writer.writerow([str(path), label, confidence.item()])
            print(f"{path.name}: {label} ({confidence.item():.3f})")


if __name__ == "__main__":
    main()
