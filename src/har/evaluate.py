"""Evaluate held-out activity images with per-class metrics and a confusion matrix."""

import argparse
import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from har.data import ActivityDataset, image_transform
from har.manifest import read_manifest
from har.models import load_classifier
from har.runtime import get_device


def evaluate(checkpoint_path, manifest, output, device="auto", batch_size=32):
    device = get_device(device)
    model, checkpoint = load_classifier(checkpoint_path, device)
    samples = read_manifest(manifest)
    known_groups = set(checkpoint["train_groups"]) | set(checkpoint["val_groups"])
    if known_groups & {sample.group for sample in samples}:
        raise ValueError("Evaluation sources overlap training or validation data")
    classes = checkpoint["classes"]
    dataset = ActivityDataset(samples, classes, image_transform(checkpoint["image_size"]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    truth, predictions, confidences = [], [], []
    with torch.inference_mode():
        for inputs, labels in tqdm(loader, desc="Evaluation"):
            probabilities = model(inputs.to(device)).softmax(1)
            confidence, prediction = probabilities.max(1)
            truth.extend(labels.tolist())
            predictions.extend(prediction.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
    indices = list(range(len(classes)))
    report = classification_report(
        truth, predictions, labels=indices, target_names=classes, output_dict=True, zero_division=0
    )
    accuracy = accuracy_score(truth, predictions)
    matrix = confusion_matrix(truth, predictions, labels=indices)
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": accuracy,
        "classes": classes,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "samples": len(samples),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "group", "actual", "predicted", "confidence"])
        for sample, prediction, confidence in zip(samples, predictions, confidences):
            writer.writerow(
                [str(sample.path), sample.group, sample.label, classes[prediction], confidence]
            )
    from har.visualize import plot_confusion_matrix

    plot_confusion_matrix(matrix, classes, output / "confusion_matrix.png")
    print(f"Accuracy: {accuracy:.4f}; weighted F1: {report['weighted avg']['f1-score']:.4f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    evaluate(args.checkpoint, args.manifest, args.output, args.device, args.batch_size)


if __name__ == "__main__":
    main()
