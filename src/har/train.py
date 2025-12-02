"""Train a ResNet classifier and save the best validation checkpoint."""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from har.data import ActivityDataset, image_transform
from har.manifest import read_manifest
from har.models import build_classifier
from har.runtime import get_device, seed_everything


def run_epoch(model, loader, criterion, device, optimizer=None, freeze_backbone=False):
    training = optimizer is not None
    model.train(training)
    if training and freeze_backbone:
        model.eval()
        model.fc.train()
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for inputs, labels in tqdm(loader, desc="train" if training else "val", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            count += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / count, correct / count


def train(args):
    seed_everything(args.seed)
    device = get_device(args.device)
    train_samples, val_samples = read_manifest(args.train), read_manifest(args.val)
    train_groups = {sample.group for sample in train_samples}
    val_groups = {sample.group for sample in val_samples}
    train_paths = {sample.path for sample in train_samples}
    if train_groups & val_groups or train_paths & {sample.path for sample in val_samples}:
        raise ValueError("Training and validation sources overlap; split by source first")
    classes = sorted({sample.label for sample in train_samples})
    if {sample.label for sample in val_samples} != set(classes):
        raise ValueError("Training and validation must contain the same activity classes")
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {}
    for phase, samples in (("train", train_samples), ("val", val_samples)):
        dataset = ActivityDataset(
            samples, classes, image_transform(args.image_size, phase == "train")
        )
        loaders[phase] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=phase == "train",
            num_workers=args.workers,
            generator=generator,
            pin_memory=device.type == "cuda",
        )
    model = build_classifier(len(classes), args.architecture, pretrained=not args.scratch).to(
        device
    )
    if args.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.fc.parameters():
            parameter.requires_grad_(True)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    best_accuracy = -1.0
    with (args.output / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
            ],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = run_epoch(
                model, loaders["train"], criterion, device, optimizer, args.freeze_backbone
            )
            val_loss, val_accuracy = run_epoch(model, loaders["val"], criterion, device)
            writer.writerow(
                dict(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_accuracy=train_accuracy,
                    val_loss=val_loss,
                    val_accuracy=val_accuracy,
                )
            )
            handle.flush()
            print(
                f"Epoch {epoch}/{args.epochs}: train loss={train_loss:.4f} "
                f"acc={train_accuracy:.4f}; val loss={val_loss:.4f} acc={val_accuracy:.4f}"
            )
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(
                    {
                        "format_version": 1,
                        "architecture": args.architecture,
                        "classes": classes,
                        "image_size": args.image_size,
                        "state_dict": {
                            key: value.detach().cpu() for key, value in model.state_dict().items()
                        },
                        "epoch": epoch,
                        "val_accuracy": val_accuracy,
                        "seed": args.seed,
                        "train_groups": sorted(train_groups),
                        "val_groups": sorted(val_groups),
                    },
                    args.output / "best.pt",
                )
    best = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["state_dict"])
    model.eval()
    print(f"Best validation accuracy: {best_accuracy:.4f} (epoch {best['epoch']})")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", choices=["resnet18", "resnet50"], default="resnet50")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--scratch", action="store_true", help="Use random weights without a download"
    )
    parser.add_argument(
        "--freeze-backbone", action="store_true", help="Train only the final linear layer"
    )
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 2
        or args.lr <= 0
        or args.image_size < 32
        or args.workers < 0
    ):
        parser.error("Use epochs>=1, batch-size>=2, lr>0, image-size>=32, and workers>=0")
    train(args)


if __name__ == "__main__":
    main()
