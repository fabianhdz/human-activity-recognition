"""Save training curves, confusion matrices, and prediction examples."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def plot_history(history: Path, output: Path):
    with history.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No training history to plot")
    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("loss", "accuracy")):
        for phase in ("train", "val"):
            axis.plot(epochs, [float(row[f"{phase}_{metric}"]) for row in rows], label=phase)
        axis.set(xlabel="Epoch", ylabel=metric.capitalize())
        axis.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def plot_confusion_matrix(matrix, classes, output: Path):
    size = max(6, len(classes) * 0.65)
    figure, axis = plt.subplots(figsize=(size, size))
    chart = axis.imshow(matrix, cmap="Blues")
    ticks = list(range(len(classes)))
    axis.set(
        xticks=ticks,
        yticks=ticks,
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted",
        ylabel="Actual",
        title="Activity confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    for row in range(len(classes)):
        for col in range(len(classes)):
            axis.text(col, row, str(matrix[row][col]), ha="center", va="center")
    figure.colorbar(chart, ax=axis)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def plot_predictions(predictions: Path, output: Path, limit=12):
    with predictions.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No predictions to plot")
    rows = rows[:limit]
    columns = min(4, len(rows))
    figure, axes = plt.subplots(
        (len(rows) + columns - 1) // columns,
        columns,
        figsize=(columns * 3, ((len(rows) + columns - 1) // columns) * 3),
        squeeze=False,
    )
    for axis in axes.flat:
        axis.axis("off")
    for axis, row in zip(axes.flat, rows):
        with Image.open(row["path"]) as image:
            axis.imshow(image.convert("RGB"))
        title = f"{row['predicted']} ({float(row['confidence']):.2f})"
        if "actual" in row:
            title += f"\nActual: {row['actual']}"
        axis.set_title(title, fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--history", type=Path)
    source.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.history:
        plot_history(args.history, args.output)
    else:
        plot_predictions(args.predictions, args.output, args.limit)


if __name__ == "__main__":
    main()
