"""CSV manifests shared by collection, pose estimation, and classification."""

import csv
import os
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    group: str


def read_manifest(path: Path) -> list[Sample]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"path", "label", "group"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must have path,label,group columns")
        samples = []
        seen = set()
        for row in reader:
            image = Path(row["path"])
            image = (path.parent / image).resolve()
            if not row["label"] or not row["group"] or not image.is_file():
                raise ValueError(f"Invalid sample in {path}: {row}")
            if image in seen:
                raise ValueError(f"Duplicate image in {path}: {image}")
            seen.add(image)
            samples.append(Sample(image, row["label"], row["group"]))
    if not samples:
        raise ValueError(f"Empty manifest: {path}")
    return samples


def write_manifest(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "group"])
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "path": os.path.relpath(sample.path.resolve(), path.parent.resolve()),
                    "label": sample.label,
                    "group": sample.group,
                }
            )
