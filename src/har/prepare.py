"""Sample video frames or prepare existing activity images without changing the sources."""

import argparse
import logging
from pathlib import Path

import cv2
from PIL import Image, ImageOps, UnidentifiedImageError

from har.manifest import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, Sample, write_manifest

LOGGER = logging.getLogger(__name__)


def sample_video(video: Path, output: Path, interval: float, max_frames: int, size: int):
    capture = cv2.VideoCapture(str(video))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not fps > 0:
            raise ValueError(f"Cannot read video or frame rate: {video}")
        output.mkdir(parents=True, exist_ok=True)
        index, saved, next_time = 0, 0, 0.0
        while saved < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if index / fps >= next_time:
                path = output / f"{index:08d}.jpg"
                frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
                if not cv2.imwrite(str(path), frame):
                    raise OSError(f"Cannot write {path}")
                yield path
                saved += 1
                next_time += interval
            index += 1
        if not saved:
            raise ValueError(f"No frames decoded: {video}")
    finally:
        capture.release()


def prepare_videos(root: Path, output: Path, interval: float, max_frames: int, size: int):
    samples = []
    for video in sorted(root.glob("*/*")):
        if video.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        label, group = video.parent.name, video.stem
        for path in sample_video(video, output / label / group, interval, max_frames, size):
            samples.append(Sample(path, label, group))
    return samples


def prepare_images(root: Path, output: Path, label_from: str, size: int):
    samples = []
    destinations = set()
    for image in sorted(root.rglob("*")):
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if label_from == "filename":
            label, separator, number = image.stem.rpartition("_")
            if not separator or not number.isdigit():
                raise ValueError(f"Expected activity_number.jpg: {image}")
        else:
            if image.parent == root:
                raise ValueError(f"Expected class subdirectories under {root}")
            label = image.relative_to(root).parts[0]
        relative = image.relative_to(root)
        destination = output / relative.with_suffix(".jpg")
        if destination in destinations:
            raise ValueError(f"Multiple source images map to {destination}")
        destinations.add(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(image) as original:
                resized = (
                    ImageOps.exif_transpose(original)
                    .convert("RGB")
                    .resize((size, size), Image.Resampling.LANCZOS)
                )
                resized.save(destination, quality=95)
        except (UnidentifiedImageError, OSError) as error:
            LOGGER.warning("Skipping %s: %s", image, error)
            continue
        samples.append(Sample(destination, label, relative.as_posix()))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--videos", type=Path, help="Root containing class/video files")
    source.add_argument("--images", type=Path, help="Root containing existing images")
    parser.add_argument("--label-from", choices=["parent", "filename"], default="parent")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between frames")
    parser.add_argument("--max-frames", type=int, default=120, help="Maximum frames per video")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    if args.interval <= 0 or args.max_frames < 1 or args.size < 1:
        parser.error("interval, max-frames, and size must be positive")
    root = (args.videos or args.images).resolve()
    if not root.is_dir():
        parser.error(f"Missing source directory: {root}")
    if args.output.resolve().is_relative_to(root) or root.is_relative_to(args.output.resolve()):
        parser.error("Source and output directories must not overlap")
    if args.videos:
        samples = prepare_videos(root, args.output, args.interval, args.max_frames, args.size)
    else:
        samples = prepare_images(root, args.output, args.label_from, args.size)
    if not samples:
        parser.error("No valid samples found")
    write_manifest(samples, args.manifest)
    print(f"Saved {len(samples)} samples to {args.manifest}")


if __name__ == "__main__":
    main()
