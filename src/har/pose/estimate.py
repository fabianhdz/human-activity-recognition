"""Pose overlays and COCO-18 keypoints for images, manifests, and videos."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from har.manifest import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, Sample, read_manifest, write_manifest
from har.pose.decode import decode, draw_poses
from har.pose.model import PoseEstimation
from har.runtime import get_device


class PoseEstimator:
    def __init__(self, weights: Path, device="auto", scales=(0.5, 1.0, 1.5, 2.0), box_size=368):
        self.device = get_device(device)
        self.scales = scales
        self.box_size = box_size
        self.model = PoseEstimation()
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(
            {key.removeprefix("module."): value for key, value in state.items()}
        )
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image):
        height, width = image.shape[:2]
        paf_sum = np.zeros((height, width, 38), dtype=np.float32)
        heat_sum = np.zeros((height, width, 19), dtype=np.float32)
        for scale in self.scales:
            factor = scale * self.box_size / height
            resized = cv2.resize(
                image, (max(1, round(width * factor)), max(1, round(height * factor)))
            )
            h, w = resized.shape[:2]
            padded = cv2.copyMakeBorder(
                resized, 0, (-h) % 8, 0, (-w) % 8, cv2.BORDER_CONSTANT, value=(128, 128, 128)
            )
            tensor = torch.from_numpy(padded.astype(np.float32) / 256.0 - 0.5)
            pafs, heatmaps = self.model(tensor.permute(2, 0, 1).unsqueeze(0).to(self.device))
            for prediction, accumulator in ((pafs, paf_sum), (heatmaps, heat_sum)):
                array = prediction[0].permute(1, 2, 0).cpu().numpy()
                # Remove stride padding before mapping joints back to the original image.
                upsampled = cv2.resize(
                    array, (padded.shape[1], padded.shape[0]), interpolation=cv2.INTER_CUBIC
                )[:h, :w]
                accumulator += cv2.resize(upsampled, (width, height), interpolation=cv2.INTER_CUBIC)
        return decode(paf_sum / len(self.scales), heat_sum / len(self.scales))

    def image(self, source: Path, output: Path):
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"Cannot read image: {source}")
        people = self.predict(image)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), draw_poses(image, people)):
            raise OSError(f"Cannot write {output}")
        output.with_suffix(".json").write_text(
            json.dumps(
                {
                    "format": "coco18",
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "people": people.tolist(),
                },
                indent=2,
            )
            + "\n"
        )

    def video(self, source: Path, output: Path):
        if output.suffix.lower() != ".mp4":
            raise ValueError("Video output must use .mp4")
        output.parent.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(source))
        writer = None
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not capture.isOpened() or not fps > 0:
                raise ValueError(f"Cannot read video or frame rate: {source}")
            size = (
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not writer.isOpened():
                raise OSError(f"Cannot open video writer: {output}")
            with output.with_suffix(".jsonl").open("w") as handle:
                index = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    people = self.predict(frame)
                    writer.write(draw_poses(frame, people))
                    handle.write(
                        json.dumps({"frame": index, "time": index / fps, "people": people.tolist()})
                        + "\n"
                    )
                    index += 1
                if not index:
                    raise ValueError(f"No frames decoded: {source}")
        finally:
            capture.release()
            if writer is not None:
                writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scales", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--box-size", type=int, default=368)
    args = parser.parse_args()
    if any(scale <= 0 for scale in args.scales) or args.box_size < 8:
        parser.error("Scales must be positive and box-size must be at least 8")
    if args.manifest and not args.output_manifest:
        parser.error("--manifest requires --output-manifest")
    if args.input:
        source_path, output_path = args.input.resolve(), args.output.resolve()
        if source_path == output_path or (
            source_path.is_dir() and output_path.is_relative_to(source_path)
        ):
            parser.error("Use a separate output path")
    estimator = PoseEstimator(args.weights, args.device, args.scales, args.box_size)
    if args.manifest:
        samples = read_manifest(args.manifest)
        processed = []
        for sample in tqdm(samples, desc="Pose estimation"):
            relative = sample.path.name
            # Numeric output directories also support labels containing spaces or slashes.
            output = args.output / f"{len(processed):08d}" / relative
            if output.resolve() == sample.path.resolve():
                parser.error("Pose output would overwrite a source image")
            estimator.image(sample.path, output)
            processed.append(Sample(output, sample.label, sample.group))
        write_manifest(processed, args.output_manifest)
    elif args.input.is_dir():
        images = sorted(
            path for path in args.input.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            parser.error("No images found")
        for image in tqdm(images, desc="Pose estimation"):
            estimator.image(image, args.output / image.relative_to(args.input))
    elif args.input.suffix.lower() in VIDEO_EXTENSIONS:
        estimator.video(args.input, args.output)
    elif args.input.suffix.lower() in IMAGE_EXTENSIONS:
        estimator.image(args.input, args.output)
    else:
        parser.error("Expected an image, video, or image directory")


if __name__ == "__main__":
    main()
