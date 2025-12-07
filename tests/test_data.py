from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from har.data import ActivityDataset, image_transform
from har.manifest import Sample, read_manifest, write_manifest
from har.prepare import prepare_images, sample_video
from har.split import split_samples


def test_video_groups_stay_in_one_split():
    samples = [
        Sample(Path(f"{label}/{video}/{frame}.jpg"), label, f"{label}_{video}")
        for label in ("cooking", "walking")
        for video in range(5)
        for frame in range(4)
    ]
    splits = split_samples(samples)
    assert splits == split_samples(samples)
    groups = {phase: {sample.group for sample in rows} for phase, rows in splits.items()}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
    assert sum(map(len, splits.values())) == len(samples)
    assert all(
        {sample.label for sample in rows} == {"cooking", "walking"} for rows in splits.values()
    )


def test_split_requires_independent_sources():
    samples = [Sample(Path(f"{i}.jpg"), "soccer", "same_video") for i in range(10)]
    with pytest.raises(ValueError, match="3 independent sources"):
        split_samples(samples)


def test_split_rejects_conflicting_video_labels():
    samples = [Sample(Path("a.jpg"), "cooking", "video"), Sample(Path("b.jpg"), "walking", "video")]
    with pytest.raises(ValueError, match="conflicting labels"):
        split_samples(samples)


def test_manifest_paths_and_labels_round_trip(tmp_path):
    image = tmp_path / "images" / "hello.jpg"
    image.parent.mkdir()
    Image.new("RGB", (24, 24)).save(image)
    sample = Sample(image, "cooking", "source")
    manifest = tmp_path / "splits" / "train.csv"
    write_manifest([sample], manifest)
    assert "../images/hello.jpg" in manifest.read_text()
    assert read_manifest(manifest) == [sample]
    with pytest.raises(ValueError, match="Duplicate image"):
        write_manifest([sample, sample], manifest)
        read_manifest(manifest)


def test_legacy_image_preparation_preserves_source(tmp_path):
    root = tmp_path / "original"
    root.mkdir()
    original = root / "playing_basketball_12.png"
    Image.new("RGBA", (30, 40), (20, 30, 40, 100)).save(original)
    (root / "broken_1.jpg").write_text("bad image")
    samples = prepare_images(root, tmp_path / "prepared", "filename", 64)
    assert original.exists()
    assert len(samples) == 1
    assert samples[0].label == "playing_basketball"
    with Image.open(samples[0].path) as image:
        assert image.mode == "RGB"
        assert image.size == (64, 64)


def test_class_order_comes_from_training_mapping(tmp_path):
    image = tmp_path / "image.jpg"
    Image.new("RGB", (40, 40)).save(image)
    samples = [Sample(image, "walking", "v1"), Sample(image, "cooking", "v2")]
    dataset = ActivityDataset(samples, ["cooking", "walking"], image_transform(32))
    assert dataset[0][1] == 1
    assert dataset[1][1] == 0
    assert dataset[0][0].shape == (3, 32, 32)
    with pytest.raises(ValueError, match="Unknown activity"):
        ActivityDataset(samples, ["cooking"], image_transform())


def test_video_sampling_uses_timestamps(tmp_path):
    video = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    assert writer.isOpened()
    try:
        for index in range(25):
            writer.write(np.full((32, 32, 3), index, dtype=np.uint8))
    finally:
        writer.release()
    frames = list(sample_video(video, tmp_path / "frames", 1.0, 100, 32))
    assert [path.stem for path in frames] == ["00000000", "00000010", "00000020"]


def test_invalid_video_fails(tmp_path):
    with pytest.raises(ValueError, match="Cannot read video"):
        list(sample_video(tmp_path / "missing.mp4", tmp_path / "frames", 1.0, 10, 32))
