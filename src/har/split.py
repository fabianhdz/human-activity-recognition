"""Create repeatable class-stratified splits while keeping each source video together."""

import argparse
import random
from collections import defaultdict
from pathlib import Path

from har.manifest import Sample, read_manifest, write_manifest


def split_samples(samples: list[Sample], seed: int = 42) -> dict[str, list[Sample]]:
    grouped = defaultdict(lambda: defaultdict(list))
    group_labels = {}
    for sample in samples:
        if sample.group in group_labels and group_labels[sample.group] != sample.label:
            raise ValueError(f"Source group has conflicting labels: {sample.group}")
        group_labels[sample.group] = sample.label
        grouped[sample.label][sample.group].append(sample)
    rng = random.Random(seed)
    splits = {name: [] for name in ("train", "val", "test")}
    for label, groups in sorted(grouped.items()):
        ids = sorted(groups)
        if len(ids) < 3:
            raise ValueError(f"{label!r} needs at least 3 independent sources; found {len(ids)}")
        rng.shuffle(ids)
        val_count = max(1, int(len(ids) * 0.2))
        test_count = max(1, int(len(ids) * 0.2))
        for index, group in enumerate(ids):
            phase = (
                "val"
                if index < val_count
                else "test"
                if index < val_count + test_count
                else "train"
            )
            splits[phase].extend(groups[group])
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for phase, samples in split_samples(read_manifest(args.manifest), args.seed).items():
        path = args.output / f"{phase}.csv"
        write_manifest(samples, path)
        print(f"{phase}: {len(samples)} samples")


if __name__ == "__main__":
    main()
