"""Collect labeled YouTube videos from activity queries or a curated URL list."""

import argparse
import csv
import json
import re
from pathlib import Path

from yt_dlp import YoutubeDL


def collect(sources: list[tuple[str, str]], output: Path) -> None:
    for label, source in sources:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", label):
            raise ValueError(f"Use a simple directory name for the label: {label!r}")
        directory = output / label
        directory.mkdir(parents=True, exist_ok=True)
        options = {
            "format": "best[ext=mp4][height<=720]/best[height<=720]",
            "outtmpl": str(directory / "%(id)s.%(ext)s"),
            "writeinfojson": True,
            "noplaylist": True,
            "ignoreerrors": False,
            "overwrites": False,
        }
        with YoutubeDL(options) as downloader:
            downloader.extract_info(source, download=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--queries", type=Path, help="JSON mapping activity labels to search queries"
    )
    source.add_argument("--urls", type=Path, help="CSV with label,url columns")
    parser.add_argument("--limit", type=int, default=5, help="Search results per activity")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.queries:
        queries = json.loads(args.queries.read_text())
        sources = [(label, f"ytsearch{args.limit}:{query}") for label, query in queries.items()]
    else:
        with args.urls.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not {"label", "url"}.issubset(reader.fieldnames or []):
                parser.error("URL CSV must contain label,url columns")
            sources = [(row["label"], row["url"]) for row in reader]
    if not sources:
        parser.error("No sources provided")
    collect(sources, args.output)


if __name__ == "__main__":
    main()
