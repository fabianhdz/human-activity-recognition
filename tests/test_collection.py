import pytest

from har.collect import collect


def test_collection_keeps_video_ids_under_activity_labels(tmp_path, monkeypatch):
    calls = []

    class Downloader:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, source, download):
            calls.append((self.options, source, download))

    monkeypatch.setattr("har.collect.YoutubeDL", Downloader)
    collect([("cooking", "ytsearch5:person cooking")], tmp_path)
    options, source, download = calls[0]
    assert options["outtmpl"] == str(tmp_path / "cooking" / "%(id)s.%(ext)s")
    assert options["writeinfojson"]
    assert source == "ytsearch5:person cooking"
    assert download


def test_collection_rejects_invalid_class_paths(tmp_path):
    with pytest.raises(ValueError, match="directory name"):
        collect([("../other", "url")], tmp_path)
