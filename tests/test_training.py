from argparse import Namespace

import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from har.evaluate import evaluate
from har.manifest import Sample, write_manifest
from har.models import build_classifier, load_classifier
from har.train import run_epoch, train


def test_validation_does_not_update_parameters():
    model = nn.Linear(3, 2)
    loader = DataLoader(TensorDataset(torch.ones(4, 3), torch.tensor([0, 1, 0, 1])), batch_size=2)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    loss, accuracy = run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    assert loss > 0
    assert 0 <= accuracy <= 1
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])


def test_best_checkpoint_is_saved_at_zero_accuracy(tmp_path, monkeypatch):
    for phase in ("train", "val"):
        samples = []
        for label in ("a", "b"):
            image = tmp_path / f"{phase}_{label}.jpg"
            Image.new("RGB", (32, 32)).save(image)
            samples.append(Sample(image, label, f"{phase}_{label}"))
        write_manifest(samples, tmp_path / f"{phase}.csv")
    tiny = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 2))
    monkeypatch.setattr("har.train.build_classifier", lambda *a, **k: tiny)
    calls = 0

    def epoch(model, *args, **kwargs):
        nonlocal calls
        calls += 1
        with torch.no_grad():
            model[1].bias.fill_(calls)
        return 1.0, 0.0

    monkeypatch.setattr("har.train.run_epoch", epoch)
    args = Namespace(
        train=tmp_path / "train.csv",
        val=tmp_path / "val.csv",
        seed=42,
        device="cpu",
        image_size=32,
        batch_size=2,
        workers=0,
        architecture="resnet18",
        scratch=True,
        freeze_backbone=False,
        lr=0.005,
        output=tmp_path / "run",
        epochs=2,
    )
    result = train(args)
    checkpoint = torch.load(args.output / "best.pt", weights_only=True)
    assert checkpoint["epoch"] == 1
    assert checkpoint["classes"] == ["a", "b"]
    torch.testing.assert_close(result[1].bias, torch.full((2,), 2.0))


def test_classifier_checkpoint_preserves_labels_and_rejects_leakage(tmp_path):
    model = build_classifier(2, "resnet18", pretrained=False)
    path = tmp_path / "model.pt"
    torch.save(
        {
            "format_version": 1,
            "architecture": "resnet18",
            "classes": ["cooking", "walking"],
            "image_size": 32,
            "state_dict": model.state_dict(),
            "train_groups": ["training_video"],
            "val_groups": ["validation_video"],
        },
        path,
    )
    restored, checkpoint = load_classifier(path, torch.device("cpu"))
    assert checkpoint["classes"] == ["cooking", "walking"]
    assert not restored.training
    image = tmp_path / "image.jpg"
    Image.new("RGB", (32, 32)).save(image)
    manifest = tmp_path / "test.csv"
    write_manifest([Sample(image, "cooking", "training_video")], manifest)
    with pytest.raises(ValueError, match="overlap"):
        evaluate(path, manifest, tmp_path / "metrics")
