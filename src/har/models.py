"""ResNet activity classifiers and self-contained inference checkpoints."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


def build_classifier(classes: int, architecture="resnet50", pretrained=True):
    if classes < 2:
        raise ValueError("At least two activity classes are required")
    if architecture == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    elif architecture == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    model.fc = nn.Linear(model.fc.in_features, classes)
    return model


def load_classifier(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Expected a checkpoint produced by har.train")
    model = build_classifier(
        len(checkpoint["classes"]), checkpoint["architecture"], pretrained=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint
