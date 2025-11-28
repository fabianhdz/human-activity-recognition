"""Six-stage OpenPose-style network with 38 PAF and 19 heatmap channels."""

import torch
from torch import nn


def make_backbone() -> nn.Sequential:
    layers = []
    channels = 3
    for width in (64, 64, "pool", 128, 128, "pool", 256, 256, 256, 256, "pool", 512, 512, 256, 128):
        if width == "pool":
            layers.append(nn.MaxPool2d(2, 2))
        else:
            layers.extend([nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(inplace=True)])
            channels = width
    return nn.Sequential(*layers)


def make_stage(output_channels: int, first: bool) -> nn.Sequential:
    specs = (
        [(128, 128, 3)] * 3 + [(128, 512, 1), (512, output_channels, 1)]
        if first
        else [(185, 128, 7)] + [(128, 128, 7)] * 4 + [(128, 128, 1), (128, output_channels, 1)]
    )
    layers = []
    for index, (input_channels, width, kernel) in enumerate(specs):
        layers.append(nn.Conv2d(input_channels, width, kernel, padding=kernel // 2))
        if index < len(specs) - 1:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class PoseEstimation(nn.Module):
    def __init__(self):
        super().__init__()
        # Keep layer names compatible with coco_pose_iter_440000.pth.tar.
        self.model0 = make_backbone()
        for stage in range(1, 7):
            setattr(self, f"model{stage}_1", make_stage(38, stage == 1))
            setattr(self, f"model{stage}_2", make_stage(19, stage == 1))

    def forward(self, images):
        features = self.model0(images)
        inputs = features
        for stage in range(1, 7):
            pafs = getattr(self, f"model{stage}_1")(inputs)
            heatmaps = getattr(self, f"model{stage}_2")(inputs)
            if stage < 6:
                inputs = torch.cat((pafs, heatmaps, features), dim=1)
        return pafs, heatmaps
