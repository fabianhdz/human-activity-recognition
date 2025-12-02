"""Activity images with a shared class mapping and ImageNet preprocessing."""

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


def image_transform(image_size=224, training=False):
    spatial = (
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        if training
        else [transforms.Resize(round(image_size * 232 / 224)), transforms.CenterCrop(image_size)]
    )
    return transforms.Compose(
        spatial
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class ActivityDataset(Dataset):
    def __init__(self, samples, classes, transform):
        self.samples = samples
        self.classes = classes
        self.class_to_index = {label: index for index, label in enumerate(classes)}
        self.transform = transform
        unknown = {sample.label for sample in samples} - set(classes)
        if unknown:
            raise ValueError(f"Unknown activity classes: {sorted(unknown)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            tensor = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
        return tensor, self.class_to_index[sample.label]
