# Human Activity Recognition

PyTorch pipeline for classifying human activities with ResNet-50 and inspecting human poses with an OpenPose-style COCO-18 network. Includes YouTube collection, frame preparation, training, evaluation, and visualization for activities such as sports, cooking, tutoring, and walking.

## Setup

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For a specific CPU or CUDA build, install the appropriate [PyTorch build](https://pytorch.org/get-started/locally/) in the virtual environment before installing the project.

Commands below run from the repository root. Data, model weights, and generated outputs are excluded from Git.

## Collect and prepare

Edit `configs/activities.json` to choose labels and search queries. Review the collected videos and labels before training.

```bash
python -m har.collect --queries configs/activities.json --limit 5 --output data/videos
python -m har.prepare --videos data/videos --output data/frames --manifest data/frames.csv
python -m har.split --manifest data/frames.csv --output data/splits
```

For curated videos, use `har.collect --urls sources.csv --output data/videos` with `label,url` columns. Downloads retain video IDs and metadata. Use videos you have permission to download; YouTube access requirements may require an updated yt-dlp and its documented runtime dependencies.

Video preparation samples one frame per second, up to 120 frames per video. Splits target 60/20/20 by source count within each class and keep every frame from a video together. Each class needs at least three independent sources. Frame counts can differ between splits.

Existing class folders can be prepared with `--images path/to/images`. For flat files named `activity_123.jpg`, also pass `--label-from filename`:

```bash
python -m har.prepare --images path/to/images --label-from filename \
  --output data/images --manifest data/images.csv
```

Source images are preserved. Image inputs receive one source group per file; if several images come from the same recording, set their `group` values to the recording ID before splitting. Manifest columns are `path,label,group`, with paths relative to the CSV file.

## Train and evaluate

```bash
python -m har.train --train data/splits/train.csv --val data/splits/val.csv --output runs/resnet50
python -m har.evaluate --checkpoint runs/resnet50/best.pt \
  --manifest data/splits/test.csv --output runs/resnet50/evaluation
python -m har.visualize --history runs/resnet50/history.csv --output runs/resnet50/curves.png
python -m har.visualize --predictions runs/resnet50/evaluation/predictions.csv \
  --output runs/resnet50/examples.png
python -m har.predict --checkpoint runs/resnet50/best.pt \
  --input path/to/image.jpg --output runs/predictions.csv
```

Training uses ImageNet initialization, SGD, and 15 epochs by default. Set `--architecture resnet18` for a smaller model, `--freeze-backbone` to train only the final layer, or `--scratch` to avoid downloading pretrained weights. `--device cpu` forces CPU execution. The best validation checkpoint includes the class order, image size, and source groups. Evaluation rejects overlap with training or validation sources.

Outputs include training loss and accuracy, per-class precision/recall/F1, a confusion matrix, and prediction CSVs. Predictions are image-based; the classifier does not model temporal motion across frames. Dataset-specific accuracy must be measured on held-out data.

## Pose estimation

Place a compatible six-stage COCO-18 PyTorch checkpoint at `checkpoints/coco_pose_iter_440000.pth.tar`. The loader accepts either a state dictionary or a checkpoint containing `state_dict`, including `module.` prefixes. Pose weights are supplied separately; Caffe and BODY_25 weights are incompatible.

```bash
python -m har.pose.estimate --weights checkpoints/coco_pose_iter_440000.pth.tar \
  --input path/to/image.jpg --output runs/pose.jpg
python -m har.pose.estimate --weights checkpoints/coco_pose_iter_440000.pth.tar \
  --input path/to/video.mp4 --output runs/pose.mp4
```

An image directory is also accepted. Image outputs include JSON keypoints; video outputs include one JSONL entry per frame. Each person has 18 `[x, y, confidence]` joints in COCO order, with zeros for missing joints. Multi-scale inference defaults to four scales; use `--scales 1.0` for a quicker pass.

To train on pose overlays, process a manifest and split the resulting manifest using the same seed:

```bash
python -m har.pose.estimate --weights checkpoints/coco_pose_iter_440000.pth.tar \
  --manifest data/frames.csv --output data/poses --output-manifest data/poses.csv
python -m har.split --manifest data/poses.csv --output data/pose-splits
```

Then use `data/pose-splits/train.csv` and `val.csv` with `har.train`. Class labels and source groups are preserved. Overlays retain the original image background.

## Code

- `collect.py`, `prepare.py`, `split.py`: collection and dataset preparation
- `pose/`: pose network, PAF decoding, image and video inference
- `data.py`, `models.py`, `train.py`: activity datasets and ResNet training
- `evaluate.py`, `predict.py`, `visualize.py`: metrics, predictions, and plots

```bash
pip install -e '.[dev]'
ruff check .
pytest
```

The pose network follows [OpenPose's part affinity field approach](https://arxiv.org/abs/1812.08008) and [COCO output conventions](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md). ResNet models use [torchvision](https://docs.pytorch.org/vision/stable/models.html); video collection uses [yt-dlp](https://github.com/yt-dlp/yt-dlp).
