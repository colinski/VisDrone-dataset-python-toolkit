# VisDrone Toolkit 2.0

[![🐍 Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white&style=for-the-badge)](https://www.python.org/downloads/)
[![🔥 PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white&style=for-the-badge)](https://pytorch.org/)
[![⚖️ License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green?logo=open-source-initiative&logoColor=white&style=for-the-badge)](LICENSE)
[![🖤 Code style: Black](https://img.shields.io/badge/Code%20Style-Black-000000?logo=python&logoColor=white&style=for-the-badge)](https://github.com/psf/black)

Modern PyTorch toolkit for the VisDrone dataset with production-ready object detection models and real-time inference capabilities.

---

## What's New in 2.0

**Core Improvements:**

- PyTorch-first design with native Dataset implementation
- Multi-architecture support: Faster R-CNN, FCOS, RetinaNet (ResNet50 & MobileNet variants)
- Real-time webcam inference with pre-trained weights
- Modern format converters (COCO & YOLO, not just VOC)
- Production-ready CLI tools and comprehensive test suite

---

## Quick Start

```bash
# Install
git clone https://github.com/dronefreak/VisDrone-dataset-python-toolkit.git
cd VisDrone-dataset-python-toolkit
python3 -m venv venv && source venv/bin/activate
pip install -e .

# Test instantly with webcam (no training required)
python scripts/webcam_demo.py --model fasterrcnn_mobilenet

# Train on your data
python scripts/train.py \
    --train-img-dir data/VisDrone2019-DET-train/images \
    --train-ann-dir data/VisDrone2019-DET-train/annotations \
    --val-img-dir data/VisDrone2019-DET-val/images \
    --val-ann-dir data/VisDrone2019-DET-val/annotations \
    --model fasterrcnn_resnet50 \
    --epochs 50 --batch-size 4 --amp

# Run inference
python scripts/inference.py \
    --checkpoint outputs/my_model/best_model.pth \
    --model fasterrcnn_resnet50 \
    --input test_images/ --output-dir results
```

---

## Features

### Core Components

- **PyTorch Dataset** — Native VisDrone format with automatic filtering
- **Model Zoo** — 4 detection architectures ready for training
- **Format Converters** — COCO and YOLO export with validation
- **Visualization** — Publication-ready plots and detection overlays
- **CLI Tools** — Train, evaluate, and infer with simple commands

### Training Features

- Mixed precision training (AMP) for 2x speedup
- Multi-GPU support via DistributedDataParallel
- Learning rate scheduling and gradient clipping
- Automatic checkpointing and resumption
- Real-time training metrics visualization

### Inference Features

- Real-time webcam detection
- Batch processing for images and videos
- Configurable confidence thresholds
- FPS benchmarking and performance profiling

---

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended, 6GB+ VRAM)
- PyTorch 2.0+

### Setup

```bash
# 1. Virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. PyTorch (choose one)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu    # CPU

# 3. Install toolkit
pip install -e .              # Basic
pip install -e ".[dev]"       # With dev tools
pip install -e ".[coco]"      # With COCO eval
```

### Dataset Download

Download from [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset):

```bash
data/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── annotations/
└── VisDrone2019-DET-val/
    ├── images/
    └── annotations/
```

See [INSTALL.md](INSTALL.md) for detailed setup instructions.

---

## Usage

### Training

```bash
# Standard training
python scripts/train.py \
    --train-img-dir data/train/images \
    --train-ann-dir data/train/annotations \
    --val-img-dir data/val/images \
    --val-ann-dir data/val/annotations \
    --model fasterrcnn_resnet50 \
    --epochs 50 --batch-size 4 \
    --output-dir outputs/my_model

# Fast training with mixed precision
python scripts/train.py \
    --model fasterrcnn_mobilenet \
    --epochs 30 --batch-size 8 --amp \
    --output-dir outputs/mobilenet

# Resume from checkpoint
python scripts/train.py \
    --resume outputs/my_model/checkpoint_epoch_20.pth \
    --epochs 50
```

### Inference

```bash
# Single image
python scripts/inference.py \
    --checkpoint outputs/my_model/best_model.pth \
    --model fasterrcnn_resnet50 \
    --input image.jpg

# Batch images
python scripts/inference.py \
    --checkpoint outputs/my_model/best_model.pth \
    --model fasterrcnn_resnet50 \
    --input test_images/ --output-dir results

# Video processing
python scripts/inference.py \
    --checkpoint outputs/my_model/best_model.pth \
    --model fasterrcnn_resnet50 \
    --input video.mp4 --output-dir results
```

### Webcam Demo

```bash
# With trained model
python scripts/webcam_demo.py \
    --checkpoint outputs/my_model/best_model.pth \
    --model fasterrcnn_resnet50

# With pre-trained COCO weights (no training needed)
python scripts/webcam_demo.py --model fasterrcnn_mobilenet

# Custom settings
python scripts/webcam_demo.py \
    --model fasterrcnn_resnet50 \
    --camera 1 --score-threshold 0.7
```

**Controls:** `q` quit | `s` save frame | `SPACE` pause

### Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint outputs/my_model/best_model.pth \
    --model fasterrcnn_resnet50 \
    --image-dir data/val/images \
    --annotation-dir data/val/annotations \
    --output-dir eval_results
```

### Format Conversion

```bash
# To COCO
python scripts/convert_annotations.py \
    --format coco \
    --image-dir data/images \
    --annotation-dir data/annotations \
    --output annotations_coco.json

# To YOLO
python scripts/convert_annotations.py \
    --format yolo \
    --image-dir data/images \
    --annotation-dir data/annotations \
    --output-dir data/yolo_labels
```

### Python API

```python
from visdrone_toolkit import VisDroneDataset, get_model
from visdrone_toolkit.utils import collate_fn
from torch.utils.data import DataLoader

# Load dataset
dataset = VisDroneDataset(
    image_dir="data/images",
    annotation_dir="data/annotations",
    filter_ignored=True,
    filter_crowd=True,
)

# Get model
model = get_model("fasterrcnn_resnet50", num_classes=12, pretrained=True)

# Create dataloader
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

# Training loop
model.train()
for images, targets in loader:
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    # ... optimization
```

---

## Models

| Model                      | Speed | Accuracy | GPU Memory | Best For                    |
| -------------------------- | ----- | -------- | ---------- | --------------------------- |
| **Faster R-CNN ResNet50**  | ★★★☆☆ | ★★★★☆    | 6GB        | General use, best balance   |
| **Faster R-CNN MobileNet** | ★★★★★ | ★★★☆☆    | 3GB        | Real-time, edge devices     |
| **FCOS ResNet50**          | ★★★☆☆ | ★★★★☆    | 6GB        | Dense objects, anchor-free  |
| **RetinaNet ResNet50**     | ★★★☆☆ | ★★★★☆    | 6GB        | Class imbalance, focal loss |

### Performance Benchmarks

**VisDrone2019-DET-val** (RTX 3090, batch_size=4):

| Model                  | mAP@50 | FPS | Training Time (50 epochs) |
| ---------------------- | ------ | --- | ------------------------- |
| Faster R-CNN ResNet50  | ~35%   | 18  | ~8 hours                  |
| Faster R-CNN MobileNet | ~30%   | 45  | ~6 hours                  |
| FCOS ResNet50          | ~33%   | 16  | ~8 hours                  |
| RetinaNet ResNet50     | ~34%   | 17  | ~8 hours                  |

_Results depend on training configuration and dataset split_

---

## Advanced Usage

### Custom Augmentations

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

dataset = VisDroneDataset(
    image_dir="data/images",
    annotation_dir="data/annotations",
    transforms=transform,
)
```

### Multi-GPU Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

dist.init_process_group(backend='nccl')
model = DistributedDataParallel(model, device_ids=[local_rank])
```

### ONNX Export

```python
import torch

model.eval()
dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    model, dummy_input, "model.onnx",
    opset_version=11,
    input_names=['input'],
    output_names=['boxes', 'labels', 'scores']
)
```

---

## Documentation

- [Installation Guide](INSTALL.md) — Detailed setup
- [Quick Reference](QUICKSTART.md) — Command cheatsheet
- [Scripts Documentation](scripts/README.md) — CLI tools
- [Configuration Guide](configs/README.md) — Training configs
- [Test Documentation](tests/README.md) — Running tests
- [Contributing Guide](CONTRIBUTING.md) — Development workflow
- [Changelog](CHANGELOG.md) — Version history

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Guide

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/VisDrone-dataset-python-toolkit.git

# Setup dev environment
make setup-venv && source venv/bin/activate
make install-dev

# Make changes and test
make format lint test

# Submit PR
git checkout -b feature/your-feature
git commit -m "Add feature"
git push origin feature/your-feature
```

---

## Citation

If you use this toolkit, please cite:

```bibtex
@misc{visdrone_toolkit_2025,
  author = {Saksena, Saumya Kumaar},
  title = {VisDrone Toolkit 2.0: Modern PyTorch Implementation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/dronefreak/VisDrone-dataset-python-toolkit}
}
```

Original VisDrone dataset:

```bibtex
@article{zhu2018visdrone,
  title={Vision Meets Drones: A Challenge},
  author={Zhu, Pengfei and Wen, Longyin and Bian, Xiao and Ling, Haibin and Hu, Qinghua},
  journal={arXiv preprint arXiv:1804.07437},
  year={2018}
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE)

---

## Acknowledgments

- **VisDrone Team** for the dataset
- **PyTorch & Torchvision** for the framework
- All contributors to this project

---

## Roadmap

- [ ] VisDrone video task support
- [ ] Weights & Biases integration
- [ ] TensorRT optimization
- [ ] Docker deployment
- [ ] DETR and YOLOv8 architectures
- [ ] Mobile deployment guide

---

**Project Stats:** v2.0.0 | Python 3.8+ | PyTorch 2.0+ | 66 tests | >80% coverage

**Issues & Support:** [GitHub Issues](https://github.com/dronefreak/VisDrone-dataset-python-toolkit/issues)
