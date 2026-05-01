from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class VisDroneDataset(Dataset):
    """VisDrone dataset.

    Returns uint8 images (CHW tensor) and target dicts. The dataset does no
    resizing or normalization on its own — pass an albumentations ``transforms``
    pipeline (e.g. ``A.Resize``, ``A.Normalize``, ``ToTensorV2``) to control
    input size, dtype, and value range.
    """

    CLASSES = [
        "ignored-regions",
        "pedestrian",
        "people",
        "bicycle",
        "car",
        "van",
        "truck",
        "tricycle",
        "awning-tricycle",
        "bus",
        "motor",
        "others",
    ]

    RELABEL_MAP = {
        "ignored-regions": "ignored-regions",
        "pedestrian": "person",
        "people": "person",
        "bicycle": "bicycle",
        "car": "car",
        "van": "van",
        "truck": "truck",
        "tricycle": "tricycle",
        "awning-tricycle": "tricycle",
        "bus": "bus",
        "motor": "motorcycle",
        "others": "ignored-regions",
    }

    def __init__(
        self,
        image_dir: str,
        annotation_dir: str,
        transforms: Callable | None = None,
        filter_ignored: bool = True,
        filter_crowd: bool = True,
        relabel_classes: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.transforms = transforms
        self.filter_ignored = filter_ignored
        self.filter_crowd = filter_crowd
        self.relabel_classes = relabel_classes

        if relabel_classes:
            self.classes = list(dict.fromkeys(self.RELABEL_MAP.values()))
            name_to_id = {name: i for i, name in enumerate(self.classes)}
            self._label_map = {
                i: name_to_id[self.RELABEL_MAP[name]] for i, name in enumerate(self.CLASSES)
            }
        else:
            self.classes = list(self.CLASSES)
            self._label_map = None

        if not self.image_dir.exists():
            raise ValueError(f"Image directory does not exist: {self.image_dir}")

        self.image_files = sorted(
            [f for f in self.image_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        )

        if not self.image_files:
            raise ValueError(f"No images found in {self.image_dir}")

        print(f"Found {len(self.image_files)} images in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def get_image_path(self, idx: int) -> Path:
        """Get the image file path at the given index."""
        return self.image_files[idx]

    def get_class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.classes):
            return self.classes[class_id]
        return "unknown"

    def get_num_classes(self) -> int:
        return len(self.classes)

    def _parse_annotation(self, annotation_path: Path) -> tuple[np.ndarray, np.ndarray]:
        if not annotation_path.exists():
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        boxes: list[list[float]] = []
        labels: list[int] = []

        with open(annotation_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue

                bbox_left, bbox_top, bbox_width, bbox_height = map(int, parts[:4])
                score, category = int(parts[4]), int(parts[5])

                if self.filter_ignored and score == 0:
                    continue
                if bbox_width <= 0 or bbox_height <= 0:
                    continue

                if self._label_map is not None:
                    category = self._label_map[category]

                if self.filter_crowd and category == 0:
                    continue

                x1, y1 = bbox_left, bbox_top
                x2, y2 = bbox_left + bbox_width, bbox_top + bbox_height
                boxes.append([x1, y1, x2, y2])
                labels.append(category)

        if not boxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor]]:
        img_path = self.image_files[idx]
        image_np = np.array(Image.open(img_path).convert("RGB"))

        ann_path = self.annotation_dir / (img_path.stem + ".txt")
        boxes_np, labels_np = self._parse_annotation(ann_path)

        if len(boxes_np) == 0:
            boxes_np = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
            labels_np = np.array([1], dtype=np.int64)

        if self.transforms is not None:
            transformed = self.transforms(image=image_np, bboxes=boxes_np, labels=labels_np)
            image_np = transformed["image"]
            boxes_np = np.asarray(transformed["bboxes"], dtype=np.float32).reshape(-1, 4)
            labels_np = np.asarray(transformed["labels"], dtype=np.int64).reshape(-1)
            if len(boxes_np) == 0:
                boxes_np = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
                labels_np = np.array([1], dtype=np.int64)

        # If transforms produced a tensor (e.g. ToTensorV2), pass through.
        # Else convert HWC numpy → CHW tensor without value scaling.
        if isinstance(image_np, np.ndarray):
            image: Tensor = torch.from_numpy(np.ascontiguousarray(image_np)).permute(2, 0, 1)
        else:
            image = image_np

        boxes = torch.as_tensor(boxes_np, dtype=torch.float32)
        labels = torch.as_tensor(labels_np, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target: dict[str, Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }

        return image, target


class VisDroneDatasetCOCO(Dataset):
    def __init__(
        self,
        image_dir: str,
        coco_json: str,
        transforms: Callable | None = None,
    ) -> None:
        from pycocotools.coco import COCO

        self.image_dir = Path(image_dir)
        self.coco = COCO(coco_json)
        self.transforms = transforms
        self.ids = sorted(self.coco.imgs.keys())

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor]]:
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.image_dir / img_info["file_name"]
        image: ImageLike = Image.open(img_path).convert("RGB")

        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

        target: dict[str, Tensor] = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        assert isinstance(image, Tensor)
        return image, target
