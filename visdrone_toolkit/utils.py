"""
Utility functions for VisDrone toolkit.

Includes model factory, collate functions, and other helper utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    FCOS_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    fcos_resnet50_fpn,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

# VisDrone class names
VISDRONE_CLASSES = [
    "ignored-regions",  # 0
    "pedestrian",  # 1
    "people",  # 2
    "bicycle",  # 3
    "car",  # 4
    "van",  # 5
    "truck",  # 6
    "tricycle",  # 7
    "awning-tricycle",  # 8
    "bus",  # 9
    "motor",  # 10
    "others",  # 11
]

# Number of classes (excluding background for torchvision models)
NUM_CLASSES = len(VISDRONE_CLASSES)


def get_model(
    model_name: str = "fasterrcnn_resnet50",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    trainable_backbone_layers: int | None = None,
    **kwargs,
) -> Any | torch.nn.Module:
    """
    Get a detection model for VisDrone.

    Args:
        model_name: One of ['fasterrcnn_resnet50', 'fasterrcnn_mobilenet',
                    'fcos_resnet50', 'retinanet_resnet50']
        num_classes: Number of classes (default: 12 for VisDrone)
        pretrained: Load pretrained weights (COCO)
        pretrained_backbone: Use pretrained backbone
        trainable_backbone_layers: Number of trainable backbone layers
        **kwargs: Additional model-specific arguments

    Returns:
        Detection model ready for training/inference
    """
    model_name = model_name.lower()

    if model_name == "fasterrcnn_resnet50":
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=None,
            trainable_backbone_layers=trainable_backbone_layers,
            **kwargs,
        )
        # Replace classifier head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    elif model_name == "fasterrcnn_mobilenet":
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights,
            weights_backbone=None,
            trainable_backbone_layers=trainable_backbone_layers,
            **kwargs,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    elif model_name == "fcos_resnet50":
        weights = FCOS_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = fcos_resnet50_fpn(
            weights=weights,
            weights_backbone=None,
            trainable_backbone_layers=trainable_backbone_layers,
            **kwargs,
        )
        # Replace classification head
        in_channels = model.head.classification_head.conv[0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = FCOSClassificationHead(
            in_channels, num_anchors, num_classes
        )

    elif model_name == "retinanet_resnet50":
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model = retinanet_resnet50_fpn_v2(
            weights=weights,
            weights_backbone=None,
            trainable_backbone_layers=trainable_backbone_layers,
            **kwargs,
        )
        # Replace classification head
        in_channels = model.head.classification_head.conv[0][0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels, num_anchors, num_classes
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from: fasterrcnn_resnet50, fasterrcnn_mobilenet, "
            f"fcos_resnet50, retinanet_resnet50"
        )

    return model


def collate_fn(batch: list) -> tuple:
    """
    Custom collate function for DataLoader.

    Handles variable number of objects per image.
    """
    return tuple(zip(*batch))


def ov_collate(batch: list) -> dict:
    """Collate VisDroneDataset items for OV detection.

    Builds a batch-wide prompt vocab as the union of each item's prompts in
    first-occurrence order, remaps per-image local label ids into that union,
    and slices GT boxes per (image, prompt) — the input format expected by
    the per-(image, prompt) Hungarian matcher.

    Boxes are converted to cxcywh-normalized using the stacked image size.

    Returns:
        images:        Tensor [B, 3, H, W]
        target_boxes:  list[Tensor] of length B*N, each [m, 4] cxcywh-normalized
                       (image-major: index b*N + p is the (image b, prompt p) entry)
        ignored_boxes: list[Tensor] of length B, each [k, 4] cxcywh-normalized
        prompts:       list[str] of length N (the batch-wide vocab)
    """
    images = torch.stack([item[0] for item in batch])
    H, W = images.shape[-2:]
    scale = torch.tensor([W, H, W, H], dtype=torch.float32)

    def to_cxcywh_norm(boxes_xyxy: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
        return torch.stack(
            [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1
        ) / scale

    prompts = list(dict.fromkeys(p for _, t in batch for p in t["prompts"]))
    name_to_id = {name: i for i, name in enumerate(prompts)}
    N = len(prompts)

    target_boxes: list[torch.Tensor] = []
    ignored_boxes: list[torch.Tensor] = []
    for _, t in batch:
        global_labels = torch.tensor(
            [name_to_id[t["prompts"][int(l)]] for l in t["labels"].tolist()],
            dtype=torch.int64,
        )
        boxes_cx = to_cxcywh_norm(t["boxes"])
        for p in range(N):
            target_boxes.append(boxes_cx[global_labels == p])
        ignored_boxes.append(to_cxcywh_norm(t["ignored_boxes"]))

    return {
        "images": images,
        "target_boxes": target_boxes,
        "ignored_boxes": ignored_boxes,
        "prompts": prompts,
    }


def compute_metrics(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute basic detection metrics (mAP would require pycocotools).

    Args:
        predictions: List of prediction dicts with 'boxes', 'labels', 'scores'
        targets: List of target dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for matching

    Returns:
        Dictionary of metrics
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, target in zip(predictions, targets):
        pred_boxes = pred["boxes"].cpu()
        pred_labels = pred["labels"].cpu()
        target_boxes = target["boxes"].cpu()
        target_labels = target["labels"].cpu()

        if len(pred_boxes) == 0 and len(target_boxes) == 0:
            continue
        elif len(pred_boxes) == 0:
            total_fn += len(target_boxes)
            continue
        elif len(target_boxes) == 0:
            total_fp += len(pred_boxes)
            continue

        # Compute IoU matrix
        ious = box_iou(pred_boxes, target_boxes)

        # Match predictions to targets
        matched_targets = set()
        for i in range(len(pred_boxes)):
            max_iou, max_idx = ious[i].max(dim=0)
            if max_iou >= iou_threshold and pred_labels[i] == target_labels[max_idx]:
                if max_idx.item() not in matched_targets:
                    total_tp += 1
                    matched_targets.add(max_idx.item())
                else:
                    total_fp += 1
            else:
                total_fp += 1

        total_fn += len(target_boxes) - len(matched_targets)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: (N, 4) tensor of [x1, y1, x2, y2]
        boxes2: (M, 4) tensor of [x1, y1, x2, y2]

    Returns:
        (N, M) tensor of IoU values
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou


def get_transform(train: bool = True):
    """
    Get basic transforms for training/validation.

    For more advanced augmentation, use albumentations.
    """
    import torchvision.transforms as T

    transforms = []
    if train:
        # Add training augmentations here
        # Note: torchvision transforms don't handle bboxes well
        # Consider using albumentations for serious augmentation
        pass

    # Convert PIL to tensor
    transforms.append(T.ToTensor())

    return T.Compose(transforms)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    filepath: str | Path,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    **kwargs,
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        **kwargs,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    device: str = "cuda",
) -> int:
    """
    Load a trusted training checkpoint.

    Security:
        This function loads model weights only (no arbitrary object deserialization).
        Safe against pickle-based code execution (Bandit B614 compliant).

    Returns:
        Starting epoch.
    """
    checkpoint = torch.load(
        filepath,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = int(checkpoint.get("epoch", 0))
    print(f"Checkpoint loaded from {filepath} (epoch {epoch})")

    return epoch
