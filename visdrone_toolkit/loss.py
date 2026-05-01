import torch
import torch.nn.functional as F
from sam3.model.box_ops import box_cxcywh_to_xyxy, fast_diag_generalized_box_iou
from torch import Tensor
from torchvision.ops import sigmoid_focal_loss


def _flatten_indices(
    indices: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    batch_idx = torch.cat([
        torch.full_like(src, q) for q, (src, _) in enumerate(indices)
    ])
    src_idx = torch.cat([src for src, _ in indices])
    tgt_idx = torch.cat([tgt for _, tgt in indices])
    return batch_idx, src_idx, tgt_idx


class SetLoss:
    """DETR-style set loss for binary OV detection.

    Components:
      loss_class: sigmoid focal BCE over all (B*N, Q) query slots
      loss_bbox:  L1 on matched pairs
      loss_giou:  1 - GIoU on matched pairs

    All components normalized by the number of matched pairs in the batch.
    Predictions and targets are expected in cxcywh-normalized format.
    """

    def __init__(
        self,
        weight_class: float = 1.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def __call__(
        self,
        pred_logits: Tensor,
        pred_boxes: Tensor,
        target_boxes: list[Tensor],
        indices: list[tuple[Tensor, Tensor]],
    ) -> dict[str, Tensor]:
        device = pred_logits.device
        batch_idx, src_idx, _ = [
            i.to(device) for i in _flatten_indices(indices)
        ]
        num_matched = max(batch_idx.numel(), 1)

        target_classes = torch.zeros_like(pred_logits)
        target_classes[batch_idx, src_idx, 0] = 1.0
        loss_class = sigmoid_focal_loss(
            pred_logits,
            target_classes,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="sum",
        ) / num_matched

        if batch_idx.numel() == 0:
            zero = pred_boxes.sum() * 0.0
            loss_bbox = zero
            loss_giou = zero
        else:
            matched_pred = pred_boxes[batch_idx, src_idx]
            matched_gt = torch.cat([
                target_boxes[q][tgt.to(device)]
                for q, (_, tgt) in enumerate(indices) if tgt.numel() > 0
            ])
            loss_bbox = F.l1_loss(matched_pred, matched_gt, reduction="sum") / num_matched
            loss_giou = (
                1 - fast_diag_generalized_box_iou(
                    box_cxcywh_to_xyxy(matched_pred),
                    box_cxcywh_to_xyxy(matched_gt),
                )
            ).sum() / num_matched

        return {
            "loss_class": self.weight_class * loss_class,
            "loss_bbox": self.weight_bbox * loss_bbox,
            "loss_giou": self.weight_giou * loss_giou,
        }
