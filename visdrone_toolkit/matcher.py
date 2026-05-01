from __future__ import annotations

import torch
from sam3.model.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from scipy.optimize import linear_sum_assignment
from torch import Tensor


class HungarianMatcher:
    """Per-(image, prompt) bipartite matcher for binary OV detection.

    Each entry along the leading batch dimension is one independent matching
    problem: Q query slots competing for the GT boxes of a single (image, prompt)
    pair. Caller is responsible for flattening (image, prompt) into the batch
    axis before calling.

    Inputs use cxcywh-normalized boxes. ``pred_logits`` carries a single
    binary objectness logit per query.
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 1.0,
        cost_giou: float = 1.0,
    ) -> None:
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def __call__(
        self,
        pred_logits: Tensor,
        pred_boxes: Tensor,
        target_boxes: list[Tensor],
    ) -> list[tuple[Tensor, Tensor]]:
        B, Q, _ = pred_logits.shape
        sizes = [int(t.shape[0]) for t in target_boxes]
        empty = (torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64))

        if sum(sizes) == 0:
            return [empty] * B

        out_prob = pred_logits.sigmoid().flatten(0, 1).squeeze(-1)
        out_bbox = pred_boxes.flatten(0, 1)
        tgt_bbox = torch.cat(target_boxes)

        cost_class = -out_prob[:, None]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox),
        )
        cost = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        ).view(B, Q, -1).cpu()

        chunks = cost.split(sizes, dim=-1)
        indices = []
        for b in range(B):
            if sizes[b] == 0:
                indices.append(empty)
                continue
            src, tgt = linear_sum_assignment(chunks[b][b].numpy())
            indices.append((
                torch.as_tensor(src, dtype=torch.int64),
                torch.as_tensor(tgt, dtype=torch.int64),
            ))
        return indices
