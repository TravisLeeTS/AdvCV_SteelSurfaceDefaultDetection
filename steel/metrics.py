"""Common COCO-style 101-point AP, greedy per-class matching, and operating point."""
import numpy as np

from .common import CLASSES, SCORE_FLOOR
from .data import as_arrays

IOUS = np.linspace(0.50, 0.95, 10)
RECALL_GRID = np.linspace(0, 1, 101)


def iou(boxes, targets):
    lo = np.maximum(boxes[:, None, :2], targets[None, :, :2])
    hi = np.minimum(boxes[:, None, 2:], targets[None, :, 2:])
    intersection = np.maximum(hi-lo, 0).prod(2)
    area_a = np.maximum(boxes[:, 2:]-boxes[:, :2], 0).prod(1)
    area_b = np.maximum(targets[:, 2:]-targets[:, :2], 0).prod(1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-12)


def match_image(row, detections):
    boxes, labels = as_arrays(row)
    det = np.asarray(detections, np.float32).reshape(-1, 6)
    det = det[np.argsort(-det[:, 4], kind="stable")]
    flags = np.zeros((len(det), len(IOUS)), dtype=bool)
    for c in np.unique(labels):
        inds = np.flatnonzero(det[:, 5] == c)
        targets = boxes[labels == c]
        if not len(inds):
            continue
        overlaps = iou(det[inds, :4], targets)
        used = np.zeros((len(IOUS), len(targets)), bool)
        for k in np.flatnonzero(overlaps.max(1) >= 0.5):
            available = np.where(used, -1, overlaps[k][None, :])
            best = available.argmax(1)
            hits = available[np.arange(len(IOUS)), best] >= IOUS
            flags[inds[k], hits] = True
            used[hits, best[hits]] = True
    return det[:, 4], det[:, 5].astype(np.int32), flags, np.bincount(labels, minlength=6)


def events(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError("Missing image predictions")
    return [match_image(r, p) for r, p in zip(rows, predictions)]


def concatenate(items):
    scores, labels, flags, gt = zip(*items)
    scores = np.concatenate(scores)
    order = np.argsort(-scores, kind="stable")
    return scores[order], np.concatenate(labels)[order], np.concatenate(flags)[order], np.sum(gt, axis=0)


def choose_threshold(items):
    scores, labels, flags, gt = concatenate(items)
    if (gt == 0).any():
        raise ValueError("Calibration requires validation GT for each class")
    # Thresholds include all equal-score candidates. Include a legitimate no-detections point.
    empty = float(np.nextafter(np.float32(1), np.float32(2)))
    if not len(scores):
        return empty
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores)-1]
    hits = flags[:, 0].astype(np.int64)
    fp = np.r_[0, np.cumsum(1-hits)[ends]]
    recall = np.r_[0, np.cumsum(hits / gt[labels])[ends] / len(CLASSES)]
    levels = np.r_[empty, scores[ends].astype(np.float64)]
    ok = np.flatnonzero(10 * fp <= len(items))
    best = np.lexsort((-levels[ok], fp[ok], -np.round(recall[ok], 14)))[0]
    return float(levels[ok[best]])


def summarize(items, threshold=SCORE_FLOOR):
    scores, labels, flags, gt = concatenate(items)
    classes, aps = {}, []
    total_fp = 0
    for c, name in enumerate(CLASSES):
        mask = labels == c
        good = flags[mask]
        tp = good.cumsum(0)
        ap = np.zeros(len(IOUS))
        if len(tp) and gt[c]:
            recall = tp / gt[c]
            precision = np.maximum.accumulate((tp / np.arange(1, len(tp)+1)[:, None])[::-1], 0)[::-1]
            for k in range(len(IOUS)):
                ix = np.searchsorted(recall[:, k], RECALL_GRID, side="left")
                ap[k] = np.r_[precision[:, k], 0][ix].mean()
        if gt[c]:
            aps.append(ap)
        operating = good[scores[mask] >= threshold, 0]
        hits, fp = int(operating.sum()), int(len(operating) - operating.sum())
        total_fp += fp
        classes[name] = dict(ap50=float(ap[0]) if gt[c] else None, gt=int(gt[c]), tp=hits, fp=fp,
                             fn=int(gt[c])-hits, recall=hits/int(gt[c]) if gt[c] else None)
    return dict(map50=float(np.mean([a[0] for a in aps])), map5095=float(np.mean(aps)),
                fp_per_image=total_fp / len(items), threshold=float(threshold), images=len(items), classes=classes)
