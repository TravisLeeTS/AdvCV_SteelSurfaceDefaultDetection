import numpy as np
import torch
from torchvision.ops import nms
from .common import MAX_DET, NMS_IOU, PRE_NMS_CAP
from .geometry import project


def candidates(output, branch, cutoff):
    if branch == "one":
        rows = output[0]
        return rows[rows[:, 4] >= cutoff], len(rows)
    anchors = output[0].T
    labels = anchors[:, 4:].argmax(1)
    scores = anchors[np.arange(len(anchors)), labels+4]
    ix = np.flatnonzero(scores >= cutoff)
    if len(ix) > PRE_NMS_CAP:
        ix = ix[np.argsort(-scores[ix], kind="stable")[:PRE_NMS_CAP]]
    centers, half = anchors[ix, :2], anchors[ix, 2:4] / 2
    return np.column_stack((centers-half, centers+half, scores[ix], labels[ix])).astype(np.float32), len(anchors)


def nms_inputs(rows):
    return (torch.from_numpy(np.ascontiguousarray(rows[:, :4])),
            torch.from_numpy(np.ascontiguousarray(rows[:, 4])),
            torch.from_numpy(rows[:, 5].astype(np.int64)))


def suppress(inputs):
    boxes, scores, classes = inputs
    if not len(scores):
        return classes.new_empty(0)
    kept = []
    for label in torch.unique(classes):
        indices = torch.where(classes == label)[0]
        kept.append(indices[nms(boxes[indices], scores[indices], NMS_IOU)])
    # Original anchor order breaks exact score ties across classes.
    indices = torch.cat(kept).sort().values
    return indices[torch.argsort(scores[indices], descending=True, stable=True)]


def finish(output, branch, cutoff, transform, shape):
    rows, total = candidates(output, branch, cutoff)
    count = len(rows)
    if branch == "dense":
        rows = rows[suppress(nms_inputs(rows)).numpy()]
    rows = rows[:MAX_DET].copy()
    h, w = shape
    rows[:, :4] = np.clip(project(rows[:, :4], transform, inverse=True), 0, [w, h, w, h])
    return rows, (total, count, len(rows))
