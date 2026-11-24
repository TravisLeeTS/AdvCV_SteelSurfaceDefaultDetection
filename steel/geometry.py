import cv2
import numpy as np


def letterbox(gray, side):
    h, w = gray.shape
    gain = min(side / w, side / h)
    width, height = round(w * gain), round(h * gain)
    left, top = (side - width) // 2, (side - height) // 2
    image = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, top, side - height - top, left, side - width - left,
                               cv2.BORDER_CONSTANT, value=114)
    return image, np.array([width / w, height / h, left, top], np.float32)


def project(boxes, transform, inverse=False):
    sx, sy, x, y = transform
    gain, pad = np.array([sx, sy, sx, sy]), np.array([x, y, x, y])
    return (boxes - pad) / gain if inverse else boxes * gain + pad


def to_yolo(boxes, w, h):
    left, right = boxes[:, :2], boxes[:, 2:]
    return np.concatenate(((left + right) / 2, right - left), 1) / np.array([w, h, w, h], np.float32)


def from_yolo(boxes, w, h):
    xywh = boxes * np.array([w, h, w, h], np.float32)
    center, half = xywh[:, :2], xywh[:, 2:] / 2
    return np.concatenate((center - half, center + half), 1)


def prepare(gray, side):
    image, transform = letterbox(gray, side)
    tensor = np.ascontiguousarray(np.repeat(image[None, None], 3, axis=1), dtype=np.float32)
    tensor *= np.float32(1 / 255)
    return tensor, transform


def augment(gray, boxes, classes, seed, epoch, index, side=480):
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, index]))
    scale = float(rng.uniform(0.8, 1.2))
    dx, dy = rng.uniform(-0.10, 0.10, 2) * side
    flip = bool(rng.random() < 0.5)
    contrast, brightness = rng.uniform(0.85, 1.15), rng.uniform(-0.15, 0.15) * 255
    canvas, transform = letterbox(gray, side)
    boxes = project(boxes, transform)
    x, y = (1 - scale) * side / 2 + dx, (1 - scale) * side / 2 + dy
    # Boxes use pixel edges. OpenCV samples pixel centers at integer coordinates.
    matrix = np.float32([[scale, 0, x + (scale - 1) / 2], [0, scale, y + (scale - 1) / 2]])
    canvas = cv2.warpAffine(canvas, matrix, (side, side), flags=cv2.INTER_LINEAR, borderValue=114)
    boxes = np.clip(project(boxes, np.float32([scale, scale, x, y])), 0, side)
    if flip:
        canvas = cv2.flip(canvas, 1)
        boxes[:, [0, 2]] = side - boxes[:, [2, 0]]
    lut = np.clip((np.arange(256) - 127.5) * contrast + 127.5 + brightness, 0, 255).round().astype(np.uint8)
    canvas = cv2.LUT(canvas, lut)
    keep = ((boxes[:, 2:] - boxes[:, :2]) >= 1).all(axis=1)
    rejected = np.bincount(classes[~keep], minlength=6)
    return canvas, boxes[keep], classes[keep], rejected, dict(scale=scale, dx=float(dx), dy=float(dy),
                                                             flip=flip, contrast=float(contrast), brightness=float(brightness))
