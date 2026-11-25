"""Pinned original YOLOv5s and Ultralytics YOLOv8n on the common dense interface."""
import copy
import hashlib
import os
from pathlib import Path
import sys
from zipfile import ZipFile

from filelock import FileLock
import torch
from torch import nn
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel, torch_safe_load
import yaml

from steel.common import CLASSES, fetch, sha


V5_REV = "915bbf294bb74c859f0b41f1c23bc395014ea679"
V5_ARCHIVE_SHA = "5aac5c8c40f89831fee611621b5ef3dbd7a7633278713b4dd0d059e1a631c7b0"
WEIGHTS = {
    "yolov8n": ("yolov8n.pt", "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
                "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"),
    "yolov5s": ("yolov5s-v7.0.pt", "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
                "8b3b748c1e592ddd8868022e8732fde20025197328490623cc16c6f24d0782ee"),
}


def v5_source(out):
    vendor = Path(out) / "vendor"
    vendor.mkdir(parents=True, exist_ok=True)
    folder = (vendor / ("yolov5-" + V5_REV)).resolve()
    with FileLock(vendor / "source.lock"):
        archive = fetch(vendor / "yolov5-v7.0.zip", f"https://codeload.github.com/ultralytics/yolov5/zip/{V5_REV}", V5_ARCHIVE_SHA)
        with ZipFile(archive) as z:
            for member in z.infolist():
                if member.is_dir():
                    continue
                target = (vendor / member.filename).resolve()
                if not target.is_relative_to(folder):
                    raise ValueError(f"Unexpected publisher source member {member.filename}")
                contents = z.read(member)
                if not target.is_file() or target.read_bytes() != contents:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(contents)
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    os.environ["YOLOv5_AUTOINSTALL"] = "false"
    return folder


def build(family, config, out):
    config = copy.deepcopy(config)
    if family == "yolov5s":
        v5_source(out)
        from models.yolo import DetectionModel as OriginalYOLOv5
        net = OriginalYOLOv5(config, ch=3, nc=len(CLASSES))
    elif family == "yolov8n":
        net = DetectionModel(config, ch=3, nc=len(CLASSES), verbose=False)
    else:
        raise ValueError(f"Unknown baseline {family}")
    net.names = dict(enumerate(CLASSES))
    return net


def initialize(family, seed, out, weights):
    filename, url, checksum = WEIGHTS[family]
    source_file = fetch(Path(weights) / filename, url, checksum)
    if family == "yolov5s":
        v5_source(out)
        # Original upstream releases serialize a module. Verify its release before loading.
        release = torch.load(source_file, map_location="cpu", weights_only=False)
    else:
        release, _ = torch_safe_load(source_file)
    source = (release.get("ema") or release["model"]).float()
    torch.manual_seed(seed)
    net = build(family, source.yaml, out)
    target = net.state_dict()
    tensors = {k:v for k,v in source.state_dict().items() if k in target and v.shape == target[k].shape}
    missing = net.load_state_dict(tensors, strict=False).missing_keys
    h = hashlib.sha256()
    for key, value in net.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return net, dict(url=url, pretrained_sha256=checksum, matching_tensors=len(tensors),
                     fresh_tensors=missing, initialization_sha256=h.hexdigest())


def criterion(net, family, out, epochs, batch):
    if family == "yolov8n":
        net.args = get_cfg(overrides={"epochs":epochs, "imgsz":480, "batch":batch})
        loss = net.init_criterion()
        def compute(batch_data):
            values, parts = loss(net(batch_data["img"]), batch_data)
            return values.sum(), parts
        return compute
    from utils.loss import ComputeLoss
    hyp = yaml.safe_load((v5_source(out) / "data/hyps/hyp.scratch-low.yaml").read_text(encoding="utf-8"))
    scale = 3 / net.model[-1].nl
    hyp["box"] *= scale
    hyp["cls"] *= len(CLASSES) / 80 * scale
    hyp["obj"] *= (480 / 640) ** 2 * scale
    net.hyp = hyp
    net.nc = len(CLASSES)
    loss = ComputeLoss(net)
    def compute(batch_data):
        labels = torch.cat((batch_data["batch_idx"][:, None], batch_data["cls"], batch_data["bboxes"]), 1)
        return loss(net(batch_data["img"]), labels)
    return compute


class DenseOutput(nn.Module):
    """Decoded centre-format boxes and per-class scores before external NMS."""

    def __init__(self, net, family):
        super().__init__()
        self.net = net
        self.anchor_based = family == "yolov5s"

    def forward(self, image):
        prediction = self.net(image)
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]
        if self.anchor_based:
            boxes = prediction[:, :, :4]
            scores = prediction[:, :, 4:5] * prediction[:, :, 5:]
            return torch.cat((boxes, scores), dim=2).transpose(1, 2)
        return prediction


def restore(run, out):
    state = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=True)
    net = build(state["family"], state["yaml"], out)
    net.load_state_dict(state["weights"], strict=True)
    return DenseOutput(net.float().eval(), state["family"]).eval()
