import csv
import hashlib
import json
import os
import time
from pathlib import Path

CLASSES = ("crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches")
SIDES = (256, 320, 480, 640)
HEADS = ("one", "dense")
MODELS = ("A", "B", "C", "D", "E")
SCORE_FLOOR = 0.001
MAX_DET = 300
PRE_NMS_CAP = 30000
NMS_IOU = 0.70
SPLIT_SEED = 2026
WEIGHTS = {
    "A": ("yolo26n.pt", "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
          "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"),
    "E": ("yolov10n.pt", "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10n.pt",
          "11287ed0735678e7ba1ac2a9b3098c049155b3fde123992e724c1264bcc16b6f"),
}


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def replace_file(temporary, destination):
    """Commit completed writes atomically, tolerating transient Windows file handles."""
    for attempt in range(12):
        try:
            temporary.replace(destination)
            return
        except PermissionError as error:
            if getattr(error,"winerror",None) not in (5,32,33) or attempt==11:
                raise
            time.sleep(min(0.1*2**attempt,2))


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    replace_file(tmp,path)


def get(path):
    return Path(path).read_text(encoding="utf-8")


def read(path):
    return json.loads(get(path))


def table(path, rows, fields=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    replace_file(tmp,path)


def source_hash(*files):
    root = Path(__file__).parent
    return digest({name: sha(root / name) for name in files})


def fetch(path, url, expected):
    import requests
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".download")
        with requests.get(url, stream=True, timeout=(15, 120)) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        if sha(tmp) != expected:
            raise ValueError(f"Download checksum does not match the recorded release: {url}")
        tmp.replace(path)
    if sha(path) != expected:
        raise ValueError(f"Unexpected SHA-256 for {path}")
    return path


def matrix(models=MODELS, seeds=(0, 1, 2)):
    for model in models:
        for seed in seeds:
            for side in SIDES:
                yield model, seed, "openvino", "one", side
            yield model, seed, "openvino", "dense", 480
            if model in ("A", "E"):
                for head in HEADS:
                    yield model, seed, "pytorch", head, 480
