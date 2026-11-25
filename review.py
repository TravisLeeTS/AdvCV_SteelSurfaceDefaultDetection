"""Restore the published files, reproduce the study, or run its saved models."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
REPORT = ARTIFACTS / "study" / "report"
SCRATCH = ROOT / ".review"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def require_digest(path, expected):
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"Missing file or SHA-256 mismatch: {path}")


def write_verified(source, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temp:
        pending = Path(temp.name)
    try:
        with pending.open("wb") as output:
            shutil.copyfileobj(source, output)
        require_digest(pending, expected)
        pending.replace(target)
    finally:
        pending.unlink(missing_ok=True)


def restore_archive(archive_path, entry):
    prefix = (ROOT / entry["prefix"]).resolve()
    if not prefix.is_relative_to(ARTIFACTS):
        raise ValueError(f"Unexpected destination for {archive_path}")
    missing = {}
    for name, digest in entry["files"].items():
        target = (ROOT / name).resolve()
        if not target.is_relative_to(prefix):
            raise ValueError(f"Unexpected member path: {name}")
        if not target.is_file() or sha256(target) != digest:
            missing[name] = (target, digest)
    if not missing:
        return
    with tarfile.open(archive_path, "r|*") as archive:
        for member in archive:
            if not member.isfile() or member.name not in entry["files"]:
                raise ValueError(f"Unexpected archive entry: {member.name}")
            if member.name in missing:
                target, digest = missing.pop(member.name)
                with archive.extractfile(member) as content:
                    write_verified(content, target, digest)
    if missing:
        raise ValueError(f"Missing members in {archive_path}: {list(missing)}")


def unpack(index):
    extra = ROOT / "bundles/cpu-extension/index.json"
    indices = [index, read_json(extra)] if extra.is_file() else [index]
    for manifest in indices:
        for name, digest in manifest["files"].items():
            require_digest(ROOT / name, digest)
        for name, entry in manifest["archives"].items():
            archive = ROOT / name
            require_digest(archive, entry["sha256"])
            restore_archive(archive, entry)
    if extra.is_file():
        print(f"Verified and restored 21 checkpoints, {75 + indices[1]['graphs']} OpenVINO graphs and the CPU measurement archives.", flush=True)
    else:
        print("Verified and restored 15 checkpoints, 75 OpenVINO graphs and 174 measured cells.", flush=True)


def install_images(local_zip):
    split = read_json(DATA / "split.json")
    records = [row for name in ("train", "val", "test") for row in read_json(DATA / f"{name}.json")]
    for name, digest in split["split_files"].items():
        require_digest(DATA / f"{name}.json", digest)
    pending = {}
    for row in records:
        path = DATA / row["image"]
        if not path.is_file() or sha256(path) != row["image_sha"]:
            pending[path.name] = (path, row["image_sha"])
    if pending:
        archive = local_zip or DATA / "downloads" / "NEU-DET.zip"
        if local_zip is None and not archive.exists():
            print("Fetching the NEU-DET archive from the publisher.", flush=True)
            try:
                with urllib.request.urlopen(split["archive_url"], timeout=90) as stream:
                    write_verified(stream, archive, split["archive_sha"])
            except (OSError, ValueError) as exc:
                raise RuntimeError("Download NEU-DET from the publisher and use images --zip PATH.") from exc
        require_digest(archive, split["archive_sha"])
        with ZipFile(archive) as source:
            for member in source.infolist():
                name = PurePosixPath(member.filename).name
                if name in pending:
                    target, digest = pending.pop(name)
                    with source.open(member) as image:
                        write_verified(image, target, digest)
        if pending:
            raise ValueError(f"Images missing in NEU-DET archive: {list(pending)}")
    print(f"Verified {len(records)} original images against the frozen split manifests.")


def run_python(*arguments):
    subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, check=True)


def verify(index):
    unpack(index)
    SCRATCH.mkdir(exist_ok=True)
    run_python("-m", "unittest", "discover", "-s", "tests", "-q")
    with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
        tables = Path(temp)
        run_python("-m", "steel", "report", "--tables-only", "--output", tables)
        for name, recorded in index["primary_tables"].items():
            require_digest(tables / name, recorded["sha256"])
        print(f"Reproduced {len(index['primary_tables'])} primary tables byte for byte.", flush=True)
        run_python("-m", "steel", "verify", "--output", tables)
        checked = read_json(tables / "verification.json")
        if checked != read_json(REPORT / "verification.json"):
            raise ValueError("The replayed verification counts differ from the published results.")
        shutil.copyfile(tables / "verification.json", SCRATCH / "verification.json")
    run_python("-m", "scripts.verify_research_metrics", "--output", SCRATCH / "research-metrics-check.json")
    for name, fresh in (("research-metrics-check.json", SCRATCH / "research-metrics-check.json"),
                        ("factorial-check.json", ARTIFACTS / "audit" / "factorial-check.json")):
        if read_json(fresh) != read_json(ROOT / "validation" / name):
            raise ValueError(f"Independent evaluation differs from the archived {name}.")
    print("Verified the 174 quality cells, 522 timing sessions and independent metric and factorial checks.")
    if (ROOT / "bundles/cpu-extension/index.json").is_file():
        run_python("-m", "cpu_study.review", "verify")
    if (ARTIFACTS / "cpu-extension/policies/freeze.json").is_file():
        run_python("-m", "cpu_study.verify_policies")
        run_python("-m", "cpu_study.policy_timing", "verify")
        print("Verified the joint selection, six-class threshold fits and 108,000 supplementary timings.")
    if (ARTIFACTS / "cpu-extension/feedback/verification.json").is_file():
        run_python("-m", "cpu_study.feedback_audit", "--verify")
        run_python("-m", "cpu_study.dense_sizes", "verify")
        run_python("-m", "cpu_study.confirmation", "verify")
        print("Verified per-seed frame and class results, paired intervals, both timing rotations, "
              "36,000 dense-size calls and the frozen supervised policy.")


def replay(args, index):
    """Run a real forward pass and compare its final boxes with frozen detections."""
    import numpy as np
    import torch
    from steel.bench import Engine, name, read_predictions
    from steel.common import matrix
    from steel.data import gray_image
    from steel.export import paired_boxes
    from steel.geometry import prepare
    from steel.hardware import pin_cpu
    from steel.post import finish

    unpack(index)
    study = ARTIFACTS / "study"
    freeze = read_json(study / "freeze.json")
    records = {row["id"]: row for row in read_json(DATA / f"{args.split}.json")}
    if args.image not in records:
        raise ValueError(f"{args.image} is outside the frozen {args.split} split.")
    row = records[args.image]
    if not (DATA / row["image"]).is_file():
        raise FileNotFoundError("Run 'python review.py images' to install the full dataset.")
    require_digest(DATA / row["image"], row["image_sha"])
    gray = gray_image(DATA, row)
    if args.all:
        cells = list(matrix())
        # The archived native timing controls cover A and E. Also exercise every B/C/D checkpoint.
        cells.extend((model, seed, "pytorch", "one", 480) for model in "BCD" for seed in freeze["seeds"])
    else:
        cells = [(args.run[0], int(args.run[1]), args.runtime, args.head, args.side)]
    comparisons = []
    pin_cpu(4)
    with torch.inference_mode():
        for model, seed, runtime, head, side in cells:
            run = f"{model}{seed}"
            engine = Engine(ARTIFACTS / "runs" / run, runtime, head, side, 4)
            inputs, transform = prepare(gray, side)
            raw = engine.forward(engine.tensor(inputs))
            boxes, _ = finish(raw, head, freeze["cutoffs"][run], transform, gray.shape)
            cell = name((model, seed, runtime, head, side))
            source = study / args.split / cell / "predictions.jsonl.gz"
            if not source.is_file():
                source = study / args.split / name((model, seed, "openvino", head, side)) / "predictions.jsonl.gz"
            expected = dict(read_predictions(source, "operating"))[row["id"]]
            match = paired_boxes(np.asarray(boxes), expected)
            if match["missed"] or match["extra"]:
                raise ValueError(f"Box mismatch for {row['id']} / {cell} against {source.parent.name}: {match}")
            comparisons.append(dict(configuration=cell, checkpoint_sha=freeze["checkpoints"][run],
                                    reference=source.parent.name, boxes=np.asarray(boxes).tolist(), parity=match))
            if not args.all:
                print(f"{cell} / {row['id']}: {len(boxes)} operating boxes; reference {source.parent.name}.")
                print(json.dumps(np.asarray(boxes).tolist()))
    if args.all:
        SCRATCH.mkdir(exist_ok=True)
        receipt = SCRATCH / f"replay-{args.split}-{row['id']}.json"
        summary = dict(split=args.split, image=row["id"], image_sha=row["image_sha"], comparisons=comparisons)
        receipt.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Per-configuration detections and comparison results: {receipt}")
    print(f"Matched the archived operating boxes for {len(cells)} fresh inference configuration(s) on {row['id']}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("unpack", help="Check SHA-256 and restore saved graphs and raw measurements.")
    actions.add_parser("verify", help="Rebuild the tables, inspect the graphs and independently replay metrics.")
    images = actions.add_parser("images", help="Install all 1,800 original images with per-file checksum validation.")
    images.add_argument("--zip", type=Path, help="Use an existing official NEU-DET.zip.")
    play = actions.add_parser("replay", help="Run fresh CPU inference and compare with archived operating boxes.")
    play.add_argument("--all", action="store_true", help="Exercise all saved graphs and all native checkpoints.")
    play.add_argument("--run", choices=[f"{m}{s}" for m in "ABCDE" for s in range(3)], default="A0")
    play.add_argument("--runtime", choices=["openvino", "pytorch"], default="openvino")
    play.add_argument("--head", choices=["one", "dense"], default="one")
    play.add_argument("--side", type=int, choices=[256, 320, 480, 640], default=480)
    play.add_argument("--split", choices=["val", "test"], default="test")
    play.add_argument("--image", default="inclusion_11")
    args = parser.parse_args()
    if args.action == "replay" and not args.all and args.head == "dense" and args.side != 480:
        parser.error("The original shared-cutoff replay uses dense at 480. For the B dense 320 result use "
                     "python -m cpu_study.review replay --family B --head dense --side 320 --policy global.")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
    index = read_json(ROOT / "bundles" / "index.json")
    if args.action == "images":
        install_images(args.zip)
    elif args.action == "unpack":
        unpack(index)
    elif args.action == "verify":
        verify(index)
    elif args.action == "replay":
        replay(args, index)


if __name__ == "__main__":
    main()
