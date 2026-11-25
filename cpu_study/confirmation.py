"""Verify the fixed supervised policy or score independently acquired, adjudicated strips."""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from steel.bench import Engine
from steel.common import CLASSES, MAX_DET, NMS_IOU, PRE_NMS_CAP, read, sha
from steel.hardware import pin_cpu
from steel.ir import checked_bundle
from steel.metrics import match_image
from .policy_timing import pipeline


SPEC = Path(__file__).with_suffix(".json")


def verify(artifacts):
    p = read(SPEC)
    target = artifacts / "cpu-extension/policies/freeze.json"
    entry = read(target)["fits"][p["graph"]]
    run = artifacts / "runs" / p["checkpoint"]
    _, graph = checked_bundle(run, p["head"], p["side"])
    if (sha(target) != p["policy_freeze_sha256"] or sha(run / "best.pt") != p["checkpoint_sha256"]
            or graph["bundle_sha"] != p["graph_sha256"] or entry["artifact_sha"] != p["graph_sha256"]
            or entry["checkpoint_sha"] != p["checkpoint_sha256"] or tuple(p["class_order"]) != CLASSES
            or entry["thresholds"][p["policy"]] != [p["all_class_cutoff"]] * len(CLASSES)
            or (NMS_IOU, PRE_NMS_CAP, MAX_DET) != (.70, 30000, 300)):
        raise ValueError("The prospective policy differs from the selected checkpoint, graph or thresholds.")
    return p


@torch.inference_mode()
def evaluate(data, artifacts, manifest):
    p = verify(artifacts)
    records = read(manifest)
    if (records.get("policy_sha256") != sha(SPEC) or not records.get("acquisition_source")
            or records.get("independently_acquired") is not True or not records.get("frames")):
        raise ValueError("Declare the acquisition source, frozen policy digest and independent exposures.")
    previous = {r["image_sha"] for split in ("train", "val", "test") for r in read(data / f"{split}.json")}
    used_ids, hashes = set(), set()
    strips, groups = defaultdict(list), set()
    line_times = []
    gt, tp = (np.zeros(len(CLASSES), dtype=int) for _ in range(2))
    pin_cpu(4)
    engine = Engine(artifacts / "runs" / p["checkpoint"], "openvino", p["head"], p["side"], 4)
    for row in records["frames"]:
        name, group, strip = (row[k] for k in ("id", "acquisition_group", "strip_id"))
        path = (manifest.parent / row["image"]).resolve()
        if not path.is_relative_to(manifest.parent.resolve()):
            raise ValueError("Independent images must be within the manifest directory.")
        h = sha(path)
        if not all((name, group, strip)) or name in used_ids or h in previous or h in hashes or h != row["image_sha"]:
            raise ValueError(f"Missing identity, duplicate exposure or reused benchmark image at {name}")
        hashes.add(h)
        used_ids.add(name)
        groups.add(group)
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None or type(row["clean_strip"]) is not bool or type(row["frame_index"]) is not int:
            raise ValueError(f"Invalid image, strip adjudication or sequence at {name}")
        labels = row["classes"]
        boxes = np.asarray(row["boxes"], dtype=float).reshape(-1, 4)
        if (len(labels) != len(boxes) or any(type(c) is not int or c not in range(len(CLASSES)) for c in labels)
                or not np.isfinite(boxes).all() or (boxes < 0).any() or (boxes[:, 2:] <= boxes[:, :2]).any()
                or (boxes[:, 2:] > [gray.shape[1], gray.shape[0]]).any() or (row["clean_strip"] and labels)):
            raise ValueError(f"Invalid adjudicated annotations at {name}")
        result = pipeline(engine, gray, p, [p["all_class_cutoff"]])
        _, predicted_classes, matches, targets = match_image(row, result)
        gt += targets
        tp += np.bincount(predicted_classes[matches[:, 0]], minlength=len(CLASSES))
        alarm = bool(len(result))
        strips[group, strip].append((row["frame_index"], row["clean_strip"], alarm))
        if any(k in row for k in ("capture_ns", "controller_ack_ns", "controller_alarm")):
            if (row.get("controller_alarm") is not alarm or type(row.get("capture_ns")) is not int
                    or type(row.get("controller_ack_ns")) is not int or row["controller_ack_ns"] <= row["capture_ns"]):
                raise ValueError(f"Incomplete timestamps or controller alarm differs from replay at {name}")
            line_times.append((row["controller_ack_ns"] - row["capture_ns"]) / 1e6)
    clean_alarms = []
    for identity, frames in strips.items():
        frames.sort()
        indices = [i for i, _, _ in frames]
        if len(set(indices)) != len(indices) or len({clean for _, clean, _ in frames}) != 1:
            raise ValueError(f"Duplicated frame index or inconsistent strip adjudication: {identity}")
        if frames[0][1]:
            if indices != list(range(p["clean_window_frames"])):
                raise ValueError(f"Supply the complete predeclared clean window (zero-based): {identity}")
            clean_alarms.append(any(alarm for _, _, alarm in frames))
    recall = {c: dict(matched=int(tp[i]), annotated=int(gt[i]), recall=float(tp[i]/gt[i]) if gt[i] else None)
              for i, c in enumerate(CLASSES)}
    rejection = float(np.mean(clean_alarms)) if clean_alarms else None
    line_p95 = float(np.percentile(line_times, 95)) if len(line_times) == len(used_ids) else None
    checks = {c: (recall[c]["recall"] >= floor if gt[CLASSES.index(c)] else None)
              for c, floor in p["critical_object_recall_floors"].items()}
    checks.update(clean_strips=rejection <= p["maximum_clean_strip_rejection_fraction"] if rejection is not None else None,
                  line_p95=line_p95 <= p["maximum_line_p95_ms"] if line_p95 is not None else None)
    return dict(policy_sha256=sha(SPEC), manifest_sha256=sha(manifest), acquisition_source=records["acquisition_source"],
                groups=sorted(groups), exposures=len(used_ids), object_recall=recall, clean_strips=len(clean_alarms),
                rejected_clean_strips=sum(clean_alarms), clean_rejection_fraction=rejection, timed_frames=len(line_times),
                capture_to_controller_p95_ms=line_p95, criteria_met=checks,
                all_criteria_met=None if None in checks.values() else all(checks.values()))


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["verify", "evaluate"])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.action == "verify":
        policy = verify(args.artifacts)
        print(f"Verified {policy['graph']} at {policy['all_class_cutoff']}. Prospective policy SHA-256: {sha(SPEC)}")
    else:
        if args.manifest is None:
            parser.error("evaluate requires --manifest")
        print(json.dumps(evaluate(args.data, args.artifacts, args.manifest), indent=2, allow_nan=False))
