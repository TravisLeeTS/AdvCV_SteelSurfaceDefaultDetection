"""Verify two frozen detectors, enroll independent coils and evaluate a paired shadow study."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import torch
from filelock import FileLock

from steel.bench import Engine
from steel.common import CLASSES, MAX_DET, NMS_IOU, PRE_NMS_CAP, put, read, sha
from steel.figures import assignment
from steel.hardware import pin_cpu
from steel.ir import checked_bundle
from .confirmation_design import CRITICAL, MODELS, TIMING, enrollment, exact_screen, paired_binary, timing_block
from .policy_timing import pipeline


SPEC = Path(__file__).with_suffix(".json")


def verify(artifacts):
    p = read(SPEC)
    target = artifacts / "cpu-extension/policies/freeze.json"
    fits = read(target)["fits"]
    if (p["version"] != 2 or tuple(p["class_order"]) != CLASSES or set(p["models"]) != set(MODELS)
            or sha(target) != p["policy_freeze_sha256"] or (NMS_IOU, PRE_NMS_CAP, MAX_DET) != (.70, 30000, 300)):
        raise ValueError("Frozen protocol, class order or postprocessing differs from the benchmark.")
    for model, head in zip(MODELS, ("dense", "one")):
        arm = p["models"][model]
        run = artifacts / "runs" / model
        _, graph = checked_bundle(run, head, 480)
        entry = fits[arm["graph"]]
        if (arm["checkpoint"] != model or arm["graph"] != f"{model}-openvino-{head}-480"
                or (arm["head"], arm["side"], arm["policy"]) != (head, 480, "global")
                or not (sha(run/"best.pt") == arm["checkpoint_sha256"] == entry["checkpoint_sha"])
                or not (graph["bundle_sha"] == arm["graph_sha256"] == entry["artifact_sha"])
                or entry["thresholds"]["global"] != [arm["all_class_cutoff"]] * len(CLASSES)):
            raise ValueError(f"{model} differs from its preregistered checkpoint, graph or cutoff.")
    return p


def freeze_roster(policy, roster, output):
    plan = enrollment(policy, sha(SPEC), roster)
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock"):
        if output.exists() and read(output) != plan:
            raise ValueError("The enrollment file already contains a different design.")
        if not output.exists():
            put(output, plan)
        return sha(output)


def local_file(base, name):
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("Input files must be within the manifest directory.")
    return path


def validate_labels(row, gray, unit):
    labels, ids = row["classes"], row["object_ids"]
    boxes = np.asarray(row["boxes"], dtype=float)
    if (len(ids) != len(labels) or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids)
            or any(type(c) is not int or c not in range(len(CLASSES)) for c in labels)
            or (boxes.shape != (len(labels), 4) and not (boxes.shape == (0,) and not labels))):
        raise ValueError(f"Invalid object IDs, classes or box count at {row['id']}.")
    boxes = boxes.reshape(-1, 4)
    if (not np.isfinite(boxes).all() or (boxes < 0).any() or (boxes[:, 2:] <= boxes[:, :2]).any()
            or (boxes[:, 2:] > [gray.shape[1], gray.shape[0]]).any()):
        raise ValueError(f"Invalid pixel coordinates at {row['id']}.")
    if unit["arm"] == "clean":
        if labels:
            raise ValueError("An adjudicated clean strip contains defect labels.")
    else:
        target = unit["object_id"]
        if target not in ids or CLASSES[labels[ids.index(target)]] != unit["arm"]:
            raise ValueError(f"Frozen physical target or class differs at {row['id']}.")


def live_trace(row, alarms):
    if "capture_ns" not in row and "controllers" not in row:
        return None
    capture, records = row.get("capture_ns"), row.get("controllers", {})
    if type(capture) is not int or capture < 0 or not isinstance(records, dict) or set(records) - set(MODELS):
        raise ValueError(f"Invalid common trigger or unrecognized controller at {row['id']}.")
    acks = {}
    for model, record in records.items():
        if not isinstance(record, dict):
            raise ValueError(f"Malformed controller record for {model} at {row['id']}.")
        if record == {"status": "drop"}:
            acks[model] = None
        elif (record.get("status") == "ack" and type(record.get("ack_ns")) is int and record["ack_ns"] > capture
                and record.get("alarm") is alarms[model]):
            acks[model] = record["ack_ns"]
        else:
            raise ValueError(f"Invalid controller time, drop status or alarm for {model} at {row['id']}.")
    return capture, acks


def summarize(policy, units, observations):
    # Keep only complete predetermined units. Incomplete units remain in the planned denominator.
    quality = {m: {arm: {} for arm in ("clean", *CRITICAL)} for m in MODELS}
    timing = {m: {} for m in MODELS}
    for cid, unit in units.items():
        values = observations.get(cid, {})
        if len(values) != unit["count"]:
            continue
        ordered = [values[i] for i in range(unit["start"], unit["start"]+unit["count"])]
        if unit["arm"] == "timing":
            for m in MODELS:
                if all(item is not None and m in item[1] for item in ordered):
                    timing[m][cid] = timing_block(ordered, m, policy["timing"])
        else:
            for m in MODELS:
                quality[m][unit["arm"]][cid] = any(item[m] for item in ordered)
    targets = policy["sampling"]["coils"]
    rules = policy["acceptance"]
    limits = {arm: (targets[arm], rules["critical_object_recall_floor"], "lower") for arm in CRITICAL}
    limits["clean"] = (targets["clean"], rules["maximum_clean_window_alarm_fraction"], "upper")
    for metric in TIMING:
        limits[metric] = (targets["timing"], rules["minimum_timing_coil_pass_fraction"], "lower")
    systems = {}
    for m in MODELS:
        for metric in TIMING:
            quality[m][metric] = {cid: row["passed"][metric] for cid, row in timing[m].items()}
        gates = {key: exact_screen(quality[m][key].values(), *args, rules["one_sided_confidence"])
                 for key, args in limits.items()}
        complete = all(row["met"] is not None for row in gates.values())
        passed = all(row["met"] for row in gates.values()) if complete else None
        systems[m] = dict(endpoints=gates, timing_by_coil=timing[m], passes_supervised_screen=passed)
    pairs = {}
    for key, (n, _, _) in limits.items():
        left, right = quality["B0"][key], quality["A0"][key]
        common = left.keys() & right.keys()
        pair = paired_binary({c: left[c] for c in common}, {c: right[c] for c in common})
        pair.update(observed_only_B0=len(left)-len(common), observed_only_A0=len(right)-len(common))
        if pair["coils"] != n:
            pair["mcnemar_two_sided_p"] = None
        pairs[key] = pair
    changes = {}
    common_timing = timing["B0"].keys() & timing["A0"].keys()
    for metric in TIMING:
        # Dropped decisions can make the p95 infinite. Expose their count, and never subtract infinities.
        values = [timing["B0"][c][metric]-timing["A0"][c][metric] for c in common_timing
                  if all(timing[m][c][metric] is not None for m in MODELS)]
        changes[metric] = dict(finite_pairs=len(values), unbounded_pairs=len(common_timing)-len(values),
                               median_B0_minus_A0=median(values) if values else None)
    complete = all(system["passes_supervised_screen"] is not None for system in systems.values())
    return dict(models=systems, paired_study_complete=complete, paired_coil_outcomes=pairs, paired_timing_differences=changes)


@torch.inference_mode()
def evaluate(data, artifacts, manifest):
    spec_sha = sha(SPEC)
    policy = verify(artifacts)
    manifest_sha = sha(manifest)
    records = read(manifest)
    if records.get("policy_sha256") != spec_sha:
        raise ValueError("This paired study requires a version-2 enrollment and its frozen policy digest.")
    base = manifest.parent
    plan_file = local_file(base, records["enrollment"])
    roster_file = local_file(base, records["roster"])
    plan_sha = sha(plan_file)
    plan = enrollment(policy, spec_sha, roster_file)
    if plan_sha != records["enrollment_sha256"] or read(plan_file) != plan:
        raise ValueError("The enrollment or blinded roster differs from the frozen selection.")
    units = plan["units"]
    runs = records.get("timing_runs", {})
    expected_graphs = {m: policy["models"][m]["graph_sha256"] for m in MODELS}
    for cid, run in runs.items():
        if (cid not in units or units[cid]["arm"] != "timing" or run.get("rig_by_model") != units[cid]["rig_by_model"]
                or run.get("graph_sha256") != expected_graphs or run.get("clock_id") != plan["campaign"]["clock_id"]):
            raise ValueError(f"Wrong timing cohort, shadow rig assignment, graph or clock for {cid}.")
    benchmark = {r["image_sha"] for s in ("train", "val", "test") for r in read(data / f"{s}.json")}
    ids, hashes = set(), set()
    observed = defaultdict(dict)
    pin_cpu(4)
    engines = {m: Engine(artifacts/"runs"/m, "openvino", a["head"], a["side"], 4) for m,a in policy["models"].items()}
    for row in records["frames"]:
        cid, index, frame_id = row["acquisition_group"], row["frame_index"], row["id"]
        unit = units.get(cid)
        if (unit is None or row["strip_id"] != unit["strip_id"] or type(index) is not int
                or not unit["start"] <= index < unit["start"]+unit["count"] or index in observed[cid]
                or not isinstance(frame_id, str) or not frame_id or frame_id in ids):
            raise ValueError(f"Unscheduled or duplicate coil, strip, exposure index or identity at {frame_id}.")
        path = local_file(base, row["image"])
        encoded = path.read_bytes()
        h = hashlib.sha256(encoded).hexdigest()
        if h in benchmark or h != row["image_sha"]:
            raise ValueError(f"Reused benchmark image or wrong digest at {frame_id}.")
        ids.add(frame_id)
        hashes.add(h)
        gray = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_GRAYSCALE) if encoded else None
        if gray is None:
            raise ValueError(f"Unreadable exposure at {frame_id}.")
        if unit["arm"] != "timing":
            validate_labels(row, gray, unit)
        outcomes = {}
        for m, arm in policy["models"].items():
            boxes = pipeline(engines[m], gray, arm, [arm["all_class_cutoff"]])
            if unit["arm"] in CRITICAL:
                idx = row["object_ids"].index(unit["object_id"])
                outcomes[m] = idx in assignment(row, boxes)
            else:
                outcomes[m] = bool(len(boxes))
        if unit["arm"] == "timing":
            trace = live_trace(row, outcomes)
            if trace is not None and cid not in runs:
                raise ValueError(f"Missing graph, rig and clock provenance for {cid}.")
            observed[cid][index] = trace
        else:
            observed[cid][index] = outcomes
    frozen_inputs = ((SPEC, spec_sha), (manifest, manifest_sha), (plan_file, plan_sha), (roster_file, plan["roster_sha256"]))
    if any(sha(path) != checksum for path, checksum in frozen_inputs):
        raise ValueError("A frozen input changed during the paired evaluation.")
    return dict(policy_sha256=spec_sha, enrollment_sha256=plan_sha, manifest_sha256=manifest_sha,
                eligible_mix=plan["eligible_mix"], selected_mix=plan["selected_mix"], observed_exposures=len(ids),
                repeated_image_content=len(ids)-len(hashes),
                **summarize(policy, units, observed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["verify", "enroll", "evaluate"])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--roster", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.action == "verify":
        verified = verify(args.artifacts)
        for model in MODELS:
            arm = verified["models"][model]
            print(f"Verified {arm['graph']} at {arm['all_class_cutoff']}.")
        print(f"Prospective policy SHA-256: {sha(SPEC)}")
    elif args.action == "enroll":
        if args.roster is None or args.output is None:
            parser.error("enroll requires --roster and --output")
        print(f"Enrollment SHA-256: {freeze_roster(verify(args.artifacts), args.roster, args.output)}")
    else:
        if args.manifest is None:
            parser.error("evaluate requires --manifest")
        print(json.dumps(evaluate(args.data, args.artifacts, args.manifest), indent=2, allow_nan=False))
