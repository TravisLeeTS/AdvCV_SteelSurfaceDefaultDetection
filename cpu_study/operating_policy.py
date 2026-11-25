"""Rebuild exploratory deployment policies from the published boxes, scores and timings."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from filelock import FileLock

from scripts.verify_research_metrics import rematch
from steel.bench import read_predictions
from steel.common import CLASSES, MAX_DET, SCORE_FLOOR, SIDES, put, read, sha, table
from steel.metrics import choose_threshold, events
from .class_thresholds import fit_vectors
from .head_calibration import load_annotations, verify_floor
from .path_selection import original_sensitivity, unified_selection


PROTOCOL = Path(__file__).with_suffix(".json")


def paths(study, out):
    original = read(study / "freeze.json")
    inventory_path = study / "export-inventory.json"
    if sha(inventory_path) != original["export_inventory_sha"]:
        raise ValueError("The original graph inventory differs from the freeze.")
    for bundle, meta in read(inventory_path).items():
        model, seed, head, side = (meta[k] for k in ("model_id", "seed", "head", "side"))
        if meta["checkpoint_sha"] != original["checkpoints"][f"{model}{seed}"]:
            raise ValueError(f"Checkpoint mismatch at {bundle}")
        if meta["reopened_ir"]["input_shape"] != [1, 3, side, side] or meta["reopened_ir"]["head"] != head:
            raise ValueError(f"Reopened graph mismatch at {bundle}")
        yield dict(run=f"{model}{seed}", model=model, seed=seed, head=head, side=side,
                   artifact_sha=meta["bundle_sha"], checkpoint_sha=meta["checkpoint_sha"])
    for row in read(out / "stock/freeze.json")["entries"]:
        yield {k: row[k] for k in ("run", "model", "seed", "head", "side", "artifact_sha", "checkpoint_sha")}


def key(entry):
    return f"{entry['run']}-openvino-{entry['head']}-{entry['side']}"


def read_path(study, out, original, rows, split, annotation_sha, entry):
    name = key(entry)
    if entry["model"].startswith("yolo"):
        folder = out / "stock" / split / entry["run"]
        quality = read(folder / "quality.json")
        meta = quality["key"]
        if (meta["split"], meta["split_sha"]) != (split, annotation_sha):
            raise ValueError(f"Historical path/split mismatch at {folder}")
        ap = quality["quality"]["map5095"]
    else:
        folder = study / split / name
        quality = read(folder / "quality.json")
        meta = quality["metadata"]
        if read(folder / "cell.json") != meta or (meta["model_id"], meta["seed"], meta["runtime"], meta["head"],
                meta["side"], meta["split"], meta["floor"], meta["cap"], meta["split_sha"]) != (
                entry["model"], entry["seed"], "openvino", entry["head"], entry["side"], split,
                SCORE_FLOOR, MAX_DET, original["split_sha"]):
            raise ValueError(f"Original path/split mismatch at {folder}")
        if split == "val" and sha(folder / "quality.json") != original["validation_evidence"][f"val/{name}/quality.json"]:
            raise ValueError(f"Validation quality differs at {folder}")
        ap = quality["metrics"]["map5095"]
    source = folder / "predictions.jsonl.gz"
    if (meta["artifact_sha"], meta["checkpoint_sha"], sha(source)) != (
            entry["artifact_sha"], entry["checkpoint_sha"], quality["predictions_sha"]):
        raise ValueError(f"Prediction identity differs at {source}")
    saved = read_predictions(source)
    if [r["id"] for r in rows] != [image for image, _ in saved]:
        raise ValueError(f"Image identity or order differs at {source}")
    boxes = [b for _, b in saved]
    for b in boxes:
        if (len(b) > MAX_DET or not np.isfinite(b).all() or (np.diff(b[:, 4]) > 0).any()
                or (b[:, 4] < SCORE_FLOOR).any() or (b[:, 4] > 1).any()
                or not np.isin(b[:, 5], np.arange(len(CLASSES))).all()):
            raise ValueError(f"Invalid low-score boxes at {source}")
    return boxes, events(rows, boxes), ap, quality["predictions_sha"]


def measure_policy(rows, boxes, items, thresholds):
    """Cross-check a filtered prefix per class against fresh scalar matching in every image."""
    cuts = np.asarray(thresholds, dtype=np.float64)
    if cuts.shape != (len(CLASSES),):
        raise ValueError("Expected one threshold per class.")
    verify_floor(rows, boxes, min(thresholds))
    gt, tp, fp = (np.zeros(len(CLASSES), dtype=np.int64) for _ in range(3))
    positive = flagged = localized = clean = clean_flagged = 0
    for row, b, (score, label, flags, target) in zip(rows, boxes, items, strict=True):
        accepted = score >= cuts[label]
        true = np.bincount(label[accepted & flags[:, 0]], minlength=len(CLASSES))
        false = np.bincount(label[accepted & ~flags[:, 0]], minlength=len(CLASSES))
        hit_sets, alarms = [set() for _ in CLASSES], Counter()
        rematch(row, b[b[:, 4] >= cuts[b[:, 5].astype(int)]].tolist(), hit_sets, alarms)
        if any((len(h), alarms[i]) != (true[i], false[i]) for i, h in enumerate(hit_sets)):
            raise ValueError(f"Class-specific rematching differs for {row['id']}")
        gt += target
        tp += true
        fp += false
        if target.sum():
            positive += 1
            flagged += bool(accepted.any())
            localized += bool(true.any())
        else:
            clean += 1
            clean_flagged += bool(accepted.any())
    recalls = tp / gt
    found, false_boxes = int(tp.sum()), int(fp.sum())
    summary = dict(images=len(rows), gt=int(gt.sum()), tp=found, fn=int(gt.sum()) - found,
                   fp=false_boxes, fp_per_image=false_boxes / len(rows), macro=float(recalls.mean()),
                   minimum=float(recalls.min()), micro=found / int(gt.sum()),
                   precision=found / (found + false_boxes) if found + false_boxes else None,
                   positive_images=positive, flagged=flagged, localized=localized,
                   flagged_only_by_fp=flagged - localized, unflagged=positive - flagged,
                   clean_images=clean, clean_flagged=clean_flagged)
    for i, c in enumerate(CLASSES):
        summary.update({f"{c}_gt": int(gt[i]), f"{c}_tp": int(tp[i]),
                        f"{c}_fp": int(fp[i]), f"{c}_recall": float(recalls[i])})
    return summary


def checked_paths(study, out):
    entries = sorted(paths(study, out), key=key)
    if (len(entries) != 81 or len({key(e) for e in entries}) != 81
            or sum(e["side"] == 480 for e in entries) != 36
            or {e["side"] for e in entries} != set(SIDES)):
        raise ValueError("Expected 81 distinct static graph paths including 36 paths at 480.")
    return entries


def run(data, artifacts, destination):
    study, out = artifacts / "study", artifacts / "cpu-extension"
    original = read(study / "freeze.json")
    entries = checked_paths(study, out)
    existing = {key(e): e["cutoff"] for e in [*read(out / "heads/freeze.json")["entries"],
                  *read(out / "stock/freeze.json")["entries"]] if e["runtime"] == "openvino"}
    candidates, sensitivity = unified_selection(out)
    tolerances = original_sensitivity(original) + sensitivity
    primary = next(r for r in sensitivity if r["tolerance_pp"] == 2)
    table(destination / "path-eligibility.csv", candidates)
    table(destination / "recall-tolerance.csv", tolerances)
    fits, summaries, sources = {}, [], {}
    for split in ("val", "test"):
        rows = load_annotations(data, split)
        annotation_sha = sha(data / f"{split}.json")
        if split == "test":
            frozen = read(destination / "freeze.json")
            fits = frozen["fits"]
        for entry in entries:
            name = key(entry)
            boxes, items, ap, file_sha = read_path(study, out, original, rows, split, annotation_sha, entry)
            sources[f"{split}/{name}"] = file_sha
            if split == "val":
                cutoff = choose_threshold(items)
                settings = {"global": [cutoff] * len(CLASSES)}
                if entry["side"] == 480:
                    if cutoff != existing[name]:
                        raise ValueError(f"480-pixel head-specific freeze differs at {name}")
                    settings.update(fit_vectors(items, len(rows) // 10))
                fits[name] = dict(**entry, thresholds=settings, validation_predictions_sha=file_sha)
            for policy, cuts in fits[name]["thresholds"].items():
                values = measure_policy(rows, boxes, items, cuts)
                if split == "val" and 10 * values["fp"] > len(rows):
                    raise ValueError(f"Validation alarm allowance exceeded at {name}")
                summaries.append(dict(split=split, path=name, model=entry["model"], seed=entry["seed"],
                                      head=entry["head"], side=entry["side"], policy=policy, map5095=ap,
                                      **values))
        if split == "val":
            record = dict(protocol=read(PROTOCOL), protocol_sha=sha(PROTOCOL),
                          split_hash=original["split_sha"], validation_annotations_sha=annotation_sha,
                          validation_table_sha=sha(out / "tables/val-paths.csv"),
                          speed_first=primary["selected"], recall_first=primary["recall_first"],
                          eligible=primary["eligible"].split("|"), fits=fits)
            target = destination / "freeze.json"
            if target.exists() and read(target) != record:
                raise ValueError(f"Validation choices differ from {target}")
            if not target.exists():
                put(target, record)
            print(f"Frozen validation choices for {len(fits)} graphs; speed {primary['selected']}, "
                  f"recall {primary['recall_first']}", flush=True)
    table(destination / "operating-results.csv", summaries)
    table(destination / "source-digests.csv", [dict(path=k, sha256=v) for k, v in sources.items()])
    results = {name: sha(destination / name) for name in (
        "operating-results.csv", "path-eligibility.csv", "recall-tolerance.csv", "source-digests.csv")}
    put(destination / "verification.json", dict(freeze_sha=sha(destination / "freeze.json"),
        protocol_sha=sha(PROTOCOL), files=results, splits=2, graphs=len(entries), policy_rows=len(summaries),
        independently_rematched_image_policies=sum(r["images"] for r in summaries)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, help="Write a separate replay for comparison with the release.")
    args = parser.parse_args()
    dest = args.output or args.artifacts / "cpu-extension/policies"
    dest.mkdir(parents=True, exist_ok=True)
    with FileLock(dest / "replay.lock", timeout=0):
        run(args.data, args.artifacts, dest)
