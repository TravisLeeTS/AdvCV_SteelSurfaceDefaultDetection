"""Validation-only operating-point selection for the existing dual-head checkpoints."""
from datetime import datetime, timezone
from itertools import groupby
from math import lcm
from pathlib import Path
import gzip
import json

import numpy as np
from filelock import FileLock

from scripts.survey_evidence import overlap
from steel.common import CLASSES, MAX_DET, NMS_IOU, PRE_NMS_CAP, SCORE_FLOOR, put, read, sha
from steel.metrics import choose_threshold, events


PRIMARY_BUDGET = 10                 # False-positive boxes per 100 annotated images.
BUDGETS = (5, PRIMARY_BUDGET, 20, 30)
DEFAULT_OUTPUT = Path("artifacts/cpu-extension/heads")
NO_DETECTIONS = float(np.nextafter(np.float32(1), np.float32(2)))


def cells():
    for model in "ABCDE":
        for seed in (0, 1, 2):
            for runtime in (("openvino", "pytorch") if model in "AE" else ("openvino",)):
                for head in ("one", "dense"):
                    yield model, seed, runtime, head


def load_annotations(data, split):
    manifest = read(data / "split.json")
    source = data / f"{split}.json"
    if sha(source) != manifest["split_files"][split]:
        raise ValueError(f"Annotation digest differs from the recorded split: {source}")
    rows = read(source)
    if (len(rows) != manifest["counts"][split] or len({r["id"] for r in rows}) != len(rows)
            or any(r["split"] != split or len(r["classes"]) != len(r["boxes"]) for r in rows)):
        raise ValueError(f"Image order, population or annotation shape is invalid for {split}")
    return rows


def load_cell(study, rows, split, key, study_freeze, inventory):
    model, seed, runtime, head = key
    run = f"{model}{seed}"
    cell = f"{run}-{runtime}-{head}-480"
    directory = study / split / cell
    meta = read(directory / "cell.json")
    quality = read(directory / "quality.json")
    source = directory / "predictions.jsonl.gz"
    if (meta != quality["metadata"] or quality["predictions_sha"] != sha(source)
            or meta["checkpoint_sha"] != study_freeze["checkpoints"][run]
            or meta["cutoff"] != study_freeze["cutoffs"][run]
            or meta["split_sha"] != study_freeze["split_sha"]
            or (meta["model_id"], meta["seed"], meta["runtime"], meta["head"], meta["split"],
                meta["side"], meta["precision"], meta["cap"], meta["floor"])
            != (model, seed, runtime, head, split, 480, "FP32", MAX_DET, SCORE_FLOOR)):
        raise ValueError(f"The frozen study and prediction identity disagree at {directory}")
    artifact_sha = (inventory[meta["export_bundle"]]["bundle_sha"] if runtime == "openvino"
                    else meta["checkpoint_sha"])
    if artifact_sha != meta["artifact_sha"]:
        raise ValueError(f"Unexpected deployment artifact at {directory}")
    if head == "dense" and (meta["nms_iou"] != NMS_IOU or not meta["single_label_dense"]
                            or meta["pre_nms_cap"] != PRE_NMS_CAP):
        raise ValueError(f"The dense postprocessor differs at {directory}")
    predictions = []
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        for row, event in zip(rows, map(json.loads, stream), strict=True):
            low = np.asarray(event["low"], np.float32).reshape(-1, 6)
            shared = np.asarray(event["operating"], np.float32).reshape(-1, 6)
            if (event["id"] != row["id"] or len(low) != event["low_counts"][-1]
                    or len(low) > MAX_DET or not np.isfinite(low).all()
                    or (low[:, 4] < SCORE_FLOOR).any() or (low[:, 4] > 1).any()
                    or (np.diff(low[:, 4]) > 0).any()
                    or not np.array_equal(low[low[:, 4] >= meta["cutoff"]], shared)):
                raise ValueError(f"Saved score coverage or shared-cutoff replay failed at {source}/{row['id']}")
            predictions.append(low)
    return meta, quality, predictions


def select_budgets(rows, predictions):
    """Independent scalar-IoU matcher with exact integer macro-recall comparisons.

    A score level enters the operating set as a whole. Unmatched predictions
    accumulate monotonically. Once the largest allowance is crossed, a lower
    level cannot qualify for any of the registered budgets.
    """
    n_classes = len(CLASSES)
    gt = [sum(row["classes"].count(c) for row in rows) for c in range(n_classes)]
    if min(gt) < 1:
        raise ValueError("Each calibration class must have labeled objects.")
    denominator = lcm(*gt)
    weight = [denominator // count for count in gt]
    population = len(rows)
    caps = {budget: population * budget // 100 for budget in BUDGETS}
    best = {budget: (0, 0, NO_DETECTIONS) for budget in BUDGETS}
    tp, fp, recall_numerator = [0] * n_classes, 0, 0
    saved = {budget: (tp.copy(), fp) for budget in BUDGETS}
    # Stable sort preserves the original within-image order for tied scores.
    ordered = sorted(((float(box[4]), i, box) for i, det in enumerate(predictions) for box in det),
                     key=lambda v: -v[0])
    available = [list(range(len(row["classes"]))) for row in rows]
    for score, tied in groupby(ordered, key=lambda v: v[0]):
        for _, i, box in tied:
            label = int(box[5])
            row = rows[i]
            candidates = ((j, overlap(box.tolist(), row["boxes"][j]))
                          for j in available[i] if row["classes"][j] == label)
            index, best_iou = max(candidates, key=lambda pair: pair[1], default=(-1, 0.0))
            if best_iou >= 0.5:
                available[i].remove(index)
                tp[label] += 1
                recall_numerator += weight[label]
            else:
                fp += 1
        if fp > max(caps.values()):
            break
        criterion = (recall_numerator, -fp, score)
        for budget, cap in caps.items():
            if fp <= cap and criterion > best[budget]:
                best[budget] = criterion
                saved[budget] = (tp.copy(), fp)
    choices = []
    for budget in BUDGETS:
        score = best[budget][2]
        class_hits, alarms = saved[budget]
        choices.append(dict(fp_budget_per_image=budget / 100, max_validation_fp=caps[budget],
                            cutoff=score, validation_fp=alarms, validation_tp=sum(class_hits),
                            validation_macro_recall=best[budget][0] / (denominator * n_classes),
                            validation_class_tp=dict(zip(CLASSES, class_hits, strict=True))))
    return choices


def verify_floor(rows, predictions, cutoff):
    """Check whether the retained low-score prefix covers this operating point."""
    if cutoff <= SCORE_FLOOR:
        raise ValueError("The proposed cutoff reaches the saved floor. Rerun raw inference.")
    truncated = [row["id"] for row, boxes in zip(rows, predictions, strict=True)
                 if len(boxes) == MAX_DET and float(boxes[-1, 4]) >= cutoff]
    if truncated:
        raise ValueError(f"Operating predictions may intersect the detection cap at {truncated[:4]}")


def freeze_validation(data=Path("data"), artifacts=Path("artifacts"), output=DEFAULT_OUTPUT):
    """Read validation predictions only. A repeat must reproduce the stored choices."""
    data, artifacts, output = Path(data), Path(artifacts), Path(output)
    study = artifacts / "study"
    rows = load_annotations(data, "val")
    original = read(study / "freeze.json")
    inventory = read(study / "export-inventory.json")
    if (original["split_sha"] != read(data / "split.json")["split_hash"]
            or original["export_inventory_sha"] != sha(study / "export-inventory.json")):
        raise ValueError("Split or graph inventory differs from the original freeze.")
    entries = []
    for key in cells():
        model, seed, runtime, head = key
        name = f"{model}{seed}-{runtime}-{head}-480"
        directory = study / "val" / name
        for filename in ("cell.json", "quality.json"):
            if sha(directory / filename) != original["validation_evidence"][f"val/{name}/{filename}"]:
                raise ValueError(f"Validation evidence changed at {directory / filename}")
        meta, quality, predictions = load_cell(study, rows, "val", key, original, inventory)
        choices = select_budgets(rows, predictions)
        for point in choices:
            verify_floor(rows, predictions, point["cutoff"])
        primary = next(p for p in choices if p["fp_budget_per_image"] == PRIMARY_BUDGET / 100)
        independent = choose_threshold(events(rows, predictions))
        if primary["cutoff"] != independent:
            raise ValueError(f"Independent optimum differs for {name}: {primary['cutoff']} vs {independent}")
        entries.append(dict(run=f"{model}{seed}", model=model, seed=seed, runtime=runtime, head=head,
                            side=480, cutoff=primary["cutoff"], validation_fp=primary["validation_fp"],
                            validation_macro_recall=primary["validation_macro_recall"],
                            checkpoint_sha=meta["checkpoint_sha"], artifact_sha=meta["artifact_sha"],
                            export_bundle=meta["export_bundle"], shared_cutoff=meta["cutoff"],
                            validation_source=f"val/{name}", predictions_sha=quality["predictions_sha"],
                            low_floor=meta["floor"], max_retained_boxes=max(map(len, predictions)),
                            operating_points=choices))
    record = dict(schema=1, scope="480 px; 15 existing checkpoints; main OpenVINO FP32 and native A/E controls",
                  selection="Maximize equal-weight six-class recall at IoU >= 0.50. Include full score ties. "
                            "Break ties by fewer false-positive boxes, then higher cutoff.",
                  budget_unit="false-positive boxes per annotated validation image",
                  primary_budget=PRIMARY_BUDGET / 100, sensitivity_budgets=[b / 100 for b in BUDGETS],
                  n_validation_images=len(rows), n_validation_groups=len({r["group"] for r in rows}),
                  original_study_freeze_sha=sha(study / "freeze.json"), validation_annotations_sha=sha(data / "val.json"),
                  design="Post-hoc extension of a previously viewed split. Cutoff selection reads validation only.",
                  independent_check="Exact integer ordering and scalar IoU independently agree with the archived "
                                    "vectorized study threshold selector for every primary cutoff.",
                  entries=entries)
    target = output / "freeze.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target) + ".lock"):
        if target.is_file():
            stored = read(target)
            previous = {k: v for k, v in stored.items() if k != "frozen_utc"}
            if previous != record:
                raise ValueError(f"New validation evidence differs from the existing freeze at {target}")
            return stored
        record["frozen_utc"] = datetime.now(timezone.utc).isoformat()
        put(target, record)
    return record


def iter_head_cutoffs(output=DEFAULT_OUTPUT):
    """Yield the primary operating points and model identity for the fresh timing matrix."""
    record = read(Path(output) / "freeze.json")
    expected = set(cells())
    actual = [(e["model"], e["seed"], e["runtime"], e["head"]) for e in record["entries"]]
    if len(actual) != len(expected) or set(actual) != expected or record["primary_budget"] != PRIMARY_BUDGET / 100:
        raise ValueError("Incomplete or inconsistent head calibration freeze.")
    yield from record["entries"]
