"""Check saved AP with the COCO evaluator and independently rematch the operating detections."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import csv
import gzip
from importlib.metadata import version
from io import StringIO
import json
from pathlib import Path
import sys

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from torchvision.ops import box_iou

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.survey_evidence import overlap
from steel.bench import name
from steel.common import CLASSES, matrix, put, read, sha
from steel.report import decision


def xywh(box):
    x, y, right, bottom = box[:4]
    return [x, y, right-x, bottom-y]


def coco_truth(rows):
    images, annotations = [], []
    for image_id, row in enumerate(rows, 1):
        images.append(dict(id=image_id, file_name=row["id"], width=row["width"], height=row["height"]))
        for label, box in zip(row["classes"], row["boxes"], strict=True):
            bbox = xywh(box)
            annotations.append(dict(id=len(annotations)+1, image_id=image_id, category_id=label+1,
                                    bbox=bbox, xyxy=box, area=bbox[2]*bbox[3], iscrowd=0))
    truth = COCO()
    truth.dataset = dict(info={}, images=images, annotations=annotations,
                         categories=[dict(id=i+1, name=name) for i, name in enumerate(CLASSES)])
    truth.createIndex()
    return truth


def rematch(row, predictions, hits, false_positives):
    free = list(range(len(row["boxes"])))
    for pred in sorted(predictions, key=lambda box: -box[4]):
        label = int(pred[5])
        options = [i for i in free if row["classes"][i] == label]
        scored = [(i, overlap(pred, row["boxes"][i])) for i in options]
        target, score = max(scored, key=lambda pair: pair[1], default=(-1, 0.0))
        if score >= 0.50:
            hits[label].add((row["id"], target))
            free.remove(target)
        else:
            false_positives[label] += 1


class Float32BoxEvaluator(COCOeval):
    """Sensitivity control with COCO matching and accumulation, using FP32 edge coordinates."""

    def computeIoU(self, imgId, catId):
        targets = self._gts[imgId, catId]
        predictions = sorted(self._dts[imgId, catId], key=lambda pred: -pred["score"])[:self.params.maxDets[-1]]
        if not targets or not predictions:
            return []
        target = torch.tensor([item["xyxy"] for item in targets], dtype=torch.float32)
        boxes = torch.tensor([item["xyxy"] for item in predictions], dtype=torch.float32)
        overlaps = box_iou(boxes, target)
        overlaps[(boxes[:, 2:] <= boxes[:, :2]).any(dim=1)] = 0
        return overlaps.numpy().astype(np.float64)


def coco_differences(truth, predicted, metrics, cap, evaluator_type=COCOeval):
    evaluator = evaluator_type(truth, predicted, "bbox")
    evaluator.params.maxDets = [cap]
    evaluator.params.areaRng = [[0, 1e10]]
    evaluator.params.areaRngLbl = ["all"]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, :, 0, 0]
    if np.any(precision < 0):
        raise ValueError("Undefined COCO class or recall entry")
    deltas = dict(map50=float(precision[0].mean())-metrics["map50"],
                  map5095=float(precision.mean())-metrics["map5095"])
    for i, name in enumerate(CLASSES):
        deltas[name+"_ap50"] = float(precision[0, :, i].mean()) - metrics["classes"][name]["ap50"]
    return deltas


def evaluate_cell(directory, rows, truth, denominators):
    quality = read(directory/"quality.json")
    meta = quality["metadata"]
    source = directory/"predictions.jsonl.gz"
    if meta != read(directory/"cell.json") or sha(source) != quality["predictions_sha"]:
        raise ValueError(f"Prediction provenance differs at {directory}")
    detections = []
    hits = [set() for _ in CLASSES]
    false_positives = Counter()
    geometry = Counter()
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        for image_id, (row, record) in enumerate(zip(rows, map(json.loads, stream), strict=True), 1):
            if record["id"] != row["id"] or len(record["low"]) > meta["cap"]:
                raise ValueError(f"Image order or detection cap differs at {directory}")
            for pred in record["low"]:
                if pred[4] < meta["floor"]:
                    raise ValueError(f"Detection below the AP floor at {directory}")
                bbox = xywh(pred)
                geometry["inverted_at_ap_floor"] += int(min(bbox[2:]) < 0)
                geometry["zero_area_at_ap_floor"] += int(bbox[2]*bbox[3] == 0)
                detections.append(dict(image_id=image_id, category_id=int(pred[5])+1, bbox=bbox, xyxy=pred[:4], score=pred[4]))
            if any(pred[4] < meta["cutoff"] for pred in record["operating"]):
                raise ValueError(f"Operating cutoff mismatch at {directory}")
            geometry["nonpositive_dimension_at_cutoff"] += sum(
                pred[2] <= pred[0] or pred[3] <= pred[1] for pred in record["operating"])
            rematch(row, record["operating"], hits, false_positives)
    with redirect_stdout(StringIO()):
        if detections:
            predicted = truth.loadRes(detections)
        else:
            predicted = COCO()
            predicted.dataset = {**truth.dataset, "annotations": []}
            predicted.createIndex()
        differences = coco_differences(truth, predicted, quality["metrics"], meta["cap"])
        controls = None
        if max(map(abs, differences.values())) > 1e-10:
            # COCO ordinarily ignores negative-area detections. The frozen evaluator
            # counts them as unmatched candidates. Only this diagnostic pass changes area.
            for record in predicted.anns.values():
                record["area"] = max(record["area"], 0.)
            admitted = coco_differences(truth, predicted, quality["metrics"], meta["cap"])
            fp32 = coco_differences(truth, predicted, quality["metrics"], meta["cap"], Float32BoxEvaluator)
            controls = dict(retain_inverted_as_false_positives=max(map(abs, admitted.values())),
                            also_use_float32_box_iou=max(map(abs, fp32.values())))
    for i, name in enumerate(CLASSES):
        target = quality["metrics"]["classes"][name]
        expected = (denominators[i], len(hits[i]), denominators[i]-len(hits[i]), false_positives[i])
        if expected != tuple(target[k] for k in ("gt", "tp", "fn", "fp")):
            raise ValueError(f"Operating counts differ for {name} at {directory}")
    if abs(sum(false_positives.values())/len(rows) - quality["metrics"]["fp_per_image"]) > 1e-12:
        raise ValueError(f"Operating FP/image differs at {directory}")
    return differences, hits, false_positives, geometry, controls


def verify_recall_intervals(data, report, matched, freeze):
    """Use paired group totals from independent object matches to reproduce recall intervals."""
    spec = read(Path(__file__).resolve().parents[1]/"protocol.json")["uncertainty"]
    with (report/"paired_intervals.csv").open(newline="", encoding="utf-8") as f:
        intervals = [r for r in csv.DictReader(f) if r["metric"].endswith("_recall")]
    seeds = freeze["seeds"]
    comparisons = {f"{m} minus A at 480" for m in freeze["models"] if m != "A"}
    comparisons.update(f"D minus {m} at 480" for m in ("B", "C") if m in freeze["models"] and "D" in freeze["models"])
    comparisons.update(f"{m} {rt} one minus dense" for m, _, rt, head, _ in matrix(freeze["models"], seeds)
                       if head == "dense")
    if chosen := freeze["selection"]["model"]:
        comparisons.update(f"{chosen} {side} minus 480" for side in (256, 320, 640))
    expected_keys = {(split, comp, seed, c+"_recall") for split in ("val", "test") for comp in comparisons
                     for seed in [*map(str, seeds), "paired mean"] for c in CLASSES}
    keys = {(r["split"], r["comparison"], r["seed"], r["metric"]) for r in intervals}
    if keys != expected_keys or len(intervals) != len(keys):
        raise ValueError("The paired recall intervals have a missing, extra or duplicate comparison")
    for split in ("val", "test"):
        rows = read(data/(split+".json"))
        groups, inverse = np.unique([r["group"] for r in rows], return_inverse=True)
        image_group = {r["id"]: g for r, g in zip(rows, inverse, strict=True)}
        gt = np.zeros((len(groups), len(CLASSES)), dtype=int)
        for r, g in zip(rows, inverse, strict=True):
            gt[g] += np.bincount(r["classes"], minlength=len(CLASSES))
        rng = np.random.default_rng(spec["seed"])
        weights = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=spec["draws"])
        denominator = weights @ gt
        rates, observed = {}, {}
        for cell, hits in matched.items():
            if cell.split("/")[0] != split:
                continue
            counts = np.zeros_like(gt)
            for c, targets in enumerate(hits):
                for image, _ in targets:
                    counts[image_group[image], c] += 1
            rates[cell] = (weights @ counts)/denominator
            observed[cell] = counts.sum(axis=0)/gt.sum(axis=0)
        for row in (r for r in intervals if r["split"] == split):
            column = CLASSES.index(row["metric"].removesuffix("_recall"))
            pairs = list(zip(row["left_source"].split("|"), row["right_source"].split("|"), strict=True))
            draws = np.mean([rates[left][:, column]-rates[right][:, column] for left, right in pairs], axis=0)
            delta = np.mean([observed[left][column]-observed[right][column] for left, right in pairs])
            expected = [float(row[k]) for k in ("delta_pp", "low_pp", "high_pp")]
            if int(row["groups"]) != len(groups) or int(row["draws"]) != len(draws):
                raise ValueError(f"Group population differs for {row['comparison']}")
            np.testing.assert_allclose(100*np.r_[delta, np.percentile(draws, [2.5, 97.5])], expected, rtol=0, atol=1e-9)
    return len(intervals)


def verify(data, artifacts, output):
    study = artifacts/"study"
    report = study/"report"
    with (report/"cells.csv").open(newline="", encoding="utf-8") as f:
        cells = list(csv.DictReader(f))
    freeze = read(study/"freeze.json")
    planned = {f"{s}/{name(key)}" for s in ("val", "test") for key in matrix(freeze["models"], freeze["seeds"])}
    if {r["cell_dir"] for r in cells} != planned or len(cells) != len(planned):
        raise ValueError("Result table coverage differs from the frozen inference matrix")
    split_manifest = read(data/"split.json")
    if split_manifest["split_hash"] != freeze["split_sha"]:
        raise ValueError("Input split differs from the frozen study")
    max_difference = 0.0
    differences = []
    geometry_totals = Counter()
    geometry_by_cell, sensitivity = {}, {}
    matched, fps, denoms, validation_rows = {}, {}, {}, {}
    for split in ("val", "test"):
        source = data/(split+".json")
        if sha(source) != split_manifest["split_files"][split]:
            raise ValueError(f"Saved {split} annotations differ from their manifest")
        rows = read(source)
        denoms[split] = Counter(label for row in rows for label in row["classes"])
        with redirect_stdout(StringIO()):
            truth = coco_truth(rows)
        for r in (r for r in cells if r["split"] == split):
            cell = r["cell_dir"]
            deltas, hits, false_positives, geometry, controls = evaluate_cell(study/cell, rows, truth, denoms[split])
            geometry_totals.update(geometry)
            if any(geometry.values()):
                geometry_by_cell[cell] = dict(geometry)
            if controls is not None:
                sensitivity[cell] = controls
            if split == "val" and r["runtime"] == "openvino" and r["head"] == "one":
                fields = ["map5095", "fp_per_image", "pipeline_p95_ms", "p95_min_ms", "p95_max_ms",
                          *(c+"_recall" for c in CLASSES), *(c+"_ap50" for c in CLASSES)]
                key = (split, r["model_id"], int(r["seed"]), r["runtime"], r["head"], int(r["side"]))
                validation_rows[key] = {k: float(r[k]) + deltas.get(k, 0) for k in fields}
            for metric, value in deltas.items():
                max_difference = max(abs(value), max_difference)
                if abs(value) > 1e-10:
                    differences.append(dict(cell=cell, metric=metric, coco_minus_study=value))
            matched[cell], fps[cell] = hits, false_positives
        print(f"Checked {split} AP and operating detections.", flush=True)

    with (report/"matched_objects.csv").open(newline="", encoding="utf-8") as f:
        paired = list(csv.DictReader(f))
    expected_pairs = set()
    for file, kind, ref_key, cand_key in (("head_comparison", "head", "dense_source", "one_source"),
                                           ("architecture_comparison", "architecture", "reference_cell_dir", "cell_dir"),
                                           ("runtime_comparison", "runtime", "pytorch_source", "openvino_source")):
        with (report/(file+".csv")).open(newline="", encoding="utf-8") as stream:
            expected_pairs.update((kind, r[ref_key], r[cand_key], c) for r in csv.DictReader(stream) for c in CLASSES)
    actual_pairs = {(r["comparison"], r["reference_source"], r["candidate_source"], r["defect"]) for r in paired}
    if actual_pairs != expected_pairs or len(paired) != len(expected_pairs):
        raise ValueError("Matched-object table has missing, duplicate or extra class comparisons")
    changed_runtime_ids = set()
    for r in paired:
        i = CLASSES.index(r["defect"])
        ref, cand = r["reference_source"], r["candidate_source"]
        left, right = matched[ref][i], matched[cand][i]
        gt = denoms[r["split"]][i]
        independent = dict(gt=gt, reference_tp=len(left), candidate_tp=len(right), both=len(left & right),
                           reference_only=len(left-right), candidate_only=len(right-left),
                           neither=gt-len(left | right), reference_fp=fps[ref][i], candidate_fp=fps[cand][i])
        if (any(int(r[k]) != v for k, v in independent.items())
                or abs(float(r["recall_change_pp"])-100*(len(right)-len(left))/gt) > 1e-10):
            raise ValueError(f"Independent matched-object counts differ at {ref}, {cand}, {r['defect']}")
        if r["comparison"] == "runtime":
            changed_runtime_ids.update((ref, cand, image, index) for image, index in left ^ right)
    with (report/"runtime_disagreements.csv").open(newline="", encoding="utf-8") as f:
        listed = [(r["pytorch_source"], r["openvino_source"], r["image_id"], int(r["annotation_index"])) for r in csv.DictReader(f)]
    if set(listed) != changed_runtime_ids or len(listed) != len(changed_runtime_ids):
        raise ValueError("The listed runtime discrepancies differ from the independent annotation matches")
    recall_intervals = verify_recall_intervals(data, report, matched, freeze)
    sensitivity_selection = decision(validation_rows, freeze["models"], freeze["seeds"])
    def eligibility(sel):
        return dict(model=sel["model"], side=sel["side"], verdict=sel["size_verdict"],
                    architecture={r["model"]: r["eligible"] for r in sel["architecture"]},
                    size={str(r["side"]): r["eligible"] for r in sel["sizes"]})
    put(output, dict(evaluator="pycocotools "+version("pycocotools"), settings="All area, six classes, 101 recall points, IoU 0.50 to 0.95, maxDets 300, stored 0.001 score floor.",
                     cells=len(cells), class_operating_rows=len(cells)*len(CLASSES), matched_class_rows=len(paired),
                     paired_recall_intervals_reproduced=recall_intervals,
                     maximum_absolute_ap_difference=max_difference, ap_differences_above_1e_10=differences,
                     geometry_totals=dict(geometry_totals), geometry_by_cell=geometry_by_cell,
                     sensitivity_maximum_absolute_ap_difference_per_cell=sensitivity,
                     frozen_validation_eligibility=eligibility(freeze["selection"]),
                     standard_coco_validation_eligibility=eligibility(sensitivity_selection),
                     operating_counts_method="Independent score-ordered scalar IoU matching to the best unused same-class annotation.",
                     matched_objects_sha256=sha(report/"matched_objects.csv"), study_freeze_sha256=sha(study/"freeze.json")))
    print(f"Checked {len(cells)} cells and {len(paired)} paired class rows. Maximum AP difference {max_difference:.12g}.")
    if any(item["also_use_float32_box_iou"] > 1e-10 for item in sensitivity.values()):
        raise ValueError(f"AP differences remain after the documented controls. See {output}.")
    if eligibility(sensitivity_selection) != eligibility(freeze["selection"]):
        raise ValueError(f"Standard COCO AP changes validation eligibility. See {output}.")
    if set("ABCD").issubset(freeze["models"]):
        from scripts.verify_factorial import verify as verify_factorial
        verify_factorial(data, artifacts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verify(args.data, args.artifacts, args.output or args.artifacts/"audit/research-metrics-check.json")
