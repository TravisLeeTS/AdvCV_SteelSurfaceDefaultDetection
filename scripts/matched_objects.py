"""Pair detections on the same annotated object in the frozen 480-pixel comparisons."""
import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Rectangle
import numpy as np

from steel.bench import read_predictions
from steel.common import CLASSES, put, read, sha, table
from steel.data import gray_image
from steel.figures import assignment
from steel.metrics import iou
from steel.report import PRIORITY


@dataclass
class CellMatches:
    metadata: dict
    hits: np.ndarray
    false_positives: np.ndarray
    predictions_sha: str


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def match_cell(study, source, annotations, labels, freeze, inventory):
    directory = study/source
    meta = read(directory/"cell.json")
    quality = read(directory/"quality.json")
    predictions = directory/"predictions.jsonl.gz"
    run = f"{meta['model_id']}{meta['seed']}"
    if (meta != quality["metadata"] or meta["split_sha"] != freeze["split_sha"]
            or meta["checkpoint_sha"] != freeze["checkpoints"][run] or meta["cutoff"] != freeze["cutoffs"][run]
            or meta["side"] != 480 or meta["precision"] != "FP32"
            or sha(predictions) != quality["predictions_sha"]):
        raise ValueError(f"Frozen prediction identity mismatch at {directory}")
    artifact = inventory[meta["export_bundle"]]["bundle_sha"] if meta["runtime"] == "openvino" else meta["checkpoint_sha"]
    if meta["artifact_sha"] != artifact:
        raise ValueError(f"Deployment artifact mismatch at {directory}")

    hits = np.zeros(len(labels), dtype=bool)
    detection_counts = np.zeros(len(CLASSES), dtype=int)
    offset = 0
    for row, (image_id, det) in zip(annotations, read_predictions(predictions, "operating"), strict=True):
        if row["id"] != image_id or row["split"] != meta["split"] or np.any(det[:, 4] < meta["cutoff"]):
            raise ValueError(f"Image order, split or operating cutoff differs at {directory}/{image_id}")
        for index in assignment(row, det):
            hits[offset + index] = True
        detection_counts += np.bincount(det[:, 5].astype(int), minlength=len(CLASSES))
        offset += len(row["classes"])
    gt = np.bincount(labels, minlength=len(CLASSES))
    tp = np.bincount(labels[hits], minlength=len(CLASSES))
    fp = detection_counts - tp
    for i, name in enumerate(CLASSES):
        saved = quality["metrics"]["classes"][name]
        if (gt[i], tp[i], gt[i]-tp[i], fp[i]) != tuple(saved[k] for k in ("gt", "tp", "fn", "fp")):
            raise ValueError(f"Reconstructed counts differ for {name} in {directory}")
    return CellMatches(meta, hits, fp, quality["predictions_sha"])


def paired_counts(kind, reference, candidate, reference_cell, candidate_cell, labels):
    a, b = reference_cell.metadata, candidate_cell.metadata
    if any(a[key] != b[key] for key in ("split", "seed", "side", "precision", "split_sha")):
        raise ValueError(f"Incompatible comparison: {reference} and {candidate}")
    axis = {"head": "head", "architecture": "model_id", "runtime": "runtime"}[kind]
    for key in ("head", "model_id", "runtime"):
        if (a[key] != b[key]) != (key == axis):
            raise ValueError(f"Unexpected {key} in the {kind} comparison: {reference} and {candidate}")
    for i, name in enumerate(CLASSES):
        mask = labels == i
        left, right = reference_cell.hits[mask], candidate_cell.hits[mask]
        ref_only = int((left & ~right).sum())
        new_only = int((~left & right).sum())
        yield dict(split=a["split"], comparison=kind, seed=a["seed"], precision=a["precision"], side=a["side"],
                   defect=name, gt=len(left), both=int((left & right).sum()), reference_only=ref_only,
                   candidate_only=new_only, neither=int((~left & ~right).sum()),
                   reference_tp=int(left.sum()), candidate_tp=int(right.sum()),
                   reference_fp=int(reference_cell.false_positives[i]), candidate_fp=int(candidate_cell.false_positives[i]),
                   recall_change_pp=100*(new_only-ref_only)/len(left) if len(left) else None,
                   reference_model=a["model_id"], candidate_model=b["model_id"], reference_head=a["head"],
                   candidate_head=b["head"], reference_runtime=a["runtime"], candidate_runtime=b["runtime"],
                   reference_cutoff=a["cutoff"], candidate_cutoff=b["cutoff"],
                   reference_source=reference, candidate_source=candidate,
                   reference_checkpoint_sha=a["checkpoint_sha"], candidate_checkpoint_sha=b["checkpoint_sha"],
                   reference_artifact_sha=a["artifact_sha"], candidate_artifact_sha=b["artifact_sha"],
                   reference_predictions_sha=reference_cell.predictions_sha, candidate_predictions_sha=candidate_cell.predictions_sha)


def choose_examples(rows, dense, one):
    """First priority-class annotation in manifest order for each direction of disagreement."""
    choices = {}
    offset = 0
    for row in rows:
        for i, label in enumerate(row["classes"]):
            key = offset+i
            if CLASSES[label] not in PRIORITY or dense.hits[key] == one.hits[key]:
                continue
            winner = "dense" if dense.hits[key] else "one"
            choices.setdefault(winner, (row, i))
        offset += len(row["classes"])
    return choices


def example_detections(row, index, record, cutoff):
    box = np.asarray(row["boxes"][index], dtype=np.float32).reshape(1, 4)
    for field in ("operating", "low"):
        det = np.asarray(record[field], dtype=np.float32).reshape(-1, 6)
        match = assignment(row, det).get(index)
        if match is None:
            continue
        pred = det[match]
        status = "matched"
        if field == "low":
            status = "below cutoff" if pred[4] < cutoff else "matched at score floor only"
        return dict(status=status, prediction=pred.tolist(), score=float(pred[4]),
                    iou=float(iou(pred[None, :4], box)[0, 0]))
    return dict(status="no assigned match", prediction=None, score=None, iou=None)


def records_for_ids(study, source, image_ids):
    with gzip.open(study/source/"predictions.jsonl.gz", "rt", encoding="utf-8") as stream:
        return {r["id"]: r for r in map(json.loads, stream) if r["id"] in image_ids}


def runtime_disagreements(study, annotations, ends, reference, candidate, reference_cell, candidate_cell):
    changed = np.flatnonzero(reference_cell.hits != candidate_cell.hits)
    if not len(changed):
        return
    indices = np.searchsorted(ends, changed, side="right")
    image_ids = {annotations[i]["id"] for i in indices}
    pytorch = records_for_ids(study, reference, image_ids)
    openvino = records_for_ids(study, candidate, image_ids)
    cutoff = reference_cell.metadata["cutoff"]
    for flat_index, row_index in zip(changed, indices, strict=True):
        row = annotations[row_index]
        index = int(flat_index - (ends[row_index-1] if row_index else 0))
        left = example_detections(row, index, pytorch[row["id"]], cutoff)
        right = example_detections(row, index, openvino[row["id"]], cutoff)
        yield dict(split=row["split"], image_id=row["id"], annotation_index=index, defect=CLASSES[row["classes"][index]],
                   pytorch_source=reference, openvino_source=candidate, cutoff=cutoff,
                   pytorch_status=left["status"], pytorch_score=left["score"], pytorch_iou=left["iou"],
                   openvino_status=right["status"], openvino_score=right["score"], openvino_iou=right["iou"])


def head_examples(data, study, annotations, cells, figures):
    names = {h: f"test/E0-openvino-{h}-480" for h in ("dense", "one")}
    examples = choose_examples(annotations, cells[names["dense"]], cells[names["one"]])
    ids = {row["id"] for row, _ in examples.values()}
    records = {head: records_for_ids(study, source, ids) for head, source in names.items()}
    meta = cells[names["one"]].metadata
    cases = []
    for winner in ("dense", "one"):
        if winner not in examples:
            continue
        row, index = examples[winner]
        if sha(data/row["image"]) != row["image_sha"]:
            raise ValueError(f"Source image differs on {row['id']}")
        heads = {h: example_detections(row, index, rec[row["id"]], meta["cutoff"]) for h, rec in records.items()}
        if [head for head in heads if heads[head]["status"] == "matched"] != [winner]:
            raise ValueError(f"Illustration disagrees with the object count on {row['id']}")
        cases.append(dict(image_id=row["id"], annotation_index=index, defect=CLASSES[row["classes"][index]],
                          gt_box=row["boxes"][index], image_sha=row["image_sha"], winner=winner, heads=heads))
    put(figures/"yolov10_head_cases.json",
        dict(selection="First crazing or rolled-in-scale annotation in saved test order in each direction of disagreement.",
             coordinates="Zero-based half-open edges on the original image. annotation_index is zero-based.",
             cutoff=meta["cutoff"], diagnostic_score_floor=meta["floor"], checkpoint_sha=meta["checkpoint_sha"],
             sources=names, artifact_sha={h: cells[n].metadata["artifact_sha"] for h, n in names.items()}, cases=cases))
    return cases


def draw_box(ax, box, color, linestyle):
    x1, y1, x2, y2 = box[:4]
    overlay = Rectangle((x1,y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=1.6, linestyle=linestyle)
    overlay.set_path_effects([patheffects.Stroke(linewidth=2.4, foreground="#253135"), patheffects.Normal()])
    ax.add_patch(overlay)


def draw_examples(data, annotations, cases, output):
    plt.rcParams["pdf.fonttype"] = 42
    by_id = {r["id"]: r for r in annotations}
    fig, axes = plt.subplots(max(len(cases), 1), 2, figsize=(6.6, 3.15*max(len(cases), 1)),
                             squeeze=False, layout="constrained")
    for i, case in enumerate(cases):
        gray = gray_image(data, by_id[case["image_id"]])
        height, width = gray.shape
        for ax, head, title in zip(axes[i], ("dense", "one"), ("Dense + NMS", "One-to-one"), strict=True):
            det = case["heads"][head]
            ax.imshow(gray, cmap="gray", vmin=0, vmax=255, extent=(0,width,height,0), interpolation="nearest")
            draw_box(ax, case["gt_box"], "#ffd461", "-")
            if det["prediction"] is not None:
                draw_box(ax, det["prediction"], "#5af4ee", "-" if det["status"] == "matched" else "--")
            ax.set(xticks=[], yticks=[], xlim=(0,width), ylim=(height,0))
            ax.set_title(f"{case['image_id']}  /  box {case['annotation_index']+1}  /  {title}", fontsize=10)
            detail = det["status"].capitalize()
            if det["score"] is not None:
                detail += f"   Score {det['score']:.3f}   IoU {det['iou']:.3f}"
            ax.set_xlabel(detail, fontsize=9)
    if not cases:
        for ax in axes.flat:
            ax.set_axis_off()
        fig.text(.5, .5, "Both heads matched the same priority-class objects.", ha="center")
    fig.savefig(output, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def paper_tables(heads, selected_rows, generated):
    lines = [r"\begin{tabular}{llrrrrrrrr}", r"\hline",
             r" & & \multicolumn{2}{c}{Pipeline p95 (ms)} & \multicolumn{2}{c}{Paired saving} & NMS & \multicolumn{3}{c}{mAP50--95} \\",
             r"Model & CPU & Dense & One & ms & \% & p95 (ms) & Dense (\%) & One (\%) & $\Delta$ (pp) \\", r"\hline"]
    groups = {}
    for row in heads:
        if row["split"] == "test":
            groups.setdefault((row["model_id"], row["runtime"]), []).append(row)
    for (model, runtime), rows in groups.items():
        fields = [model, {"openvino": "OV", "pytorch": "PT"}[runtime]]
        for key, multiplier, fmt in (("dense_p95_ms",1,".2f"), ("one_p95_ms",1,".2f"),
                                      ("saving_ms",1,"+.3f"), ("saving_percent",1,"+.2f"), ("dense_nms_p95_ms",1,".3f"),
                                      ("dense_map5095",100,".2f"), ("one_map5095",100,".2f"), ("one_minus_dense_map5095",100,"+.2f")):
            fields.append(format(multiplier*mean(float(r[key]) for r in rows), fmt))
        lines.append(" & ".join(fields) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    (generated/"head_tradeoff.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")

    keys = ("gt", "both", "reference_only", "candidate_only", "neither")
    lines = [r"\begin{tabular}{lrrrrr}", r"\hline", r"Defect & GT & Both & Dense only & One-to-one only & Neither \\", r"\hline"]
    for r in selected_rows:
        lines.append(" & ".join([r["defect"].replace("_", " "), *(str(r[k]) for k in keys)]) + r" \\")
    lines.extend([r"\hline", " & ".join(["All objects", *(str(sum(r[k] for r in selected_rows)) for k in keys)]) + r" \\",
                  r"\hline", r"\end{tabular}"])
    (generated/"yolov10_matched.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")


def make_matched_objects(data, study, generated):
    report = study/"report"
    figures = report/"figures"
    figures.mkdir(parents=True, exist_ok=True)
    freeze = read(study/"freeze.json")
    manifest = read(data/"split.json")
    inventory_path = study/"export-inventory.json"
    if manifest["split_hash"] != freeze["split_sha"] or sha(inventory_path) != freeze["export_inventory_sha"]:
        raise ValueError("The split or export inventory disagrees with the frozen study")
    inventory = read(inventory_path)
    heads = csv_rows(report/"head_comparison.csv")
    pairs = [("head", r["dense_source"], r["one_source"]) for r in heads]
    pairs += [("architecture", r["reference_cell_dir"], r["cell_dir"]) for r in csv_rows(report/"architecture_comparison.csv")]
    pairs += [("runtime", r["pytorch_source"], r["openvino_source"]) for r in csv_rows(report/"runtime_comparison.csv")]
    cells = {}
    counts = []
    runtime_cases = []
    for split in ("val", "test"):
        source = data/(split+".json")
        if sha(source) != manifest["split_files"][split]:
            raise ValueError(f"Frozen {split} annotations differ from the source manifest")
        annotations = read(source)
        labels = np.asarray([c for r in annotations for c in r["classes"]], dtype=int)
        ends = np.cumsum([len(r["classes"]) for r in annotations])
        for kind, left, right in pairs:
            if Path(left).parts[0] != split:
                continue
            for name in (left, right):
                if name not in cells:
                    cells[name] = match_cell(study, name, annotations, labels, freeze, inventory)
            counts.extend(paired_counts(kind, left, right, cells[left], cells[right], labels))
            if kind == "runtime":
                runtime_cases.extend(runtime_disagreements(study, annotations, ends, left, right, cells[left], cells[right]))
        if split == "test":
            cases = head_examples(data, study, annotations, cells, figures)
            draw_examples(data, annotations, cases, figures/"yolov10_head_cases.pdf")
    table(report/"matched_objects.csv", counts)
    table(report/"runtime_disagreements.csv", runtime_cases,
          fields=("split", "image_id", "annotation_index", "defect", "pytorch_source", "openvino_source", "cutoff",
                  "pytorch_status", "pytorch_score", "pytorch_iou", "openvino_status", "openvino_score", "openvino_iou"))
    selected_rows = [r for r in counts if r["comparison"] == "head" and r["candidate_source"] == "test/E0-openvino-one-480"]
    paper_tables(heads, selected_rows, generated)
    copyfile(figures/"yolov10_head_cases.pdf", generated/"yolov10_head_cases.pdf")
    return selected_rows, cases, runtime_cases
