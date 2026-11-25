"""Compare PConv dense at 320 and 480 with validation-fitted cutoffs and paired CPU sessions."""
import argparse
import csv
import gzip
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from statistics import median
from time import monotonic

import numpy as np
import psutil
import torch
from filelock import FileLock

from steel.bench import Engine, image_order, measure, read_predictions
from steel.common import CLASSES, MAX_DET, SCORE_FLOOR, digest, put, read, replace_file, sha, table
from steel.data import gray_image
from steel.export import export_cell
from steel.hardware import ac_status, cpu_status, pin_cpu
from steel.ir import checked_bundle
from steel.metrics import choose_threshold, events, summarize
from .benchmark import require_idle_power
from .head_calibration import load_annotations, select_budgets
from .heads import grouped_draws
from .feedback_audit import labeled_counts, weighted_rates
from .operating_policy import measure_policy
from . import policy_timing
from .policy_timing import pipeline, verify_boxes


def source(artifacts, dest, split, graph, side):
    return (artifacts / "study" if side == 480 else dest) / split / graph / "predictions.jsonl.gz"


def low_scores(path, engine, rows, images, entry):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending")
    with gzip.open(pending, "wt", encoding="utf-8") as stream:
        for row, gray in zip(rows, images, strict=True):
            boxes = pipeline(engine, gray, entry, [SCORE_FLOOR])
            stream.write(json.dumps(dict(id=row["id"], low=boxes.tolist()), separators=(",", ":")) + "\n")
    replace_file(pending, path)


def quality(data, artifacts, dest, collect):
    from scripts.verify_research_metrics import Float32BoxEvaluator, coco_differences, coco_truth, xywh
    old = read(artifacts / "cpu-extension/policies/freeze.json")
    graph_entries = {}
    for seed in range(3):
        run = artifacts / "runs" / f"B{seed}"
        for side in (320, 480):
            if collect and side == 320:
                export_cell(data, run, "dense", side)
            _, ir = checked_bundle(run, "dense", side)
            graph = f"{run.name}-openvino-dense-{side}"
            graph_entries[graph] = dict(graph=graph, seed=seed, run=run.name, side=side, head="dense",
                policy="global", artifact_sha=ir["bundle_sha"], checkpoint_sha=ir["checkpoint_sha"])
            existing = old["fits"][f"{run.name}-openvino-dense-480"]
            if ir["checkpoint_sha"] != existing["checkpoint_sha"] or (side == 480 and ir["bundle_sha"] != existing["artifact_sha"]):
                raise ValueError(f"Wrong graph for {graph}")
    frozen, results, effects = {}, [], []
    spec = read(Path(__file__).resolve().parents[1] / "protocol.json")["uncertainty"]
    for split in ("val", "test"):
        if split == "test":
            frozen = read(dest / "freeze.json")["fits"]
        rows = load_annotations(data, split)
        groups, draws = grouped_draws(rows, spec)
        weights = np.vstack([np.ones((1,len(rows)),dtype=np.int64),draws])
        boot = {}
        with redirect_stdout(StringIO()):
            truth = coco_truth(rows)
        images = checked_images(data, rows) if collect else None
        for graph, entry in graph_entries.items():
            side = entry["side"]
            path = source(artifacts, dest, split, graph, side)
            folder = dest / split / graph
            identity = dict(artifact_sha=entry["artifact_sha"], split_sha=sha(data / f"{split}.json"), floor=SCORE_FLOOR)
            receipt = folder / "predictions.json"
            if collect and not receipt.exists():
                if side == 320:
                    engine = Engine(artifacts / "runs" / entry["run"], "openvino", "dense", side, 4)
                    low_scores(path, engine, rows, images, entry)
                put(receipt, dict(**identity, sha256=sha(path)))
            if read(receipt) != dict(**identity, sha256=sha(path)):
                raise ValueError(f"Changed predictions: {path}")
            if side == 480:
                original = read(path.parent / "quality.json")
                if original["predictions_sha"] != sha(path) or original["metadata"]["artifact_sha"] != entry["artifact_sha"]:
                    raise ValueError(f"Changed original 480 graph or detections: {path}")
            saved = read_predictions(path)
            if [r["id"] for r in rows] != [i for i, _ in saved]:
                raise ValueError(f"Wrong image order at {path}")
            boxes = [b for _, b in saved]
            items = events(rows, boxes)
            if split == "val":
                cutoff = choose_threshold(items)
                alternate = next(p for p in select_budgets(rows, boxes) if p["fp_budget_per_image"] == .10)
                if cutoff != alternate["cutoff"] or (side == 480 and cutoff != old["fits"][graph]["thresholds"]["global"][0]):
                    raise ValueError(f"Validation optimizers or existing freeze disagree at {graph}")
                frozen[graph] = dict(**entry, thresholds=[cutoff] * len(CLASSES), val_boxes_sha=sha(path))
            cuts = frozen[graph]["thresholds"]
            values = measure_policy(rows, boxes, items, cuts)
            boot[entry["seed"],side], _ = weighted_rates(*labeled_counts(rows, boxes, items, cuts), weights)
            if split == "val" and values["fp"] > len(rows) // 10:
                raise ValueError(f"False-box allowance exceeded for {graph}")
            metrics = summarize(items, cuts[0])
            detections = [dict(image_id=i,category_id=int(b[5])+1,bbox=xywh(b),xyxy=b[:4],score=b[4])
                          for i,frame in enumerate(boxes,1) for b in frame.tolist()]
            with redirect_stdout(StringIO()):
                observed = truth.loadRes(detections)
                for box in observed.anns.values():
                    box["area"] = max(0.,box["area"])
                delta = coco_differences(truth,observed,metrics,MAX_DET,Float32BoxEvaluator)
            if max(map(abs,delta.values())) > 1e-9:
                raise ValueError(f"Independent COCO evaluation differs for {graph}/{split}")
            results.append(dict(split=split, graph=graph, seed=entry["seed"], side=side, cutoff=cuts[0],
                                map5095=metrics["map5095"], coco_max_abs_gap=max(map(abs,delta.values())), **values))
        for metric in ("macro_recall","crazing_recall","rolled_in_scale_recall","correct_match","false_boxes_per_defective_image"):
            delta = np.mean([boot[i,480][metric]-boot[i,320][metric] for i in range(3)],axis=0)
            low, high = np.percentile(delta[1:],[2.5,97.5])
            effects.append(dict(split=split,metric=metric,contrast="480 minus 320",difference=delta[0],
                low_95=low,high_95=high,groups=len(groups),draws=spec["draws"],
                unit="boxes/defective image" if metric=="false_boxes_per_defective_image" else "percentage points"))
        if split == "val":
            freeze = dict(rule="Per seed and size, maximize six-class macro recall at IoU 0.50 with at most 36 false boxes on 360 defective validation images. Whole score ties, fewer FP, then higher cutoff.",
                selection="Exploratory paired-size comparison. The separate supervised candidate remains fixed at 480.",
                validation_sha=sha(data / "val.json"), fits=frozen)
            target = dest / "freeze.json"
            if collect and not target.exists():
                put(target, freeze)
            if read(target) != freeze:
                raise ValueError("Recomputed validation freeze differs.")
    return results, frozen, effects


def checked_images(data, rows):
    if any(sha(data / row["image"]) != row["image_sha"] for row in rows):
        raise ValueError("Input image differs from the frozen split.")
    return [gray_image(data, row) for row in rows]


def collect_session(artifacts, folder, entry, rows, images, path, order, identity):
    require_idle_power(artifacts / "cpu-extension")
    engine = Engine(artifacts / "runs" / entry["run"], "openvino", "dense", entry["side"], 4)
    cutoffs = np.asarray(entry["thresholds"], dtype=np.float32)
    verify_boxes(folder, engine, rows, images, entry, cutoffs, path)
    before = cpu_status()
    next_check = monotonic() + 1
    def invoke(i):
        nonlocal next_check
        if monotonic() >= next_check:
            if ac_status() is not True:
                raise RuntimeError("AC disconnected during timing.")
            next_check = monotonic() + 1
        return pipeline(engine, images[i], entry, cutoffs)
    session = identity["session"]
    records, summary = measure(invoke, order, 200, 1000, "pipeline", rows, session, identity["slot"])
    after = cpu_status()
    if not after["ac_online"] or after["power_profile"] != before["power_profile"]:
        raise RuntimeError("Power profile changed during timing.")
    path = folder / f"timings-s{session}.csv.gz"
    pending = path.with_suffix(".pending")
    with gzip.open(pending, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    replace_file(pending, path)
    put(folder / f"session-{session}.json", dict(identity=identity, sha256=sha(path), summary=summary,
        runtime=engine.precision, before=before, after=after, affinity=psutil.Process().cpu_affinity()))


def timing(data, artifacts, dest, fits, collect):
    results = []
    freeze_sha = sha(dest / "freeze.json")
    graphs = sorted(fits)
    for split_index, split in enumerate(("val", "test")):
        rows = load_annotations(data, split)
        images = checked_images(data, rows) if collect else None
        for session in range(3):
            order = image_order(len(rows), 1200, session)
            rng = np.random.default_rng(np.random.SeedSequence([20261125, 321, split_index, session]))
            for slot, index in enumerate(rng.permutation(len(graphs))):
                graph = graphs[index]
                entry = fits[graph]
                folder = dest / split / graph
                path = source(artifacts, dest, split, graph, entry["side"])
                identity = dict(graph=graph, freeze_sha=freeze_sha, split=split, session=session, slot=slot,
                                order_sha=digest(order.tolist()), measurements=1000, warmups=200)
                receipt = folder / f"session-{session}.json"
                if collect and not receipt.exists():
                    collect_session(artifacts, folder, entry, rows, images, path, order, identity)
                    print(f"{split} session {session+1}/3: {graph}", flush=True)
                rec = read(receipt)
                raw = folder / f"timings-s{session}.csv.gz"
                replay = read(folder / "replay.json")
                if (rec["identity"] != identity or rec["sha256"] != sha(raw) or
                    len(set(rec["affinity"])) != 4 or (collect and rec["affinity"] != psutil.Process().cpu_affinity()) or
                    rec["runtime"]["precision"] != "f32" or
                    (rec["runtime"]["threads"], rec["runtime"]["streams"]) != (4, 1) or
                    not all(rec[k]["ac_online"] for k in ("before", "after")) or
                    rec["before"]["power_profile"] != rec["after"]["power_profile"] or replay["images"] != len(rows) or
                    replay["graph_sha"] != entry["artifact_sha"] or replay["predictions_sha"] != sha(path) or
                    replay["thresholds"] != entry["thresholds"] or replay["implementation"] != sha(policy_timing.__file__)):
                    raise ValueError(f"Changed timing, graph, live parity or runtime at {receipt}")
                with gzip.open(raw, "rt", encoding="utf-8", newline="") as stream:
                    values = list(csv.DictReader(stream))
                if ([r["image_id"] for r in values] != [rows[i]["id"] for i in order[200:]] or
                    any(int(r["repetition"]) != i or int(r["invocation_order"]) != slot or
                        int(r["session"]) != session or r["boundary"] != "pipeline" for i, r in enumerate(values))):
                    raise ValueError(f"Wrong order at {raw}")
                ns = np.asarray([int(r["elapsed_ns"]) for r in values])
                p95 = float(np.percentile(ns / 1e6, 95))
                if ns.min() <= 0 or abs(rec["summary"]["p95_ms"] - p95) > 1e-10:
                    raise ValueError(f"Wrong measured percentile at {raw}")
                results.append(dict(split=split, graph=graph, seed=entry["seed"], side=entry["side"],
                                    session=session, slot=slot, p95_ms=p95))
    return results


@torch.inference_mode()
def run(data, artifacts, dest, collect):
    if collect:
        pin_cpu(4)
    results, fits, effects = quality(data, artifacts, dest, collect)
    times = timing(data, artifacts, dest, fits, collect)
    for row in results:
        row["p95_ms"] = median(r["p95_ms"] for r in times if (r["split"],r["graph"]) == (row["split"],row["graph"]))
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as name:
        checked = dest if collect else Path(name)
        table(checked / "results.csv", results)
        table(checked / "sessions.csv", times)
        table(checked / "size-effects.csv", effects)
        proof = dict(freeze_sha=sha(dest / "freeze.json"), graphs=len(fits), new_graphs=3,
                     splits=2, sessions=len(times), timed_calls=1000*len(times),
                     results_sha=sha(checked / "results.csv"), sessions_sha=sha(checked / "sessions.csv"),
                     effects_sha=sha(checked / "size-effects.csv"), coco_cells=len(results))
        if collect:
            put(dest / "verification.json", proof)
        elif (read(dest / "verification.json") != proof or sha(dest / "results.csv") != proof["results_sha"]
              or sha(dest / "sessions.csv") != proof["sessions_sha"] or sha(dest/"size-effects.csv") != proof["effects_sha"]):
            raise ValueError("Verified results differ from the published dense-size comparison.")
        print(proof, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["collect", "verify"])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dest = args.output or args.artifacts / "cpu-extension/dense-sizes"
    dest.mkdir(parents=True, exist_ok=True)
    with FileLock(dest / "study.lock", timeout=0):
        run(args.data, args.artifacts, dest, args.action == "collect")
