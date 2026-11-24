"""Replay the registered fusion contrasts from COCO matches, alarms and raw CPU timings."""
import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_research_metrics import Float32BoxEvaluator, coco_truth, rematch, xywh
from steel.common import CLASSES, put, read, sha


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def replay(cell, rows, truth, inverse, weights):
    detections, hits, fps = [], [set() for _ in CLASSES], Counter()
    with gzip.open(cell/"predictions.jsonl.gz", "rt", encoding="utf-8") as stream:
        for image, (row, pred) in enumerate(zip(rows, map(json.loads, stream), strict=True), 1):
            assert row["id"] == pred["id"]
            rematch(row, pred["operating"], hits, fps)
            for box in pred["low"]:
                detections.append(dict(image_id=image, category_id=int(box[5])+1, score=box[4],
                                       xyxy=box[:4], bbox=xywh(box)))
    groups = weights.shape[1]
    image_group = {r["id"]: g for r, g in zip(rows, inverse, strict=True)}
    with redirect_stdout(StringIO()):
        predicted = truth.loadRes(detections)
        # The study counts inverted low-score boxes as unmatched predictions.
        for p in predicted.anns.values():
            p["area"] = max(p["area"], 0)
        evaluator = Float32BoxEvaluator(truth, predicted, "bbox")
        evaluator.params.maxDets = [300]
        evaluator.params.areaRng = [[0, 1e10]]
        evaluator.params.areaRngLbl = ["all"]
        evaluator.evaluate()
    grid = evaluator.params.recThrs
    map_draws = np.zeros(len(weights))
    ap50_draws = np.zeros(len(weights))
    metrics = {}
    for c, name in enumerate(CLASSES):
        gt = np.zeros(groups, dtype=int)
        for row, g in zip(rows, inverse, strict=True):
            gt[g] += row["classes"].count(c)
        denominators = weights @ gt
        tp = np.bincount([image_group[img] for img, _ in hits[c]], minlength=groups)
        metrics[name+"_recall"] = weights @ tp / denominators
        per_image = [x for x in evaluator.evalImgs if x and x["category_id"] == c+1]
        scores = np.concatenate([p["dtScores"] for p in per_image])
        order = np.argsort(-scores, kind="stable")
        flags = np.concatenate([p["dtMatches"].T != 0 for p in per_image])[order]
        image_groups = np.repeat([inverse[p["image_id"]-1] for p in per_image],
                                 [len(p["dtScores"]) for p in per_image])[order]
        # Explicitly duplicate detections by their acquisition-group multiplicity.
        # COCO supplies the per-image matches. This accumulation shares no AP or bootstrap helper with the report.
        for j, mult in enumerate(weights):
            expanded = np.repeat(flags, mult[image_groups], axis=0)
            ranks = np.arange(1, len(expanded)+1)[:, None]
            cumulative = np.cumsum(expanded, axis=0)
            envelope = np.maximum.accumulate((cumulative / ranks)[::-1], axis=0)[::-1]
            envelope = np.vstack([envelope, np.zeros((1,10))])
            recall = cumulative / denominators[j]
            aps = [envelope[np.searchsorted(recall[:, k], grid), k].mean() for k in range(10)]
            map_draws[j] += np.mean(aps) / 6
            ap50_draws[j] += aps[0] / 6
    p95 = []
    for s in range(3):
        with gzip.open(cell/f"timings-s{s}.csv.gz", "rt", newline="", encoding="utf-8") as stream:
            elapsed = [int(r["elapsed_ns"]) / 1e6 for r in csv.DictReader(stream) if r["boundary"] == "pipeline"]
        assert len(elapsed) == 1000
        p95.append(np.percentile(elapsed,95))
    metrics.update(map5095=map_draws, map50=ap50_draws, fp_per_image=np.array([sum(fps.values())/len(rows)]),
                   pipeline_p95_ms=np.array([np.median(p95)]))
    return metrics


def verify(data, artifacts):
    study = artifacts/"study"
    report = study/"report"
    freeze = read(study/"freeze.json")
    if not set("ABCD").issubset(freeze["models"]):
        return
    spec = read(Path(__file__).resolve().parents[1]/"protocol.json")["uncertainty"]
    points, intervals = (csv_rows(report/name) for name in ("fusion_contrasts.csv","fusion_intervals.csv"))
    seen = set()
    interval_seen = set()
    seeds = freeze["seeds"]
    for split in ("val","test"):
        rows = read(data/f"{split}.json")
        groups, inverse = np.unique([r["group"] for r in rows], return_inverse=True)
        rng = np.random.default_rng(spec["seed"])
        weights = rng.multinomial(len(groups), np.full(len(groups),1/len(groups)), size=spec["draws"])
        weights = np.vstack([np.ones(len(groups), dtype=int), weights])
        with redirect_stdout(StringIO()):
            truth = coco_truth(rows)
        cells = {seed: {m: replay(study/split/f"{m}{seed}-openvino-one-480",rows,truth,inverse,weights)
                        for m in "ABCD"} for seed in seeds}
        for metric in cells[seeds[0]]["A"]:
            multiplier = 1 if metric in ("fp_per_image","pipeline_p95_ms") else 100
            by_seed = {}
            for seed in seeds:
                a,b,c,d = [cells[seed][m][metric] for m in "ABCD"]
                by_seed[str(seed)] = {"B-A":b-a,"C-A":c-a,"D-B":d-b,"D-C":d-c,
                                      "D-A":d-a,"(D-B)-(C-A)":(d-b)-(c-a)}
            by_seed["paired mean"] = {key: np.mean([by_seed[str(s)][key] for s in seeds],axis=0)
                                      for key in by_seed[str(seeds[0])]}
            for sid, contrasts in by_seed.items():
                for contrast, draws in contrasts.items():
                    key = (split, contrast, metric, sid)
                    draws = multiplier * draws
                    point = [r for r in points if tuple(r[k] for k in ("split","contrast","metric","seed")) == key]
                    assert len(point) == 1
                    row=point[0]
                    assert row["unit"] == ({"fp_per_image":"FP/image","pipeline_p95_ms":"ms"}.get(metric,"pp"))
                    for m in "ABCD":
                        included=seeds if sid=="paired mean" else [int(sid)]
                        assert row[m+"_checkpoint_sha"].split("|") == [freeze["checkpoints"][f"{m}{s}"] for s in included]
                        assert row[m+"_source"].split("|") == [f"{split}/{m}{s}-openvino-one-480" for s in included]
                    np.testing.assert_allclose(draws[0],float(row["estimate"]),rtol=0,atol=1e-8)
                    seen.add(key)
                    if metric!="map5095" and not metric.endswith("_recall"):
                        continue
                    sel = [r for r in intervals if tuple(r[k] for k in ("split","contrast","metric","seed")) == key]
                    assert len(sel) == 1 and int(sel[0]["groups"]) == len(groups) and int(sel[0]["draws"])==len(draws)-1
                    np.testing.assert_allclose(np.r_[draws[0],np.percentile(draws[1:],[2.5,97.5])],
                                               [float(sel[0][k]) for k in ("estimate","low","high")],rtol=0,atol=1e-8)
                    interval_seen.add(key)
        print(f"Replayed {split} factorial points and all {spec['draws']} paired group draws.",flush=True)
    assert len(seen)==len(points) and len(interval_seen)==len(intervals)
    put(artifacts/"audit/factorial-check.json",dict(contrasts=6,point_rows=len(points),interval_rows=len(intervals),
        draws=spec["draws"],folds=["val","test"],seeds=seeds,freeze_sha=sha(study/"freeze.json"),
        method="COCO per-image FP32 matches, explicit group repetition and 101-point interpolation; independent operating rematch and raw timing percentiles.",
        points_sha=sha(report/"fusion_contrasts.csv"),intervals_sha=sha(report/"fusion_intervals.csv")))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("data"))
    parser.add_argument("--artifacts",type=Path,default=Path("artifacts"))
    args=parser.parse_args()
    verify(args.data,args.artifacts)
