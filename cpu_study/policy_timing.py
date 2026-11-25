"""Time frozen policies on the stock YOLO26n graphs and the validation-selected path."""
import argparse
import csv
import ctypes
import gzip
import os
from pathlib import Path
from statistics import median
from time import monotonic

import numpy as np
import psutil
import torch
from filelock import FileLock

from steel.bench import Engine, image_order, measure, read_predictions
from steel.common import MAX_DET, PRE_NMS_CAP, digest, put, read, replace_file, sha, table
from steel.data import gray_image
from steel.geometry import prepare, project
from steel.hardware import ac_status, cpu_status, pin_cpu
from steel.ir import checked_bundle
from steel.post import candidates, finish, nms_inputs, suppress
from .benchmark import require_idle_power
from .head_calibration import load_annotations


def conditions(freeze):
    """The stock size sweep, stock head pair, and primary unified validation choice."""
    for graph, entry in sorted(freeze["fits"].items()):
        selected = f"{entry['model']}/{entry['head']}" == freeze["speed_first"] and entry["side"] == 480
        if entry["model"] != "A" and not selected:
            continue
        for policy, thresholds in entry["thresholds"].items():
            yield dict(graph=graph, policy=policy, thresholds=thresholds, run=entry["run"],
                       model=entry["model"], head=entry["head"], seed=entry["seed"], side=entry["side"],
                       artifact_sha=entry["artifact_sha"], checkpoint_sha=entry["checkpoint_sha"])


def pipeline(engine, gray, entry, cutoffs):
    array, transform = prepare(gray, entry["side"])
    output = engine.forward(engine.tensor(array))
    head = entry["head"]
    if entry["policy"] == "global":
        return finish(output, head, cutoffs[0], transform, gray.shape)[0]
    # These static graphs emit at most 4,725 raw anchors at 480. No pre-NMS
    # candidate can be displaced before applying its class-specific threshold.
    rows, count = candidates(output, head, cutoffs.min())
    if count > PRE_NMS_CAP:
        raise ValueError("This graph needs class filtering before the pre-NMS cap.")
    rows = rows[rows[:, 4] >= cutoffs[rows[:, 5].astype(int)]]
    if head == "dense":
        rows = rows[suppress(nms_inputs(rows)).numpy()]
    rows = rows[:MAX_DET].copy()
    h, w = gray.shape
    rows[:, :4] = np.clip(project(rows[:, :4], transform, inverse=True), 0, [w, h, w, h])
    return rows


def schedule(freeze):
    entries = list(conditions(freeze))
    for session in range(3):
        rng = np.random.default_rng(np.random.SeedSequence([20261125, 224, session]))
        for slot, index in enumerate(rng.permutation(len(entries))):
            yield session, slot, entries[index]


def verify_boxes(path, engine, rows, images, entry, cutoffs, source):
    identity = dict(graph_sha=entry["artifact_sha"], thresholds=entry["thresholds"],
                    predictions_sha=sha(source), implementation=sha(Path(__file__)))
    receipt = path / "replay.json"
    if receipt.is_file() and read(receipt) == dict(**identity, images=len(rows)):
        return
    for row, gray, (name, boxes) in zip(rows, images, read_predictions(source), strict=True):
        emitted = pipeline(engine, gray, entry, cutoffs)
        expected = boxes[boxes[:, 4] >= cutoffs[boxes[:, 5].astype(int)]]
        if row["id"] != name:
            raise ValueError(f"Image order differs for {source}")
        np.testing.assert_allclose(emitted, expected, atol=.0001, rtol=0, err_msg=f"{name}: {entry['graph']}")
    put(receipt, dict(**identity, images=len(rows)))


def live_session(artifacts, path, entry, rows, images, order, session, slot, identity):
    require_idle_power(artifacts / "cpu-extension")
    run = artifacts / "runs" / entry["run"]
    _, graph = checked_bundle(run, entry["head"], entry["side"])
    if graph["bundle_sha"] != entry["artifact_sha"] or sha(run / "best.pt") != entry["checkpoint_sha"]:
        raise ValueError(f"Frozen graph or checkpoint changed for {run}")
    engine = Engine(run, "openvino", entry["head"], entry["side"], 4)
    cutoffs = np.asarray(entry["thresholds"], dtype=np.float32)
    source = artifacts / "study/test" / entry["graph"] / "predictions.jsonl.gz"
    verify_boxes(path, engine, rows, images, entry, cutoffs, source)
    before = cpu_status()
    next_check = monotonic()

    def invoke(i):
        nonlocal next_check
        if monotonic() >= next_check:
            if ac_status() is not True:
                raise RuntimeError("AC power was disconnected during a timing block.")
            next_check = monotonic() + 1
        return pipeline(engine, images[i], entry, cutoffs)

    records, summary = measure(invoke, order, 200, 1000, "pipeline", rows, session, slot)
    after = cpu_status()
    if not after["ac_online"] or before["power_profile"] != after["power_profile"]:
        raise RuntimeError("The CPU power state changed during timing.")
    raw = path / f"timings-s{session}.csv.gz"
    stage = raw.with_suffix(".tmp")
    with gzip.open(stage, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    replace_file(stage, raw)
    put(path / f"session-{session}.json", dict(identity=identity, observations_sha=sha(raw),
        affinity=psutil.Process().cpu_affinity(),
        summary=summary, before=before, after=after, runtime=engine.precision))


@torch.inference_mode()
def run(data, artifacts, output, collect):
    out = artifacts / "cpu-extension"
    frozen_path = out / "policies/freeze.json"
    freeze = read(frozen_path)
    rows = load_annotations(data, "test")
    images = [gray_image(data, r) for r in rows] if collect else None
    hardware = pin_cpu(4) if collect else None
    freeze_sha, code_sha = sha(frozen_path), sha(Path(__file__))
    results, cohorts = [], {}
    for session, slot, entry in schedule(freeze):
        cell = f"{entry['graph']}-{entry['policy']}"
        path = output / cell
        receipt = path / f"session-{session}.json"
        order = image_order(len(rows), 1200, session)
        identity = dict(freeze_sha=freeze_sha, code_sha=code_sha, condition=entry, split="test",
                        session=session, slot=slot, order_sha=digest(order.tolist()))
        if collect and not receipt.exists():
            path.mkdir(parents=True, exist_ok=True)
            live_session(artifacts, path, entry, rows, images, order, session, slot, identity)
            print(f"Completed session {session + 1}/3, path {slot + 1}: {cell}", flush=True)
        rec = read(receipt)
        raw = path / f"timings-s{session}.csv.gz"
        if rec["identity"] != identity or rec["observations_sha"] != sha(raw):
            raise ValueError(f"Timing or source identity differs at {raw}")
        rt = rec["runtime"]
        if ((rt["threads"], rt["streams"], rt["precision"]) != (4, 1, "f32")
                or rec["before"]["power_profile"] != rec["after"]["power_profile"]
                or not rec["before"]["ac_online"] or not rec["after"]["ac_online"]
                or len(set(rec["affinity"])) != 4):
            raise ValueError(f"Unexpected execution conditions at {receipt}")
        if hardware and rec["affinity"] != hardware["affinity"]:
            raise ValueError(f"Active CPU affinity differs at {receipt}")
        replay = read(path / "replay.json")
        if session == 0 and replay != dict(graph_sha=entry["artifact_sha"], thresholds=entry["thresholds"],
                predictions_sha=sha(artifacts / "study/test" / entry["graph"] / "predictions.jsonl.gz"),
                implementation=code_sha, images=len(rows)):
            raise ValueError(f"Live replay receipt differs at {path}")
        with gzip.open(raw, "rt", encoding="utf-8", newline="") as stream:
            values = list(csv.DictReader(stream))
        if ([v["image_id"] for v in values] != [rows[i]["id"] for i in order[200:]]
                or any(int(v["session"]) != session or int(v["repetition"]) != j or
                       int(v["invocation_order"]) != slot for j, v in enumerate(values))):
            raise ValueError(f"Wrong timed image order at {raw}")
        p95 = float(np.percentile([int(v["elapsed_ns"]) / 1e6 for v in values], 95))
        if abs(p95 - rec["summary"]["p95_ms"]) > 1e-9:
            raise ValueError(f"Raw p95 differs at {raw}")
        cohorts.setdefault(cell, []).append(p95)
        if session == 2:
            results.append(dict(model=entry["model"], seed=entry["seed"], head=entry["head"], side=entry["side"],
                                policy=entry["policy"], p95_ms=median(cohorts[cell]),
                                **{f"session{s}_ms": v for s, v in enumerate(cohorts[cell])}))
    results.sort(key=lambda r: (r["model"], r["side"], r["head"], r["policy"], r["seed"]))
    table(output / "timing.csv", results)
    put(output / "verification.json", dict(freeze_sha=freeze_sha, implementation=code_sha,
        conditions=len(results), sessions=3 * len(results), individual_timings=3000 * len(results),
        warmups_per_session=200, verified_live_images_per_condition=len(rows), table_sha=sha(output / "timing.csv")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "verify"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, help="Destination for a separate repeat timing run.")
    args = parser.parse_args()
    destination = args.output or args.artifacts / "cpu-extension/policy-timing"
    destination.mkdir(parents=True, exist_ok=True)
    if args.action == "collect" and os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    with FileLock(destination / "timing.lock", timeout=0):
        run(args.data, args.artifacts, destination, args.action == "collect")
