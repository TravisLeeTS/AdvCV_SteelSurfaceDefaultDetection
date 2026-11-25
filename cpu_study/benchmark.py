"""Time the full CPU service path, including filtering and NMS at each frozen cutoff."""
from pathlib import Path
from time import monotonic
import csv
import gzip

from filelock import FileLock, Timeout
import numpy as np
import torch

from steel.bench import Engine, image_order, measure, read_predictions
from steel.common import digest, put, read, replace_file, sha
from steel.data import gray_image
from steel.geometry import prepare
from steel.hardware import ac_status, cpu_status
from steel.ir import checked_bundle
from steel.post import finish
from .deployment import HistoricalEngine, runs
from .head_calibration import iter_head_cutoffs, load_annotations


class NeedsAC(RuntimeError):
    pass


def require_idle_power(out):
    if ac_status() is not True:
        raise NeedsAC("Connect AC power for the registered CPU measurements.")
    for run in runs(out):
        try:
            with FileLock(run / "training.lock", timeout=0):
                pass
        except Timeout as error:
            raise RuntimeError(f"Training is active at {run}.") from error


def name(entry):
    return f"{entry['run']}-{entry['runtime']}-{entry['head']}-480"


def matrix(out):
    existing = list(iter_head_cutoffs(out / "heads"))
    stock = read(out / "stock/freeze.json")["entries"]
    entries = existing + stock
    if len(entries) != 48 or len({name(e) for e in entries}) != len(entries):
        raise ValueError("Expected 42 dual-head and six historical configurations.")
    return entries


def live_engine(artifacts, out, entry):
    if entry["model"].startswith("yolo"):
        engine = HistoricalEngine(out / "runs" / entry["run"])
        artifact = engine.manifest["artifact_sha"]
    else:
        run = artifacts / "runs" / entry["run"]
        checkpoint = sha(run / "best.pt")
        if checkpoint != entry["checkpoint_sha"]:
            raise ValueError(f"Frozen checkpoint differs for {run}")
        artifact = (checked_bundle(run, entry["head"], 480)[1]["bundle_sha"]
                    if entry["runtime"] == "openvino" else checkpoint)
        engine = Engine(run, entry["runtime"], entry["head"], 480, 4)
    if entry["artifact_sha"] != artifact:
        raise ValueError(f"Frozen deployment artifact differs for {name(entry)}")
    return engine


def predictions(artifacts, out, split, entry):
    if entry["model"].startswith("yolo"):
        return out / "stock" / split / entry["run"] / "predictions.jsonl.gz"
    return artifacts / "study" / split / name(entry) / "predictions.jsonl.gz"


def pipeline(engine, gray, entry):
    array, transform = prepare(gray, 480)
    output = engine.forward(engine.tensor(array))
    return finish(output, entry["head"], entry["cutoff"], transform, gray.shape)


def verify_replay(path, source, engine, rows, images, entry):
    target = path / "live-replay.json"
    key = dict(predictions_sha=sha(source), artifact_sha=entry["artifact_sha"], cutoff=entry["cutoff"])
    if target.is_file() and read(target)["key"] == key:
        return
    counts = []
    for row, gray, (image_id, low) in zip(rows, images, read_predictions(source), strict=True):
        current, (total, candidates, kept) = pipeline(engine, gray, entry)
        saved = low[low[:, 4] >= entry["cutoff"]]
        if row["id"] != image_id or current.shape != saved.shape or not np.allclose(current, saved, rtol=0, atol=1e-4):
            raise ValueError(f"Live threshold predictions disagree with the frozen replay: {name(entry)}/{image_id}")
        counts.append(dict(image_id=image_id, before_filter=total, after_filter=candidates, final=kept))
    put(target, dict(key=key, verified_images=len(counts), counts=counts))


def timing_key(out, entry, split, session, slot, affinity):
    return dict(head_freeze=sha(out / "heads/freeze.json"), historical_freeze=sha(out / "stock/freeze.json"),
                implementation=sha(Path(__file__)), artifact=entry["artifact_sha"], cutoff=entry["cutoff"],
                split=split, session=session, invocation=slot, affinity=affinity,
                warmups=200, observations=1000, boundary="decoded image to final boxes")


def is_complete(path, key, session):
    status = path / f"session-{session}.json"
    if not status.exists():
        return False
    record = read(status)
    if record["key"] != key or record["observations_sha"] != sha(path / f"timings-s{session}.csv.gz"):
        raise ValueError(f"Timing evidence differs from its recorded plan in {path}")
    return True


def measure_cell(path, engine, rows, images, entry, order, session, slot, key):
    before = cpu_status()
    if not before["ac_online"]:
        raise NeedsAC("AC is offline before timing.")
    next_check = monotonic() + 1
    def run(i):
        nonlocal next_check
        # The same guard is used for every family. Reject a session if power changes mid-block.
        now = monotonic()
        if now >= next_check:
            if ac_status() is not True:
                raise NeedsAC("AC was disconnected during this timing block.")
            next_check = now + 1
        return pipeline(engine, images[i], entry)
    values, summary = measure(run, order, 200, 1000, "pipeline", rows, session, slot)
    after = cpu_status()
    if not after["ac_online"] or before["power_profile"] != after["power_profile"]:
        raise NeedsAC("Power or the performance profile changed during timing.")
    raw = path / f"timings-s{session}.csv.gz"
    tmp = raw.with_suffix(".tmp")
    with gzip.open(tmp, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    replace_file(tmp, raw)
    put(path / f"session-{session}.json", dict(key=key, summary=summary, observations_sha=sha(raw),
        order_sha=digest(order.tolist()), hardware_before=before, hardware_after=after, runtime=engine.precision))


@torch.inference_mode()
def benchmark(data, artifacts, out, split, affinity):
    require_idle_power(out)
    if split == "test" and not (out / "stock/selection.json").exists():
        raise ValueError("Test timing requires the validation-only baseline decision.")
    entries = matrix(out)
    # Three seeded permutations counterbalance model order and use the same image stream per block.
    rows = load_annotations(data, split)
    images = [gray_image(data, row) for row in rows]
    for session in range(3):
        order = image_order(len(rows), 1200, session)
        permutation = np.random.default_rng(np.random.SeedSequence([20261125, session])).permutation(len(entries))
        for slot, index in enumerate(permutation):
            entry = entries[index]
            path = out / "timing" / split / name(entry)
            path.mkdir(parents=True, exist_ok=True)
            key = timing_key(out, entry, split, session, slot, affinity)
            if is_complete(path, key, session):
                continue
            require_idle_power(out)
            engine = live_engine(artifacts, out, entry)
            source = predictions(artifacts, out, split, entry)
            verify_replay(path, source, engine, rows, images, entry)
            measure_cell(path, engine, rows, images, entry, order, session, slot, key)
            print(f"{split} session {session + 1}/3, path {slot + 1}/{len(entries)}: {name(entry)}", flush=True)
            del engine
