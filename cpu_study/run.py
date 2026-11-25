"""Run or resume the registered extension from training through the CPU readout."""
import argparse
import ctypes
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
from time import sleep

from filelock import FileLock
import psutil

from steel.common import put, read, sha
from steel.hardware import ac_status, pin_cpu
from .benchmark import NeedsAC, benchmark
from .deployment import evaluate, export, reopen, runs, stock_freeze
from .head_calibration import freeze_validation
from .heads import evaluate as evaluate_heads
from .readout import make_tables


def training_active():
    for process in psutil.process_iter(["name","cmdline"]):
        try:
            if process.info["name"].lower().startswith("python") and "cpu_study.train" in (process.info["cmdline"] or []):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            continue
    return False


def record_registration(data, artifacts, out):
    path = out / "registration.json"
    protocol = Path(__file__).with_name("protocol.json")
    configuration = read(protocol)
    key = dict(protocol_sha=sha(protocol), original_freeze_sha=sha(artifacts / "study/freeze.json"),
               split_sha=read(data / "split.json")["split_hash"], prior_test_exposure=configuration["prior_exposure"],
               scope=configuration)
    if path.exists():
        current = read(path)
        if any(current[k] != v for k,v in key.items()):
            raise ValueError("Study registration and the frozen benchmark differ.")
    else:
        put(path, dict(registered_utc=datetime.now(timezone.utc).isoformat(), **key))


def finish_training(data, out, weights):
    # If another invocation is training, consume finished runs as they become available.
    from .train import fit
    while True:
        pending = []
        for run in runs(out):
            if not (run / "complete.json").is_file():
                pending.append(run)
            elif not (out / "stock/val" / run.name / "quality.json").exists():
                export(data, out, run)
                evaluate(data, out, run, "val")
                print(f"Verified validation export and predictions for {run.name}", flush=True)
        if not pending:
            return
        if training_active():
            sleep(30)
            continue
        family, seed = pending[0].name.rsplit("-", 1)
        print(f"Resuming {family}, seed {seed}", flush=True)
        fit(data, out, weights, family, int(seed))


def timed_split(data, artifacts, out, split, affinity):
    announced = False
    while True:
        try:
            benchmark(data, artifacts, out, split, affinity)
            break
        except NeedsAC as problem:
            if not announced:
                print(str(problem), flush=True)
                announced = True
            while ac_status() is not True:
                sleep(30)
    return make_tables(data, artifacts, out, split)


def reuse_training(previous, out):
    if previous.resolve() == out.resolve():
        return
    archive = previous / "vendor/yolov5-v7.0.zip"
    if archive.is_file():
        from .models import V5_ARCHIVE_SHA
        if sha(archive) != V5_ARCHIVE_SHA:
            raise ValueError("The archived YOLOv5 source differs from the pinned release.")
        destination = out / "vendor" / archive.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file() or sha(destination) != V5_ARCHIVE_SHA:
            shutil.copyfile(archive, destination)
    for source in runs(previous):
        xml, manifest = reopen(source)
        target = out / "runs" / source.name
        inputs = [source / f for f in ("best.pt","complete.json","config.json","epochs.csv","transfer.json")]
        inputs += list(xml.parent.glob("*.json")) + [xml, xml.with_suffix(".bin")]
        for path in inputs:
            dest = target / path.relative_to(source)
            if dest.is_file():
                if sha(path) != sha(dest):
                    raise ValueError(f"The new measurement directory contains a different file: {dest}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            staging = dest.with_name(dest.name + ".copying")
            shutil.copyfile(path, staging)
            staging.replace(dest)
        if reopen(target)[1]["artifact_sha"] != manifest["artifact_sha"]:
            raise ValueError(f"Copied deployment differs for {source.name}")


def run_all(data, artifacts, weights, out=None, trained_from=None):
    out = out or artifacts / "cpu-extension"
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(out / "runner.lock", timeout=0):
        record_registration(data, artifacts, out)
        hardware = pin_cpu(4)
        if trained_from:
            reuse_training(trained_from, out)
        freeze_validation(data, artifacts, out / "heads")
        if not (out / "heads/verification.json").is_file():
            evaluate_heads(data, artifacts, out / "heads")
        finish_training(data, out, weights)
        entries = stock_freeze(data, out)
        timed_split(data, artifacts, out, "val", hardware["affinity"])
        for entry in entries:
            evaluate(data, out, out / "runs" / entry["run"], "test", entry)
        timed_split(data, artifacts, out, "test", hardware["affinity"])
        tables = sorted((out / "tables").glob("*.csv"))
        put(out / "complete.json", dict(selection_sha=sha(out / "stock/selection.json"),
            tables={p.name:sha(p) for p in tables}, paths=48, splits=2, sessions_per_path=3,
            training_runs=6, tested_checkpoints=21, completed_utc=datetime.now(timezone.utc).isoformat()))
        print("Completed six historical trainings, both 48-path rotations and the frozen stock comparison.", flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--output", type=Path, help="Separate directory for a new timing run")
    parser.add_argument("--trained-from", type=Path, help="Reuse six trained weights and their static graphs")
    args=parser.parse_args()
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        run_all(args.data,args.artifacts,args.weights,args.output,args.trained_from)
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
