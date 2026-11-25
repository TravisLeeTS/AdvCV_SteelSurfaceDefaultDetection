"""Export the historical detectors as separate FP32 bundles and score the grouped split."""
from pathlib import Path
from tempfile import TemporaryDirectory
import gzip
import json

from filelock import FileLock
import numpy as np
import openvino as ov
import torch

from steel.bench import complexity
from steel.common import CLASSES, SCORE_FLOOR, digest, put, read, replace_file, sha
from steel.data import gray_image
from steel.export import PARITY, compile_cpu, sample_parity, save_ir, validate_parity, validation_sample
from steel.geometry import prepare
from steel.metrics import choose_threshold, events, summarize
from steel.post import finish
from .head_calibration import load_annotations, select_budgets, verify_floor
from .models import WEIGHTS, restore


def runs(out):
    return [out / "runs" / f"{family}-{seed}" for seed in range(3) for family in WEIGHTS]


def identity(run):
    record = read(run / "complete.json")
    checkpoint = sha(run / "best.pt")
    if record["epochs"] != 200 or record["best_sha"] != checkpoint or record["run"] != run.name:
        raise ValueError(f"Incomplete training or changed checkpoint in {run}")
    return checkpoint


def graph_shape(path, family):
    graph = ov.Core().read_model(str(path))
    input_port, = graph.inputs
    output_port, = graph.outputs
    expected = 14175 if family == "yolov5s" else 4725
    if (list(input_port.shape) != [1, 3, 480, 480] or list(output_port.shape) != [1, 10, expected]
            or any(port.element_type.get_type_name() != "f32" for port in (input_port, output_port))):
        raise ValueError(f"Unexpected input, head layout or precision in {path}")
    operations = [op.get_type_name().lower() for op in graph.get_ops()]
    if any("nonmaxsuppression" in op or "nms" in op or op == "topk" for op in operations):
        raise ValueError(f"Dense export contains embedded selection in {path}")
    return dict(input=list(input_port.shape), output=list(output_port.shape),
                precision="FP32", head="dense", integrated_nms=False)


def bundle(run):
    checkpoint = identity(run)
    return run / "exports" / checkpoint / "FP32" / "dense-480"


def reopen(run):
    folder = bundle(run)
    manifest = read(folder / "manifest.json")
    xml = folder / "model.xml"
    key = dict(checkpoint_sha=identity(run), xml_sha=sha(xml), bin_sha=sha(xml.with_suffix(".bin")),
               run=run.name, **graph_shape(xml, run.name.rsplit("-", 1)[0]))
    if (manifest["key"] != key or digest(key) != manifest["artifact_sha"]
            or sha(folder / "parity.json") != manifest["parity_sha"]):
        raise ValueError(f"Bundle identity or parity record changed in {folder}")
    return xml, manifest


@torch.inference_mode()
def export(data, out, run):
    target = bundle(run)
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(target.parent / "dense-480.lock"):
        if target.exists():
            return reopen(run)
        family = run.name.rsplit("-", 1)[0]
        view = restore(run, out)
        example = torch.from_numpy(prepare(gray_image(data, validation_sample(data)[0]), 480)[0])
        cost = complexity(view, example)
        with TemporaryDirectory(dir=target.parent) as temporary:
            stage = Path(temporary) / "complete"
            xml = stage / "model.xml"
            ir = save_ir(view, (example,), xml)
            key = dict(checkpoint_sha=identity(run), xml_sha=ir["xml_sha"], bin_sha=ir["bin_sha"],
                       run=run.name, **graph_shape(xml, family))
            checks, precision = sample_parity(data, validation_sample(data), view, xml, "dense", 480, 4)
            count, matched = validate_parity(checks, xml)
            put(stage / "parity.json", dict(limits=PARITY, matched=matched, compared=count, images=checks))
            put(stage / "manifest.json", dict(key=key, artifact_sha=digest(key), complexity=cost,
                 runtime=precision, constant_types=ir["constant_types"], parity_sha=sha(stage / "parity.json"),
                 scores="objectness times class" if family == "yolov5s" else "class scores"))
            stage.rename(target)
    return reopen(run)


class HistoricalEngine:
    def __init__(self, run):
        xml, self.manifest = reopen(run)
        compiled, self.precision = compile_cpu(xml, 4)
        self.request = compiled.create_infer_request()

    def tensor(self, array):
        return ov.Tensor(array, shared_memory=True)

    def forward(self, tensor):
        return self.request.infer(tensor, share_inputs=True, share_outputs=True)[0]


def evaluate(data, out, run, split, frozen=None):
    if (split == "test") != (frozen is not None):
        raise ValueError("Validation fits the cutoff. Test requires a frozen validation entry.")
    if split == "test" and not (out / "stock" / "selection.json").is_file():
        raise ValueError("Freeze the baseline decision using validation before opening test.")
    rows = load_annotations(data, split)
    destination = out / "stock" / split / run.name
    destination.mkdir(parents=True, exist_ok=True)
    engine = HistoricalEngine(run)
    manifest = engine.manifest
    key = dict(split=split, split_sha=sha(data / f"{split}.json"), artifact_sha=manifest["artifact_sha"],
               checkpoint_sha=manifest["key"]["checkpoint_sha"],
               selection_sha=sha(out / "stock/selection.json") if split == "test" else None)
    archive = destination / "predictions.jsonl.gz"
    result_file = destination / "quality.json"
    if result_file.is_file():
        result = read(result_file)
        if result["key"] != key or result["predictions_sha"] != sha(archive):
            raise ValueError(f"Established stock result differs at {destination}")
        return result
    view = restore(run, out)
    predictions, native = [], []
    scratch = archive.with_suffix(".tmp")
    with gzip.open(scratch, "wt", encoding="utf-8") as stream, torch.inference_mode():
        for row in rows:
            gray = gray_image(data, row)
            array, transform = prepare(gray, 480)
            raw = engine.forward(engine.tensor(array))
            det, counts = finish(raw, "dense", SCORE_FLOOR, transform, gray.shape)
            native_raw = view(torch.from_numpy(array)).numpy()
            native_det, _ = finish(native_raw, "dense", SCORE_FLOOR, transform, gray.shape)
            predictions.append(det)
            native.append(native_det)
            stream.write(json.dumps(dict(id=row["id"],low=det.tolist(),low_counts=counts), separators=(",", ":")) + "\n")
    items = events(rows, predictions)
    if split == "val":
        cutoff = choose_threshold(items)
        alternate = next(v for v in select_budgets(rows, predictions) if v["fp_budget_per_image"] == .1)
        if cutoff != alternate["cutoff"]:
            raise ValueError(f"Independent threshold selectors disagree for {run.name}")
    else:
        if (frozen["checkpoint_sha"], frozen["artifact_sha"]) != (key["checkpoint_sha"], key["artifact_sha"]):
            raise ValueError("Test model and validation freeze differ")
        cutoff = frozen["cutoff"]
    verify_floor(rows, predictions, cutoff)
    q = summarize(items, cutoff)
    ref = summarize(events(rows, native), cutoff)
    class_ap_delta = max(abs(q["classes"][c]["ap50"] - ref["classes"][c]["ap50"]) for c in CLASSES)
    delta = abs(q["map5095"] - ref["map5095"])
    if delta > PARITY["map5095_abs"] or class_ap_delta > PARITY["class_ap50_abs"]:
        raise ValueError(f"Native/IR quality differs for {run.name}/{split}: {delta}, {class_ap_delta}")
    if split == "val" and 10 * sum(v["fp"] for v in q["classes"].values()) > len(rows):
        raise ValueError("Validation exceeds the registered budget")
    replace_file(scratch, archive)
    record = dict(key=key, quality=q, native_quality=ref, predictions_sha=sha(archive),
                  native_ir_gap=dict(map5095=delta, max_class_ap50=class_ap_delta),
                  cutoffs_source="validation", complexity=manifest["complexity"])
    put(result_file, record)
    return record


def stock_freeze(data, out):
    entries = []
    for run in runs(out):
        result = evaluate(data, out, run, "val")
        family, seed = run.name.rsplit("-", 1)
        q, key = result["quality"], result["key"]
        entries.append(dict(run=run.name, model=family, seed=int(seed), head="dense", side=480,
                            runtime="openvino", cutoff=q["threshold"], checkpoint_sha=key["checkpoint_sha"],
                            artifact_sha=key["artifact_sha"], validation_fp=sum(c["fp"] for c in q["classes"].values()),
                            validation_source_sha=sha(out / "stock/val" / run.name / "quality.json")))
    path = out / "stock/freeze.json"
    result = dict(registration_sha=sha(out / "registration.json"), validation_sha=sha(data / "val.json"),
                  original_heads_sha=sha(out / "heads/freeze.json"), entries=entries)
    if path.exists() and read(path) != result:
        raise ValueError("The existing historical cutoff freeze disagrees with validation.")
    if not path.exists():
        put(path, result)
    return entries
