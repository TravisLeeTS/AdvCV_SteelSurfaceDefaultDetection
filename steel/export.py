from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

import numpy as np
import openvino as ov
import torch
from filelock import FileLock

from .common import SCORE_FLOOR, digest, put, read, replace_file, sha, source_hash
from .data import gray_image, load_rows
from .geometry import prepare
from .ir import PRECISION, bundle_directory, bundle_id, checked_bundle, inspect_ir
from .metrics import iou
from .models import SingleBranch, restore
from .post import finish

PARITY = dict(box_px=0.15, match_iou=0.98, score_abs=0.001, min_match_fraction=0.99,
              near_floor_or_topk_score=0.0001, map5095_abs=0.001, class_ap50_abs=0.005,
              raw_dense_box_px=0.15, raw_dense_score_abs=0.001)


def validation_sample(data, per_class=2):
    rows = load_rows(data, "val")
    return [r for c in range(6) for r in [x for x in rows if x["stratum"] == c][:per_class]]


def trace_view(checkpoint, branch, tensor):
    model, state = restore(checkpoint)
    head = model.model[-1]
    head.end2end = branch == "one"
    with torch.no_grad():
        reference, _ = model(tensor)
    view = SingleBranch(model, branch).eval()
    visited = []
    handles = []
    for tag in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        for index, level in enumerate(getattr(head, tag)):
            path = f"model.{len(model.model)-1}.{tag}.{index}"
            handles.append(level.register_forward_hook(lambda _m, _a, _o, p=path: visited.append(p)))
    with torch.no_grad():
        result = view(tensor)
    for h in handles:
        h.remove()
    if len(visited) != 6 or any(("one2one_" in p) != (branch == "one") for p in visited):
        raise ValueError(f"Unexpected branch trace {visited}")
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    return view, dict(executed_modules=visited, native_reference_max_abs=float((result-reference).abs().max()),
                      epoch=state["epoch"]+1, model=state["model_id"], seed=state["seed"],
                      selector="fresh unfused state -> explicit pinned Detect.forward_head -> _inference",
                      layout="B x 300 x 6 xyxy/score/class, native top-k" if branch=="one" else
                             "B x 10 x anchors, xywh + six class scores; external single-label class-aware NMS")


def save_ir(module, args, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        module(*args)  # populate static anchors once before tracing
        traced = torch.jit.trace(module, args, strict=False, check_trace=False)
        shapes = [list(t.shape) for t in args]
        graph = ov.convert_model(traced, example_input=args, input=shapes)
    for input_node, shape in zip(graph.inputs, shapes, strict=True):
        if not input_node.partial_shape.is_static or list(input_node.shape) != shape:
            raise ValueError(f"OpenVINO shape differs from {shape}")
    low = {"f16","bf16"}
    constants = Counter(op.get_output_element_type(0).get_type_name() for op in graph.get_ops() if op.get_type_name()=="Constant")
    if low.intersection(constants):
        raise ValueError(f"Unexpected constant types {constants}")
    ov.save_model(graph, path, compress_to_fp16=False)
    return dict(xml_sha=sha(path), bin_sha=sha(path.with_suffix(".bin")), constant_types=dict(constants), shapes=shapes)


def compile_cpu(xml, threads):
    core = ov.Core()
    settings = {"INFERENCE_PRECISION_HINT":"f32", "INFERENCE_NUM_THREADS":str(threads),
                "NUM_STREAMS":"1", "PERFORMANCE_HINT":"LATENCY", "ENABLE_CPU_PINNING":False}
    compiled = core.compile_model(str(xml), "CPU", settings)
    precision = compiled.get_property("INFERENCE_PRECISION_HINT").get_type_name()
    device = compiled.get_property("EXECUTION_DEVICES")
    precisions = Counter()
    for node in compiled.get_runtime_model().get_ops():
        info = node.get_rt_info()
        if "runtimePrecision" in info:
            precisions[str(info["runtimePrecision"].value).lower()] += 1
    if precision != "f32" or not all("CPU" in v for v in device) or not precisions or {"bf16","f16","fp16"}.intersection(precisions):
        raise ValueError(f"Runtime precision/device mismatch: {precision}, {device}, {precisions}")
    return compiled, dict(precision=precision, device=device, runtime_types=dict(precisions),
                          streams=int(compiled.get_property("NUM_STREAMS")),
                          threads=int(compiled.get_property("INFERENCE_NUM_THREADS")))


def paired_boxes(a, b):
    # Compare sets in original pixels. Greedy by native score, with a same-class
    # score constraint to resolve top-k permutations among equal-valued scores.
    a = a[np.argsort(-a[:,4], kind="stable")]
    used = np.zeros(len(b), bool)
    missed, max_err, min_iou = [], 0.0, 1.0
    overlap = iou(a[:,:4], b[:,:4])
    for k, row in enumerate(a):
        delta = np.abs(b[:,4]-row[4])
        close = np.max(np.abs(b[:,:4]-row[:4]), axis=1) <= PARITY["box_px"] if len(b) else np.empty(0,bool)
        ok = (~used) & (b[:,5]==row[5]) & (delta<=PARITY["score_abs"]) & ((overlap[k]>=PARITY["match_iou"]) | close)
        ix = np.flatnonzero(ok)
        if not len(ix):
            missed.append(float(row[4]))
            continue
        j = ix[np.argmax(overlap[k,ix] - delta[ix])]
        used[j] = True
        min_iou = min(min_iou, float(overlap[k,j]))
        max_err = max(max_err, float(delta[j]))
    extra = b[~used,4].tolist()
    return dict(native=len(a), exported=len(b), matches=int(used.sum()), missed=missed, extra=extra,
                max_score_error=max_err, min_iou=min_iou)


def export_cell(data, run, branch, side, threads=4):
    run = Path(run)
    checkpoint = run/"best.pt"
    checkpoint_sha=sha(checkpoint)
    out=bundle_directory(run,branch,side,checkpoint_sha)
    data = Path(data)
    source = dict(checkpoint_sha=checkpoint_sha, side=side, head=branch, precision=PRECISION, parity=PARITY,
                  validation_sha=sha(data/"val.json"), openvino=str(ov.__version__), torch=str(torch.__version__),
                  code_sha=source_hash("models.py","post.py","geometry.py","export.py","ir.py"), threads=threads)
    key = digest(source)
    out.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(out.parent/f".{branch}-{side}.lock"):
        if out.exists():
            _,record=checked_bundle(run,branch,side,checkpoint_sha)
            if record["key"]!=key:
                approvals=read(Path(__file__).with_name("export_compatibility.json"))
                approval=next((a for a in approvals if a["from"]==record["code_sha"] and a["to"]==source["code_sha"]),None)
                identical_settings=all(record.get(k)==v for k,v in source.items() if k!="code_sha")
                if not approval or not identical_settings or record["key"]!=digest(dict(source,code_sha=record["code_sha"])):
                    raise ValueError(f"Stored export settings differ in immutable bundle {out}. Use a separate artifact root.")
                if approval.get("scope")=="external-dense-postprocessing":
                    verify_current_post(data,run,out,record,source)
            return record
        with TemporaryDirectory(dir=out.parent,prefix=f".{branch}-{side}-") as temporary:
            staging=Path(temporary)/"bundle"
            _write_export(data,run,checkpoint,branch,side,threads,source,key,staging)
            replace_file(staging,out)
        record=checked_bundle(run,branch,side,checkpoint_sha)[1]
        print(f"Verified {run.name} {branch} {side}: {record['detections_matched']}/{record['detections_checked']} detections",flush=True)
        return record


def _write_export(data,run,checkpoint,branch,side,threads,source,key,out):
    xml,manifest=out/"model.xml",out/"manifest.json"
    samples = validation_sample(data)
    first, _ = prepare(gray_image(data, samples[0]), side)
    view, trace = trace_view(checkpoint, branch, torch.from_numpy(first))
    ir = save_ir(view, (torch.from_numpy(first),), xml)
    checks, precision = sample_parity(data,samples,view,xml,branch,side,threads)
    out.mkdir(parents=True, exist_ok=True)
    put(out/"parity.json", checks)
    total, matched = validate_parity(checks,xml)
    record = dict(key=key, **source, trace=trace, ir=ir, effective=precision,
                  bundle_sha=bundle_id(source["checkpoint_sha"],branch,side,ir),reopened_ir=inspect_ir(xml,branch,side),
                  detections_checked=total, detections_matched=matched)
    put(manifest, record)
    return record


def sample_parity(data,samples,view,xml,branch,side,threads):
    compiled, precision = compile_cpu(xml, threads)
    request = compiled.create_infer_request()
    checks = []
    for row in samples:
        gray = gray_image(data, row)
        x, t = prepare(gray, side)
        with torch.no_grad():
            a = view(torch.from_numpy(x)).numpy()
        b = request.infer(x)[0]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite native/export output")
        if branch=="dense":
            if np.abs(a[:,:4]-b[:,:4]).max()>PARITY["raw_dense_box_px"] or np.abs(a[:,4:]-b[:,4:]).max()>PARITY["raw_dense_score_abs"]:
                raise ValueError(f"Dense raw parity failed on {row['id']}")
        pa, _ = finish(a, branch, SCORE_FLOOR, t, gray.shape)
        pb, _ = finish(b, branch, SCORE_FLOOR, t, gray.shape)
        check = paired_boxes(pa,pb)
        ties = [SCORE_FLOOR]
        if branch=="one" and len(pa)==300:
            ties.append(float(pa[:,4].min()))
        check["unexplained"] = sum(min(abs(s-v) for v in ties)>PARITY["near_floor_or_topk_score"] for s in check["missed"]+check["extra"])
        checks.append(dict(image=row["id"], **check))
    return checks, precision


def validate_parity(checks,xml):
    total = sum(max(c["native"], c["exported"]) for c in checks)
    matched = sum(c["matches"] for c in checks)
    if not checks or any(c["unexplained"] for c in checks) or (total and matched/total < PARITY["min_match_fraction"]):
        raise ValueError(f"Native/export parity failed at {xml}")
    return total, matched


def verify_current_post(data,run,folder,record,source):
    """Retest the existing XML/BIN and preserve its original publication record."""
    path = run/"diagnostics"/"post-parity"/record["bundle_sha"]/(source["code_sha"]+".json")
    key = digest(source)
    samples = validation_sample(data)
    ids = [s["id"] for s in samples]
    if path.exists():
        saved = read(path)
        if (saved["key"]==key and saved["bundle_sha"]==record["bundle_sha"] and
                saved["evidence_sha"]==digest(saved["checks"]) and [c["image"] for c in saved["checks"]]==ids):
            validate_parity(saved["checks"],folder/"model.xml")
            return
    branch,side = source["head"],source["side"]
    first,_ = prepare(gray_image(data,samples[0]),side)
    view,trace = trace_view(run/"best.pt",branch,torch.from_numpy(first))
    checks,effective = sample_parity(data,samples,view,folder/"model.xml",branch,side,source["threads"])
    total,matched = validate_parity(checks,folder/"model.xml")
    put(path,dict(key=key,bundle_sha=record["bundle_sha"],producer_code=record["code_sha"],source=source,
                  checks=checks,evidence_sha=digest(checks),effective=effective,trace=trace,
                  detections_checked=total,detections_matched=matched))
    print(f"Rechecked {run.name} {branch} {side} on {len(samples)} validation images: {matched}/{total}",flush=True)
