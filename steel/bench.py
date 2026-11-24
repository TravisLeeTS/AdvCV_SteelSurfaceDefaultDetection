import csv
import gzip
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import openvino as ov
import torch

from .common import CLASSES, MAX_DET, MODELS, NMS_IOU, PRE_NMS_CAP, SCORE_FLOOR, digest, matrix, put, read, sha, source_hash, table
from .data import gray_image, load_rows
from .export import PARITY, compile_cpu, export_cell, paired_boxes, save_ir, trace_view
from .geometry import prepare
from .hardware import cpu_status, pin_cpu
from .ir import PRECISION, bundle_directory, checked_bundle
from .metrics import choose_threshold, events, summarize
from .models import SingleBranch, restore
from .post import candidates, finish, nms_inputs, suppress


def name(cell):
    model, seed, runtime, head, side = cell
    return f"{model}{seed}-{runtime}-{head}-{side}"


class Engine:
    def __init__(self, run, runtime, head, side, threads):
        self.native = runtime == "pytorch"
        if self.native:
            net, _ = restore(run/"best.pt")
            self.model = SingleBranch(net, head).eval()
            self.precision = dict(precision="f32", device=["CPU"], threads=threads, streams=1)
        else:
            folder,_=checked_bundle(run,head,side)
            compiled, self.precision = compile_cpu(folder/"model.xml", threads)
            self.request = compiled.create_infer_request()

    def tensor(self, array):
        return torch.from_numpy(array) if self.native else ov.Tensor(array, shared_memory=True)

    def forward(self, tensor):
        if self.native:
            return self.model(tensor).numpy()
        return self.request.infer(tensor, share_inputs=True, share_outputs=True)[0]


def read_predictions(path, key="low"):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [(r["id"], np.asarray(r[key], np.float32).reshape(-1,6)) for r in map(json.loads, f)]


def calibration(data, dest, run, threads):
    path = dest/"thresholds"/(run.name+".json")
    config = dict(checkpoint_sha=sha(run/"best.pt"), val_sha=sha(data/"val.json"),
                  code_sha=source_hash("bench.py","post.py","geometry.py","metrics.py","models.py"),
                  head="native one-to-one", side=480, score_floor=SCORE_FLOOR, fp_budget=0.10,
                  tie="maximum macro recall, then fewer FP, then higher threshold")
    if path.exists() and read(path)["config"] == config:
        return read(path)
    rows = load_rows(data, "val")
    engine = Engine(run, "pytorch", "one", 480, threads)
    pred = []
    for row in rows:
        gray = gray_image(data, row)
        array, trans = prepare(gray, 480)
        raw = engine.forward(engine.tensor(array))
        det, _ = finish(raw, "one", SCORE_FLOOR, trans, gray.shape)
        pred.append(det)
    items = events(rows, pred)
    cutoff = choose_threshold(items)
    quality = summarize(items, cutoff)
    if quality["fp_per_image"] > 0.10:
        raise ValueError("Operating-point calibration exceeded the FP budget")
    result = dict(config=config, cutoff=cutoff, quality=quality)
    put(path, result)
    return result


def reference_suppression(rows):
    """Audit the deployed classwise operator with a global, score-ordered NumPy loop."""
    order = np.argsort(-rows[:, 4], kind="stable")
    areas = (rows[:, 2]-rows[:, 0]) * (rows[:, 3]-rows[:, 1])
    kept = []
    while len(order):
        first, rest = order[0], order[1:]
        kept.append(first)
        same = np.flatnonzero(rows[rest, 5] == rows[first, 5])
        peers = rest[same]
        corners = np.minimum(rows[first, 2:4], rows[peers, 2:4]) - np.maximum(rows[first, :2], rows[peers, :2])
        intersection = np.maximum(corners, 0).prod(axis=1)
        union = areas[first] + areas[peers] - intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(union), where=union>0)
        keep = np.ones(len(rest), dtype=bool)
        keep[same[overlap > NMS_IOU]] = False
        order = rest[keep]
    return np.asarray(kept, dtype=np.int64)


def quality_cell(path, engine, run, cell, rows, images, cutoff, metadata, threads):
    _, _, runtime, head, side = cell
    dest = path/"predictions.jsonl.gz"
    tmp = dest.with_suffix(".tmp")
    predictions, operating, native_predictions = [], [], []
    native = Engine(run, "pytorch", head, side, threads) if runtime=="openvino" else None
    checked_negative = checked_sets = 0
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=3) as log:
        for row, gray in zip(rows, images, strict=True):
            array, trans = prepare(gray, side)
            raw = engine.forward(engine.tensor(array))
            low, low_counts = finish(raw, head, SCORE_FLOOR, trans, gray.shape)
            active, counts = finish(raw, head, cutoff, trans, gray.shape)
            if head=="dense":
                for level in (SCORE_FLOOR,cutoff):
                    cand,_=candidates(raw,head,level)
                    if not np.array_equal(reference_suppression(cand), suppress(nms_inputs(cand)).numpy()):
                        raise ValueError(f"Classwise NMS differs from independent greedy suppression on {row['id']} at {level}")
                    checked_sets += 1
                    checked_negative += int(bool(len(cand)) and cand[:,:4].min()<0)
            log.write(json.dumps(dict(id=row["id"], low=low.tolist(), operating=active.tolist(),
                                      low_counts=low_counts, operating_counts=counts), separators=(",",":"))+"\n")
            predictions.append(low)
            operating.append(active)
            if native:
                ref = native.forward(native.tensor(array))
                native_predictions.append(finish(ref, head, SCORE_FLOOR, trans, gray.shape)[0])
    metrics = summarize(events(rows, predictions), cutoff)
    operating_metrics = summarize(events(rows, operating), cutoff)
    for c in CLASSES:
        if metrics["classes"][c]["tp"] != operating_metrics["classes"][c]["tp"] or metrics["classes"][c]["fp"] != operating_metrics["classes"][c]["fp"]:
            raise ValueError(f"Operating set differs from AP prefix for {c}")
    if native:
        reference = summarize(events(rows, native_predictions), cutoff)
        diffs = {c:metrics["classes"][c]["ap50"]-reference["classes"][c]["ap50"] for c in CLASSES}
        delta = metrics["map5095"]-reference["map5095"]
        put(path/"dataset_parity.json", dict(reference=reference, class_ap50_deltas=diffs, map5095_delta=delta))
        if abs(delta)>PARITY["map5095_abs"] or max(map(abs, diffs.values()))>PARITY["class_ap50_abs"]:
            raise ValueError(f"Dataset parity tolerance exceeded: {path}")
    if head=="dense":
        put(path/"suppression_check.json", dict(sets_checked=checked_sets, negative_edge_sets_compared=checked_negative,
                                               reference="independent score-ordered NumPy NMS on original decoded coordinates"))
    tmp.replace(dest)
    put(path/"quality.json", dict(metadata=metadata, metrics=metrics, predictions_sha=sha(dest)))


def image_order(n, count, session):
    rng = np.random.default_rng(np.random.SeedSequence([2026, session]))
    sequence = rng.permutation(n)
    return np.resize(sequence, count)


def measure(invoke, order, warm, count, boundary, rows, session, invocation):
    for i in order[:warm]:
        invoke(i)
    values = []
    session_start = perf_counter_ns()
    for j, i in enumerate(order[warm:warm+count]):
        begin = perf_counter_ns()
        result = invoke(i)
        elapsed = perf_counter_ns()-begin
        values.append(dict(session=session, invocation_order=invocation, boundary=boundary, repetition=j,
                           image_id=rows[i]["id"], elapsed_ns=elapsed))
        del result
    session_seconds = (perf_counter_ns()-session_start)/1e9
    ms = np.array([r["elapsed_ns"] for r in values]) / 1e6
    summary = dict(boundary=boundary, session=session, timed=count, warmups=warm, p50_ms=float(np.median(ms)),
                   p95_ms=float(np.percentile(ms,95)), fps=count/session_seconds, elapsed_s=session_seconds,
                   timed_work_s=float(ms.sum()/1000))
    return values, summary


def timing_session(path, engine, cell, rows, images, cutoff, session, invocation, warm, count):
    _, _, _, head, side = cell
    # Pre-create native runtime tensors and keep NumPy storage alive through the prepared-input timer.
    arrays = [prepare(gray, side)[0] for gray in images]
    tensors = [engine.tensor(x) for x in arrays]
    order = image_order(len(rows), warm+count, session)
    def model(i):
        return engine.forward(tensors[i])
    def pipeline(i):
        gray = images[i]
        x, trans = prepare(gray, side)
        raw = engine.forward(engine.tensor(x))
        return finish(raw, head, cutoff, trans, gray.shape)
    tasks = [("prepared", model), ("pipeline", pipeline)]
    if head == "dense":
        cached, counts = [], []
        for i in range(len(rows)):
            det, total = candidates(model(i), head, cutoff)
            item = nms_inputs(det)
            kept = len(suppress(item))
            counts.append(dict(image_id=rows[i]["id"], before_filter=total, after_filter=len(det), after_nms=kept,
                               final=min(kept, MAX_DET)))
            cached.append(item)
        table(path/f"candidates-s{session}.csv", counts)
        tasks.append(("nms", lambda i: suppress(cached[i])))
    if session % 2:
        tasks[:2] = reversed(tasks[:2])
    all_values, summaries = [], []
    before = cpu_status()
    for boundary, fn in tasks:
        v, s = measure(fn, order, warm, count, boundary, rows, session, invocation)
        all_values.extend(v)
        summaries.append(s)
    after = cpu_status()
    if not before["ac_online"] or not after["ac_online"] or before.get("power_profile")!=after.get("power_profile"):
        raise RuntimeError("Target AC power or power profile changed during timing")
    raw = path/f"timings-s{session}.csv.gz"
    tmp = raw.with_suffix(".tmp")
    with gzip.open(tmp, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_values[0].keys())
        w.writeheader()
        w.writerows(all_values)
    tmp.replace(raw)
    put(path/f"session-{session}.json", dict(summaries=summaries, observations_sha=sha(raw), invocation_order=invocation,
                                            hardware_before=before, hardware_after=after, image_order_sha=digest(order.tolist())))


def complexity(view, example):
    from torch.utils.flop_counter import FlopCounterMode
    active = {}
    handles = []
    def mark(module, args):
        active.update({id(p):p.numel() for p in module.parameters(recurse=False)})
    for module in view.modules():
        handles.append(module.register_forward_pre_hook(mark))
    with FlopCounterMode(display=False) as ops:
        view(example)
    for h in handles:
        h.remove()
    return dict(active_parameters=sum(active.values()), flops=int(ops.get_total_flops()),
                counted={str(k):int(v) for k,v in ops.get_flop_counts()["Global"].items()},
                convention="2 FLOPs per multiply-accumulate. PyTorch dispatcher convolution and matrix products. "
                           "BN, activations, pooling, routing, decode, top-k and external postprocessing excluded.")


def cached_features(run, split, view, rows, images):
    cache = run/"diagnostics"/split
    cache.mkdir(parents=True,exist_ok=True)
    key = digest([sha(run/"best.pt"),source_hash("geometry.py","models.py"),[(r["id"],r["image_sha"]) for r in rows]])
    meta = cache/"features.json"
    files = [cache/f"p{i}.npy" for i in (3,4,5)]
    if not meta.exists() or read(meta)["key"]!=key:
        maps = []
        for i,gray in enumerate(images):
            x = torch.from_numpy(prepare(gray,480)[0])
            outputs = [v.contiguous().numpy() for v in view.neck(x)]
            if not maps:
                maps = [np.lib.format.open_memmap(f,mode="w+",dtype=np.float32,shape=(len(rows),*v.shape))
                        for f,v in zip(files,outputs,strict=True)]
            for mat,v in zip(maps,outputs,strict=True): mat[i]=v
        for mat in maps: mat.flush()
        put(meta,dict(key=key,hashes={f.name:sha(f) for f in files}))
    record=read(meta)
    if any(sha(f)!=record["hashes"][f.name] for f in files):
        raise ValueError("Cached neck features changed")
    arrays=[np.load(f,mmap_mode="c") for f in files]
    return [tuple(a[i] for a in arrays) for i in range(len(rows))],digest(record["hashes"])


def head_diagnostic(path, run, branch, rows, images, cutoff, session, invocation, warm, count, threads):
    x, _ = prepare(images[0],480)
    view, _ = trace_view(run/"best.pt", branch, torch.from_numpy(x))
    vectors, cache_sha = cached_features(run, path.parent.name, view, rows, images)
    checkpoint_sha = sha(run/"best.pt")
    head_path = run/"diagnostics"/"heads"/checkpoint_sha/PRECISION/f"{branch}-480"/"isolated-head.xml"
    head_manifest = head_path.with_suffix(".json")
    if not head_manifest.exists() or read(head_manifest)["checkpoint_sha"]!=checkpoint_sha:
        record = save_ir(view.head, tuple(torch.from_numpy(v) for v in vectors[0]), head_path)
        put(head_manifest, dict(checkpoint_sha=checkpoint_sha,ir=record))
    compiled, precision = compile_cpu(head_path, threads)
    request = compiled.create_infer_request()
    full = Engine(run, "openvino", branch,480,threads)
    inputs = [[ov.Tensor(v, shared_memory=True) for v in f] for f in vectors]
    threshold_crossings=[]
    low_matches=low_count=0
    def forward(i):
        return request.infer(inputs[i],share_inputs=True,share_outputs=True)[0]
    for i, gray in enumerate(images):
        x, trans = prepare(gray, 480)
        a, b = full.forward(full.tensor(x)), forward(i)
        pa, _ = finish(a,branch,cutoff,trans,gray.shape)
        pb, _ = finish(b,branch,cutoff,trans,gray.shape)
        check = paired_boxes(pa,pb)
        unmatched=check["missed"]+check["extra"]
        if unmatched:
            if any(abs(v-cutoff)>PARITY["near_floor_or_topk_score"] for v in unmatched):
                raise ValueError(f"Isolated head mismatch for {rows[i]['id']}")
            threshold_crossings.append(dict(image_id=rows[i]["id"],cutoff=cutoff,**check))
        if branch=="one":
            low_a=finish(a,branch,SCORE_FLOOR,trans,gray.shape)[0]
            low_b=finish(b,branch,SCORE_FLOOR,trans,gray.shape)[0]
            low=paired_boxes(low_a,low_b)
            ties=[SCORE_FLOOR]
            if len(low_a)==MAX_DET: ties.append(float(low_a[:,4].min()))
            if any(min(abs(v-t) for t in ties)>PARITY["near_floor_or_topk_score"] for v in low["extra"]+low["missed"]):
                raise ValueError(f"Isolated one-to-one low-floor parity failed on {rows[i]['id']}")
            low_matches+=low["matches"]
            low_count+=max(len(low_a),len(low_b))
        if branch=="dense" and (np.abs(a[:,:4]-b[:,:4]).max()>PARITY["raw_dense_box_px"] or
                                np.abs(a[:,4:]-b[:,4:]).max()>PARITY["score_abs"]):
            raise ValueError("Isolated dense head raw outputs differ")
    if low_count and low_matches/low_count<PARITY["min_match_fraction"]:
        raise ValueError("Isolated head low-floor detection agreement is below the declared tolerance")
    obs, summary = measure(forward,image_order(len(rows),warm+count,session),warm,count,"head",rows,session,invocation)
    dest = path/f"head-s{session}.csv"
    table(dest,obs)
    put(path/f"head-s{session}.json",dict(summary=summary, sha=sha(dest), precision=precision, checkpoint_sha=checkpoint_sha,
        feature_source="unfused PyTorch FP32 neck on identical decoded images; deterministic cache",
        features_sha=cache_sha, included="head + decode" + (" + top-k" if branch=="one" else ""),
        operating_threshold_crossings=threshold_crossings, low_floor_matched=low_matches, low_floor_checked=low_count))


@torch.inference_mode()
def benchmark(data, out, split, phase="runs", models=MODELS, seeds=(0,1,2),
              threads=4, warm=200, count=1000, sessions=3, only=None):
    data, out = Path(data), Path(out)
    dest = out/("study" if phase=="runs" else phase+"-study")
    hardware = pin_cpu(threads)
    if phase=="runs" and (warm,count,sessions)!=(200,1000,3):
        raise ValueError("The full study requires the frozen 200/1000/3 timing design")
    if split not in ("val","test") or (split=="test" and phase!="runs"):
        raise ValueError("Test evaluation is reserved for the frozen full study")
    if split=="test":
        from .report import verify_freeze
        verify_freeze(data,out,models,seeds)
        locked_inventory=read(dest/"export-inventory.json")
        admission=dest/"heldout-start.json"
        identity=dict(freeze_sha=sha(dest/"freeze.json"),inventory_sha=sha(dest/"export-inventory.json"),
                      models=list(models),seeds=list(seeds))
        if admission.exists():
            if read(admission)["identity"]!=identity:
                raise ValueError("The held-out evaluation was opened under a different validation freeze")
        else:
            put(admission,dict(started_utc=datetime.now(timezone.utc).isoformat(),identity=identity,
                boundary="after validation freeze verification, before reading test images or evaluating any test cell"))
    design = list(matrix(models,seeds))
    cells = [c for c in design if only is None or name(c) in only]
    split_info = read(data/"split.json")
    if sha(data/f"{split}.json")!=split_info["split_files"][split]:
        raise ValueError("Split file SHA mismatch")
    rows = load_rows(data,split)
    if phase=="smoke":
        rows = [r for c in range(6) for r in [a for a in rows if a["stratum"]==c][:2]]
    images = [gray_image(data,r) for r in rows]
    thresholds = {}
    for model,seed in dict.fromkeys((c[0],c[1]) for c in cells):
        run = out/phase/f"{model}{seed}"
        if split=="val":
            thresholds[run.name] = calibration(data,dest,run,threads)["cutoff"]
        else:
            thresholds[run.name] = read(dest/"freeze.json")["cutoffs"][run.name]
    metadata = {}
    bundles = {}
    for cell in cells:
        model,seed,runtime,head,side = cell
        run = out/phase/f"{model}{seed}"
        ex = (checked_bundle(run,head,side)[1] if split=="test" else export_cell(data,run,head,side,threads))
        bundle=bundle_directory(run,head,side,ex["checkpoint_sha"])
        relative_bundle=bundle.relative_to(out).as_posix()
        if runtime=="openvino":
            bundles[relative_bundle]=dict(checkpoint_sha=ex["checkpoint_sha"],model_id=model,seed=seed,
                precision=PRECISION,head=head,side=side,bundle_sha=ex["bundle_sha"],xml_sha=ex["ir"]["xml_sha"],
                bin_sha=ex["ir"]["bin_sha"],reopened_ir=ex["reopened_ir"],
                comparisons=(["resolution","heads"] if head=="one" and side==480 else ["resolution"] if head=="one" else ["heads"]))
            if split=="test" and locked_inventory.get(relative_bundle)!=bundles[relative_bundle]:
                raise ValueError(f"Held-out cell would use a different fixed graph: {relative_bundle}")
        metadata[name(cell)] = dict(split=split,model_id=model,seed=seed,runtime=runtime,precision="FP32",
            head=head,side=side,checkpoint_sha=sha(run/"best.pt"),split_sha=split_info["split_hash"],
            artifact_sha=ex["bundle_sha"] if runtime=="openvino" else sha(run/"best.pt"),
            export_bundle=relative_bundle if runtime=="openvino" else None,
            graph_input=ex["reopened_ir"]["input_shape"] if runtime=="openvino" else None,
            graph_output=ex["reopened_ir"]["output_shape"] if runtime=="openvino" else None,
            xml_sha=ex["ir"]["xml_sha"] if runtime=="openvino" else None,
            bin_sha=ex["ir"]["bin_sha"] if runtime=="openvino" else None,
            cutoff=thresholds[run.name],floor=SCORE_FLOOR,nms_iou=NMS_IOU if head=="dense" else None,
            cap=MAX_DET,pre_nms_cap=PRE_NMS_CAP,single_label_dense=True,streams=1,batch=1,
            threads=threads,affinity=hardware["affinity"],power_profile=hardware["power_profile"],
            sessions=sessions,warm=warm,timed=count,phase=phase,code_sha=source_hash("bench.py","export.py","ir.py","geometry.py","metrics.py","post.py","models.py"))
    if only is None:
        if len(bundles)!=5*len(models)*len(seeds) or len({graph["bundle_sha"] for graph in bundles.values()})!=len(bundles):
            raise ValueError("The size/head sweep does not contain five distinct fixed exports per checkpoint")
        inventory=dest/"export-inventory.json"
        if inventory.exists():
            if read(inventory)!=bundles:
                raise ValueError(f"The exports differ from the saved inventory at {inventory}")
        elif split=="val":
            put(inventory,bundles)
        else:
            raise ValueError("Missing validation export inventory")
    for s in range(sessions):
        order = design.copy()
        if s%2: order.reverse()
        shift = (len(order)*s//3) % len(order)
        order = order[shift:] + order[:shift]
        for i,cell in enumerate(order):
            if cell not in cells: continue
            model,seed,runtime,head,side = cell
            run = out/phase/f"{model}{seed}"
            path = dest/split/name(cell)
            meta = metadata[name(cell)]
            path.mkdir(parents=True,exist_ok=True)
            config = path/"cell.json"
            if config.exists() and read(config)!=meta:
                raise ValueError(f"Changed cell inputs: {path}; use a fresh artifact root")
            put(config,meta)
            cutoff = thresholds[run.name]
            qual = path/"quality.json"
            session_file = path/f"session-{s}.json"
            diagnostic = runtime=="openvino" and model in ("A","E") and seed==0 and side==480
            if not qual.exists() or read(qual)["predictions_sha"]!=sha(path/"predictions.jsonl.gz"):
                engine = Engine(run,runtime,head,side,threads)
                quality_cell(path,engine,run,cell,rows,images,cutoff,meta,threads)
            if not session_file.exists() or read(session_file)["observations_sha"]!=sha(path/f"timings-s{s}.csv.gz"):
                engine = Engine(run,runtime,head,side,threads)
                timing_session(path,engine,cell,rows,images,cutoff,s,i,warm,count)
                print(f"{phase} {split} {name(cell)} s{s} complete",flush=True)
            if diagnostic and not (path/f"head-s{s}.json").exists():
                head_diagnostic(path,run,head,rows,images,cutoff,s,i,warm,count,threads)
            if side==480 and not (path/"complexity.json").exists():
                net,_ = restore(run/"best.pt")
                view = SingleBranch(net,head).eval()
                put(path/"complexity.json", complexity(view,torch.from_numpy(prepare(images[0],side)[0])))
