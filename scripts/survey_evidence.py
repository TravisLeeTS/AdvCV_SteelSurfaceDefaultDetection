"""Audit the selected steel detector using its saved boxes and synchronous timings."""
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from steel.common import CLASSES, put, read, sha
from steel.ir import bundle_directory


PRIORITY = ("crazing", "rolled_in_scale")
BUDGETS_MS = (10, 15, 20)
SIZE_BANDS = ("<20 px", "20 to <50 px", ">=50 px")


def overlap(a, b):
    area = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area
    return area / union if union > 0 else 0.


def ap50(events, count):
    ordered = sorted(events, key=lambda item: -item[0])
    hits = np.cumsum([matched for score, matched in ordered])
    if not len(hits):
        return 0.
    envelope = np.maximum.accumulate((hits/np.arange(1,len(hits)+1))[::-1])[::-1]
    indices = np.searchsorted(hits/count, np.linspace(0,1,101))
    return float(np.append(envelope, 0.)[indices].mean())


def group_intervals(groups):
    # Columns are images, false positives, six TP counts and six GT counts.
    totals = np.stack([groups[key] for key in sorted(groups)])
    rng = np.random.default_rng(5122026)
    weights = rng.multinomial(len(totals), np.full(len(totals),1/len(totals)), size=1000)
    sample = weights @ totals
    rates = np.column_stack((sample[:,1]/sample[:,0], sample[:,2:8]/sample[:,8:14]))
    return np.percentile(rates, [2.5,97.5],axis=0).T.tolist()


def operating_quality(data, path, cutoff):
    meta = read(path/"cell.json")
    split = meta["split"]
    source = data/(split+".json")
    truth = read(source)
    reference = read(path/"quality.json")
    predictions = path/"predictions.jsonl.gz"
    if (reference["metadata"]!=meta or meta["cutoff"]!=cutoff or meta["head"]!="one" or sha(predictions)!=reference["predictions_sha"]
            or sha(source)!=read(data/"split.json")["split_files"][split]):
        raise ValueError(f"Saved predictions, annotations or cutoff differ at {path}")
    groups = defaultdict(lambda: np.zeros(14, dtype=np.int64))
    counts = np.zeros((6,3), dtype=np.int64)  # GT, TP, FP
    events = [[] for _ in CLASSES]
    bands = {c: np.zeros((3,2), dtype=np.int64) for c in PRIORITY}
    with gzip.open(predictions, "rt", encoding="utf-8") as stream:
        for row,record in zip(truth,map(json.loads,stream),strict=True):
            if row["id"]!=record["id"] or [d for d in record["low"] if d[4]>=cutoff]!=record["operating"]:
                raise ValueError(f"Operating prediction prefix differs on {row['id']}")
            group = groups[row["group"]]
            group[0] += 1
            targets = list(zip(row["classes"],row["boxes"],strict=True))
            used, detected = set(), set()
            for det in sorted(record["low"],key=lambda d:-d[4]):
                label,score = int(det[5]), det[4]
                choices = [(i,overlap(det,box)) for i,(cls,box) in enumerate(targets) if cls==label and i not in used]
                index,iou = max(choices,key=lambda x:x[1],default=(-1,0.))
                match = iou>=0.5
                events[label].append((score,match))
                if match:
                    used.add(index)
                if score>=cutoff:
                    counts[label,1 if match else 2] += 1
                    group[2+label if match else 1] += 1
                    if match:
                        detected.add(index)
            for i,(label,box) in enumerate(targets):
                counts[label,0] += 1
                group[8+label] += 1
                name = CLASSES[label]
                if name in bands:
                    edge = min(box[2]-box[0],box[3]-box[1])
                    band = 0 if edge<20 else 1 if edge<50 else 2
                    bands[name][band] += [1,int(i in detected)]
    intervals = group_intervals(groups)
    classes = {}
    for c,name in enumerate(CLASSES):
        gt,tp,fp = map(int,counts[c])
        rec = dict(gt=gt,tp=tp,fp=fp,fn=gt-tp,recall=tp/gt,ap50=ap50(events[c],gt),
                   recall_group_interval=intervals[c+1])
        expected = reference["metrics"]["classes"][name]
        for key in ("gt","tp","fp","fn","recall","ap50"):
            if abs(expected[key]-rec[key])>1e-10:
                raise ValueError(f"Independent {key} differs for {split} {name}")
        classes[name] = rec
    size_quality = {name:{band:dict(gt=int(gt),tp=int(tp),recall=float(tp/gt) if gt else None)
                          for band,(gt,tp) in zip(SIZE_BANDS,matrix,strict=True)} for name,matrix in bands.items()}
    return dict(images=len(truth),groups=len(groups),classes=classes,fp=int(counts[:,2].sum()),
                fp_per_image=float(counts[:,2].sum()/len(truth)),fp_per_image_group_interval=intervals[0],
                macro_ap50=float(np.mean([c["ap50"] for c in classes.values()])),box_short_side=size_quality,
                predictions_sha256=sha(predictions),annotations_sha256=sha(source))


def timing_budgets(path):
    sessions = []
    for i in range(3):
        file = path/f"timings-s{i}.csv.gz"
        summary = read(path/f"session-{i}.json")
        if sha(file)!=summary["observations_sha"]:
            raise ValueError(f"Raw timing hash differs for {file}")
        with gzip.open(file, "rt", encoding="utf-8",newline="") as f:
            values = [int(r["elapsed_ns"]) for r in csv.DictReader(f) if r["boundary"]=="pipeline"]
        ms = np.array(values)/1e6
        p50,p95,p99 = np.percentile(ms,[50,95,99])
        official = next(r for r in summary["summaries"] if r["boundary"]=="pipeline")
        if len(ms)!=official["timed"] or not np.allclose([p50,p95],[official["p50_ms"],official["p95_ms"]],atol=1e-10,rtol=0):
            raise ValueError(f"Session percentiles disagree with {file}")
        sessions.append(dict(session=i,n=len(ms),p50_ms=p50,p95_ms=p95,p99_ms=p99,
                             over_budget={str(b):int((ms>b).sum()) for b in BUDGETS_MS},sha256=sha(file)))
    return sessions


def make_survey_evidence(data, study, generated):
    frozen = read(study/"freeze.json")
    selected = frozen["selection"]
    run = f"{selected['model']}0"
    cell = f"{run}-openvino-one-{selected['side']}"
    inventory_file = study/"export-inventory.json"
    if sha(inventory_file) != frozen["export_inventory_sha"]:
        raise ValueError("The OpenVINO inventory has changed since validation")
    inventory = read(inventory_file)
    bundle = bundle_directory(Path("runs")/run, "one", selected["side"], frozen["checkpoints"][run]).as_posix()
    exported = inventory[bundle]
    quality = {}
    for split in ("val","test"):
        path = study/split/cell
        meta = read(path/"cell.json")
        if meta["checkpoint_sha"]!=frozen["checkpoints"][run] or meta["split_sha"]!=frozen["split_sha"]:
            raise ValueError(f"Cell differs from the frozen selection at {path}")
        if meta["export_bundle"] != bundle or meta["artifact_sha"] != exported["bundle_sha"]:
            raise ValueError(f"Cell uses a different OpenVINO export at {path}")
        quality[split] = operating_quality(Path(data),path,frozen["cutoffs"][run])
    timing = timing_budgets(study/"test"/cell)
    evidence = dict(selected_cell=cell,checkpoint_sha256=frozen["checkpoints"][run],cutoff=frozen["cutoffs"][run],
                    export_bundle=bundle,bundle_sha256=exported["bundle_sha"],
                    graph_input=exported["reopened_ir"]["input_shape"],graph_output=exported["reopened_ir"]["output_shape"],
                    study_freeze_sha256=sha(study/"freeze.json"),post_hoc=True,
                    intervals="1,000 acquisition-group resamples, percentile 95%, fixed checkpoint and cutoff; seed 5122026",
                    size_measure="Shorter side of the original 200-pixel image annotation box. Bins split at 10% and 25% of the image side.",
                    budget_measure="Descriptive hypothetical software budgets, elapsed time strictly greater than the listed milliseconds. Repeated synchronous requests on decoded images.",
                    **quality,timing_sessions=timing)
    put(study/"report/survey_evidence.json",evidence)
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\hline",
             r"Defect & GT & TP & FN & FP & AP50 (\%) & Recall (\%) & 95\% interval (\%) \\",r"\hline"]
    for name,v in quality["test"]["classes"].items():
        lo,hi = v["recall_group_interval"]
        lines.append(f"{name.replace('_',' ')} & {v['gt']} & {v['tp']} & {v['fn']} & {v['fp']} & "
                     f"{100*v['ap50']:.1f} & {100*v['recall']:.1f} & [{100*lo:.1f}, {100*hi:.1f}] "+r"\\")
    lines.extend([r"\hline",r"\end{tabular}"])
    (generated/"selected_operating.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    lines = [r"\begin{tabular}{rrrrrr}",r"\hline",
             r"Session & p95 (ms) & p99 (ms) & Over 10 ms & Over 15 ms & Over 20 ms \\",r"\hline"]
    for s in timing:
        values = [str(s["session"]+1),f"{s['p95_ms']:.2f}",f"{s['p99_ms']:.2f}",
                  *(str(s["over_budget"][str(b)]) for b in BUDGETS_MS)]
        lines.append(" & ".join(values)+r" \\")
    lines.extend([r"\hline",r"\end{tabular}"])
    (generated/"latency_budgets.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
