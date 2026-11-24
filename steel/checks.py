import csv
import gzip
import json
from pathlib import Path

import numpy as np

from .common import CLASSES, digest, matrix, put, read, sha
from .bench import image_order, name
from .data import load_rows
from .report import verify_freeze
from .stages import STAGES
from .ir import checked_bundle


def verify_results(data,out,dest,models,seeds):
    data,out,dest=Path(data),Path(out),Path(dest)
    study=out/"study"
    verify_freeze(data,out,models,seeds)
    frozen=read(study/"freeze.json")
    if "numerical_revision" in frozen:
        admission=read(study/"heldout-start.json")
        if (admission["identity"]["freeze_sha"]!=sha(study/"freeze.json") or
                admission["started_utc"]<frozen["frozen_utc"] or admission["identity"]["models"]!=list(models)):
            raise ValueError("Held-out evaluation start does not follow the frozen validation decision")
    groups=[{r["group"] for r in load_rows(data,s)} for s in ("train","val","test")]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)): raise ValueError("Cross-split group")
    for k in seeds:
        starts={read(out/"runs"/f"{m}{k}"/"transfer.json")["paired_initial_sha"] for m in models if m!="E"}
        if len(starts)>1:
            raise ValueError("Paired initialization differs")
    cells=list(matrix(models,seeds))
    inventory=read(study/"export-inventory.json")
    smoke=read(study/"reopened-before-timing.json")
    if [(r["head"],r["input_shape"][-1]) for r in smoke["graphs"]]!=[("one",256),("one",480),("dense",480)]:
        raise ValueError("The production reopening check is incomplete")
    if any(inventory[r["bundle"]]["bundle_sha"]!=r["bundle_sha"] for r in smoke["graphs"]):
        raise ValueError("The production reopening check used different exports")
    for model in ("C","D"):
        if model not in models: continue
        initial=read(out/"audit"/"fusion-initial.json")["results"]
        learned=read(out/"audit"/"fusion-learned.json")["results"]
        for seed in seeds:
            run=f"{model}{seed}"
            if (initial[run]["exactly_equal_output_tensors"]<2 or
                    learned[run]["checkpoint_sha"]!=sha(out/"runs"/run/"best.pt")):
                raise ValueError(f"Missing paired gate checks at {run}")
            if set(learned[run]["branch_weights"])!={"12","15"}:
                raise ValueError(f"Missing learned fusion sites at {run}")
        receipt=read(study/f"reopened-before-timing-{model}0.json")
        if [(g["head"],g["input_shape"][-1]) for g in receipt["graphs"]]!=[("one",256),("one",480),("dense",480)]:
            raise ValueError(f"Incomplete reopened controls for {model}")
        if any(inventory[g["bundle"]]["bundle_sha"]!=g["bundle_sha"] for g in receipt["graphs"]):
            raise ValueError(f"Stale reopened control for {model}")
    for path,entry in inventory.items():
        folder,record=checked_bundle(out/"runs"/f"{entry['model_id']}{entry['seed']}",entry["head"],entry["side"])
        if folder.relative_to(out).as_posix()!=path or record["bundle_sha"]!=entry["bundle_sha"] or record["reopened_ir"]!=entry["reopened_ir"]:
            raise ValueError(f"Final graph differs from its validation inventory: {path}")
    invocation_positions=[]
    for session in range(3):
        order=cells[::-1] if session%2 else cells
        shift=len(order)*session//3
        invocation_positions.append({cell:i for i,cell in enumerate(order[shift:]+order[:shift])})
    timings={}
    diagnostic_hashes={}
    head_observations=0
    count,observations=0,0
    session_count=0
    for split in ("val","test"):
        sample=load_rows(data,split)
        session_orders=[image_order(len(sample),1200,s).tolist() for s in range(3)]
        expected_images=[[sample[i]["id"] for i in order[200:]] for order in session_orders]
        for cell in cells:
            path=study/split/name(cell)
            meta=read(path/"cell.json")
            if cell[2]=="openvino":
                graph=inventory[meta["export_bundle"]]
                if (graph["bundle_sha"],graph["side"],graph["head"],graph["checkpoint_sha"])!=(meta["artifact_sha"],cell[4],cell[3],meta["checkpoint_sha"]):
                    raise ValueError(f"Timing cell points to the wrong IR: {path}")
            m=read(path/"quality.json")
            if m["predictions_sha"]!=sha(path/"predictions.jsonl.gz") or m["metadata"]!=meta:
                raise ValueError(f"Prediction provenance mismatch at {path}")
            c=m["metrics"]["classes"]
            if any(c[n]["tp"]+c[n]["fn"]!=c[n]["gt"] for n in CLASSES):
                raise ValueError(f"Recall denominator mismatch at {path}")
            actual=sum(c[n]["fp"] for n in CLASSES)/m["metrics"]["images"]
            np.testing.assert_allclose(actual,m["metrics"]["fp_per_image"],rtol=0,atol=1e-12)
            trace=read(path/"stage-trace.json")
            if trace["observations_sha"]!=sha(path/"stage-trace.csv"):
                raise ValueError(f"Instrumented stage trace digest mismatch at {path}")
            with (path/"stage-trace.csv").open(newline="",encoding="utf-8") as f:
                stage_rows=list(csv.DictReader(f))
            if [r["image_id"] for r in stage_rows]!=[r["id"] for r in sample]:
                raise ValueError(f"Instrumented stage image coverage differs at {path}")
            for field in STAGES:
                ms=[int(r[field])/1e6 for r in stage_rows]
                np.testing.assert_allclose(np.percentile(ms,[50,95]),[trace["summary"][field]["p50_ms"],trace["summary"][field]["p95_ms"]])
            dense_counts={}
            if cell[3]=="dense":
                suppression=read(path/"suppression_check.json")
                if suppression["sets_checked"]!=2*len(sample):
                    raise ValueError(f"Missing independent suppression checks at {path}")
                with gzip.open(path/"predictions.jsonl.gz","rt",encoding="utf-8") as predictions:
                    for event in map(json.loads,predictions):
                        dense_counts[event["id"]]=tuple(event["operating_counts"])
            p95=[]
            for s in range(3):
                raw=path/f"timings-s{s}.csv.gz"
                session=read(path/f"session-{s}.json")
                if sha(raw)!=session["observations_sha"]: raise ValueError(f"Timing digest mismatch: {raw}")
                if session["invocation_order"]!=invocation_positions[s][cell]:
                    raise ValueError(f"Invocation order differs from the planned session rotation at {path}")
                for status in (session["hardware_before"],session["hardware_after"]):
                    if not status["ac_online"] or status.get("power_profile")!=meta["power_profile"]:
                        raise ValueError(f"Power source or profile changed at {path}, session {s}")
                if session["image_order_sha"]!=digest(session_orders[s]):
                    raise ValueError(f"Unmatched timing sequence at {path}, session {s}")
                with gzip.open(raw,"rt",newline="",encoding="utf-8") as f:
                    data_rows=list(csv.DictReader(f))
                for summary in session["summaries"]:
                    session_count+=1
                    boundary=[v for v in data_rows if v["boundary"]==summary["boundary"]]
                    if [v["image_id"] for v in boundary]!=expected_images[s]:
                        raise ValueError(f"Unmatched timed images at {path}, session {s}, {summary['boundary']}")
                    part=[int(v["elapsed_ns"])/1e6 for v in boundary]
                    if len(part)!=1000 or summary["warmups"]!=200: raise ValueError("Timing budget mismatch")
                    np.testing.assert_allclose([summary["p50_ms"],summary["p95_ms"]],np.percentile(part,[50,95]),rtol=1e-12)
                    if summary["boundary"]=="pipeline":p95.append(float(np.percentile(part,95)))
                observations+=len(data_rows)
                model,seed,runtime,head,side=cell
                if head=="dense":
                    with (path/f"candidates-s{s}.csv").open(newline="",encoding="utf-8") as f:
                        cached={r["image_id"]:tuple(int(r[k]) for k in ("before_filter","after_filter","final"))
                                for r in csv.DictReader(f)}
                    if cached!=dense_counts: raise ValueError(f"Cached NMS candidates disagree with the operating path at {path}")
                if runtime=="openvino" and seed==0 and model in ("A","E") and side==480:
                    stage=read(path/f"head-s{s}.json")
                    raw_head=path/f"head-s{s}.csv"
                    if sha(raw_head)!=stage["sha"]: raise ValueError(f"Head diagnostic digest mismatch at {path}")
                    with raw_head.open(encoding="utf-8",newline="") as f:
                        head_rows=list(csv.DictReader(f))
                        head_ms=[int(row["elapsed_ns"])/1e6 for row in head_rows]
                    if [r["image_id"] for r in head_rows]!=expected_images[s]:
                        raise ValueError("Isolated heads used different image sequences")
                    if len(head_ms)!=1000: raise ValueError("Missing head-stage events")
                    np.testing.assert_allclose(np.percentile(head_ms,[50,95]),[stage["summary"]["p50_ms"],stage["summary"]["p95_ms"]])
                    key=(split,model)
                    if diagnostic_hashes.setdefault(key,stage["features_sha"])!=stage["features_sha"]:
                        raise ValueError("Isolated head inputs differ across branches or sessions")
                    head_observations+=len(head_ms)
                    session_count+=1
            timings[(split,*cell)]=np.array(p95)
            count+=1
    with (dest/"head_comparison.csv").open() as f: pairs=list(csv.DictReader(f))
    with (dest/"resolution_sweep.csv").open() as f: sizes=list(csv.DictReader(f))
    with (dest/"timing_sessions.csv").open() as f: reported_sessions=list(csv.DictReader(f))
    session_keys={(r["split"],r["model_id"],r["seed"],r["runtime"],r["head"],r["side"],r["session"],r["boundary"]) for r in reported_sessions}
    if len(reported_sessions)!=session_count or len(session_keys)!=session_count:
        raise ValueError("Session table is missing or duplicating a timed boundary")
    if len(sizes)!=2*sum(c[2:4]==("openvino","one") for c in cells) or len(pairs)!=2*sum(c[3]=="dense" for c in cells):
        raise ValueError("Report does not cover the intended resolution and head matrices")
    for p in pairs:
        if p["runtime"]=="openvino":
            size=next(r for r in sizes if r["split"]==p["split"] and r["model_id"]==p["model_id"]
                      and r["seed"]==p["seed"] and r["side"]=="480")
            if (size["artifact_sha"],size["export_bundle"])!=(p["one_artifact_sha"],p["one_export_bundle"]):
                raise ValueError("The resolution and head comparisons refer to different 480 one-to-one graphs")
        k=(p["split"],p["model_id"],int(p["seed"]),p["runtime"])
        d=timings[(*k,"dense",480)]
        o=timings[(*k,"one",480)]
        np.testing.assert_allclose(float(p["saving_ms"]),np.median(d-o),rtol=1e-12)
        np.testing.assert_allclose(float(p["saving_percent"]),np.median(100*(d-o)/d),rtol=1e-12)
    put(dest/"verification.json",dict(cells=count,distinct_openvino_bundles=len(inventory),timing_session_rows=session_count,primary_and_nms_observations=observations,head_observations=head_observations,
        resolution_rows=len(sizes),head_rows=len(pairs),checks=["group separation","paired initializations","prediction and timing digests",
        "six-class TP+FN denominators","FP/image","each raw p50/p95","matched image sequences","session rotation and AC profile",
        "instrumented stage trace coverage and p50/p95","cached NMS candidate counts versus deployed operating paths",
        "independent NumPy greedy NMS for every dense image at both score thresholds",
        "reopened fixed shapes and learned graph branches","480 one-to-one bundle reuse","paired signed savings","complete matrix row counts"]))
