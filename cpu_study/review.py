"""Verify the extension from saved predictions and per-image timings, or replay a model live."""
import argparse
import csv
import gzip
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from steel.bench import Engine, image_order, read_predictions
from steel.common import CLASSES, read, sha
from steel.data import gray_image
from steel.export import PARITY, paired_boxes
from steel.geometry import prepare
from steel.hardware import pin_cpu
from steel.metrics import choose_threshold, events, match_image, summarize
from .benchmark import live_engine, matrix, name, pipeline, predictions
from .deployment import reopen, runs
from .head_calibration import freeze_validation, load_annotations
from .models import restore
from .readout import aggregate, stock_rows


def verify_times(out, split, entry, rows, power_profile):
    directory = out / "timing" / split / name(entry)
    scores = []
    for session in range(3):
        raw = directory / f"timings-s{session}.csv.gz"
        meta = read(directory / f"session-{session}.json")
        source = meta["key"]
        if (meta["observations_sha"] != sha(raw) or not meta["hardware_before"]["ac_online"]
                or not meta["hardware_after"]["ac_online"] or source["observations"] != 1000
                or source["cutoff"] != entry["cutoff"] or source["artifact"] != entry["artifact_sha"]
                or any(meta[boundary]["power_profile"] != power_profile for boundary in ("hardware_before","hardware_after"))):
            raise ValueError(f"Identity or power conditions failed at {raw}")
        with gzip.open(raw, "rt", encoding="utf-8", newline="") as f:
            obs = list(csv.DictReader(f))
        order = image_order(len(rows), 1200, session)[200:]
        if ([r["image_id"] for r in obs] != [rows[i]["id"] for i in order]
                or [int(r["repetition"]) for r in obs] != list(range(1000))
                or any(r["boundary"] != "pipeline" or int(r["session"]) != session for r in obs)):
            raise ValueError(f"Raw repetitions or paired image order changed at {raw}")
        ms = np.asarray([int(v["elapsed_ns"]) for v in obs]) / 1e6
        if min(ms) <= 0:
            raise ValueError(f"Nonpositive timing at {raw}")
        p95 = float(np.percentile(ms, 95))
        if abs(meta["summary"]["p95_ms"] - p95) > 1e-10:
            raise ValueError(f"Recomputed percentile differs at {raw}")
        scores.append(p95)
    return scores


def verify(data, artifacts):
    out=artifacts / "cpu-extension"
    receipt=read(out / "complete.json")
    selection=read(out / "stock/selection.json")
    if receipt["selection_sha"] != sha(out / "stock/selection.json") or selection["source_split"] != "val":
        raise ValueError("Frozen family selection differs from the study receipt.")
    for filename, checksum in receipt["tables"].items():
        if sha(out / "tables" / filename) != checksum:
            raise ValueError(f"Changed table: {filename}")
    head_freeze = freeze_validation(data, artifacts, out / "heads")
    if len(head_freeze["entries"]) != 42:
        raise ValueError("The validation head freeze is incomplete.")
    for run in runs(out):
        _, manifest = reopen(run)
        if manifest["key"]["checkpoint_sha"] != read(run / "complete.json")["best_sha"]:
            raise ValueError(f"Changed trained weights at {run}")
    entries = matrix(out)
    reference_session = read(artifacts / "study/val/A0-openvino-one-480/session-0.json")
    power_profile = reference_session["hardware_before"]["power_profile"]
    for split in ("val", "test"):
        rows = load_annotations(data, split)
        with (out / "tables" / f"{split}-paths.csv").open(newline="",encoding="utf-8") as stream:
            published = {(r["run"],r["runtime"],r["head"]):r for r in csv.DictReader(stream)}
        recomputed=[]
        for entry in entries:
            observed = read_predictions(predictions(artifacts,out,split,entry))
            if [r["id"] for r in rows] != [i for i,_ in observed]:
                raise ValueError("Image identity mismatch")
            items=events(rows,[boxes for _,boxes in observed])
            q=summarize(items,entry["cutoff"])
            if split=="val" and (choose_threshold(items) != entry["cutoff"] or q["fp_per_image"] > .1):
                raise ValueError(f"Threshold was not the validation optimum for {name(entry)}.")
            times=verify_times(out,split,entry,rows,power_profile)
            reference=published[entry["run"],entry["runtime"],entry["head"]]
            values=dict(map5095=q["map5095"],fp_per_image=q["fp_per_image"],p95_ms=float(np.median(times)),
                        macro_recall=mean(c["recall"] for c in q["classes"].values()),
                        tp=sum(c["tp"] for c in q["classes"].values()),fp=sum(c["fp"] for c in q["classes"].values()),
                        **{f"recall_{c}":q["classes"][c]["recall"] for c in CLASSES},
                        **{f"session{s}_ms":ms for s,ms in enumerate(times)})
            if any(abs(float(reference[k])-v)>1e-10 for k,v in values.items()):
                raise ValueError(f"Archived results differ for {split}/{name(entry)}")
            recomputed.append(dict(**entry,**values,split=split,
                                   parameters=int(reference["parameters"]),flops=int(reference["flops"])))
        families=aggregate(stock_rows(recomputed))
        with (out / "tables" / f"{split}-families.csv").open(newline="",encoding="utf-8") as stream:
            reported={r["family"]:r for r in csv.DictReader(stream)}
        for f in families:
            for k in ("map5095","macro_recall","fp_per_image","p95_ms","p95_min_session","p95_max_session"):
                if abs(float(reported[f["family"]][k])-f[k])>1e-10:
                    raise ValueError(f"Published stock family value differs for {split}/{f['family']}/{k}")
        if split == "val":
            order=[r["family"] for r in sorted(families,key=lambda r:r["p95_ms"])]
            stock=stock_rows(recomputed)
            reference={r["seed"]:r for r in stock if r["model"]=="A"}
            eligible=[]
            for f in families:
                group=sorted((r for r in stock if r["model"]==f["model"]),key=lambda r:r["seed"])
                passes=True
                for key in ("map5095", *(f"recall_{c}" for c in CLASSES)):
                    diff=[r[key]-reference[r["seed"]][key] for r in group]
                    limit=.01 if key=="map5095" else .02
                    passes &= min(diff[0],mean(diff)) >= -limit - 1e-12
                if passes:
                    eligible.append(f["family"])
            if order!=selection["speed_order"] or next(n for n in order if n in eligible)!=selection["fastest_eligible"]:
                raise ValueError("Recomputed validation speed or quality decision differs.")
        print(f"Verified {split}: {len(entries)} operating paths and {3*len(entries)} timing sessions.",flush=True)
    print("Verified 288,000 individual measurements, 96 operating results and the validation recommendation.")
    from .coco_check import historical_coco
    coco=historical_coco(data,out)
    previous=out/"diagnostics/historical-coco.json"
    if previous.is_file() and read(previous)!=coco:
        raise ValueError("The independent COCO results differ from their published receipt.")
    return coco


@torch.inference_mode()
def replay(data, artifacts, family, seed, head, image_id, split, save=None, runtime="openvino", policy="global", side=480):
    out=artifacts / "cpu-extension"
    model={"YOLO26n":"A","YOLOv10n":"E","YOLOv8n":"yolov8n","YOLOv5s":"yolov5s"}.get(family,family)
    head=head or ("one" if model in ("A","B","C","D","E") else "dense")
    smaller = side == 320
    if smaller and (model,head,runtime,policy) != ("B","dense","openvino","global"):
        raise ValueError("The supplementary 320-pixel replay uses PConv B, OpenVINO dense and its size-specific global cutoff.")
    if policy != "global" and runtime != "openvino":
        raise ValueError("Class-specific thresholds were fitted on the OpenVINO paths.")
    candidates=[e for e in matrix(out) if (e["model"],e["seed"],e["head"])==(model,seed,head)]
    entry=next((e for e in candidates if e["runtime"]==runtime),None)
    parity_only=entry is None and runtime=="pytorch"
    if parity_only:
        entry=next((e for e in candidates if e["runtime"]=="openvino"),None)
    if entry is None:
        raise ValueError(f"No recorded operating path for {family}, seed {seed}, head {head}")
    row=next((r for r in load_annotations(data,split) if r["id"]==image_id),None)
    if row is None:
        raise ValueError(f"{image_id} is absent from {split}")
    native_run=(out if model.startswith("yolo") else artifacts)/"runs"/entry["run"]
    if parity_only and sha(native_run/"best.pt") != entry["checkpoint_sha"]:
        raise ValueError("Native checkpoint and validation freeze have different SHA-256 values.")
    if smaller:
        from steel.ir import checked_bundle
        selected=read(out/"dense-sizes/freeze.json")["fits"][f"B{seed}-openvino-dense-320"]
        _,graph=checked_bundle(native_run,head,side)
        if (selected["artifact_sha"],selected["checkpoint_sha"]) != (graph["bundle_sha"],entry["checkpoint_sha"]):
            raise ValueError("Wrong graph for the 320-pixel dense-size freeze.")
        entry=dict(entry,cutoff=selected["thresholds"][0])
        engine=Engine(native_run,runtime,head,side,4)
    elif parity_only and model.startswith("yolo"):
        view=restore(native_run,out)
        class NativeHistorical:
            tensor=staticmethod(torch.from_numpy)
            def forward(self,tensor):
                return view(tensor).numpy()
        engine=NativeHistorical()
    elif parity_only:
        engine=Engine(native_run,"pytorch",head,480,4)
    else:
        engine=live_engine(artifacts,out,entry)
    gray=gray_image(data,row)
    source_path=(out/"dense-sizes"/split/f"B{seed}-openvino-dense-320"/"predictions.jsonl.gz" if smaller
                 else predictions(artifacts,out,split,entry))
    source=read_predictions(source_path)
    saved=next(low for i,low in source if i==image_id)
    if policy == "global":
        if smaller:
            from steel.post import finish
            array,transform=prepare(gray,side)
            boxes,counts=finish(engine.forward(engine.tensor(array)),head,entry["cutoff"],transform,gray.shape)
        else:
            boxes,counts=pipeline(engine,gray,entry)
        expected=saved[saved[:,4]>=entry["cutoff"]]
        setting=f"Global validation cutoff {entry['cutoff']:.6f}"
        details=f"{counts[0]} raw candidates, {counts[1]} above cutoff, {counts[2]} final boxes."
    else:
        from .policy_timing import pipeline as apply_thresholds
        selected=read(out/"policies/freeze.json")["fits"][name(entry)]
        if selected["artifact_sha"]!=entry["artifact_sha"]:
            raise ValueError("Thresholds belong to another OpenVINO graph.")
        cuts=np.asarray(selected["thresholds"][policy],dtype=np.float32)
        boxes=apply_thresholds(engine,gray,dict(entry,policy=policy),cuts)
        expected=saved[saved[:,4]>=cuts[saved[:,5].astype(int)]]
        setting="Per class, maximum "+("mean recall" if policy=="class_macro" else "minimum recall")
        details=", ".join(f"{c} {t:.4f}" for c,t in zip(CLASSES,cuts,strict=True))+f". {len(boxes)} final boxes."
    if parity_only:
        paired=paired_boxes(boxes,expected)
        boundary=paired["missed"]+paired["extra"]
        if any(abs(s-entry["cutoff"])>PARITY["score_abs"] for s in boundary):
            raise ValueError("Native checkpoint and frozen OpenVINO detections disagree beyond the parity limit.")
        print(f"Native checkpoint replay at the frozen OpenVINO cutoff. Matched boxes: {paired['matches']}. "
              f"Unpaired scores within the cutoff tolerance: {len(boundary)}.")
    else:
        np.testing.assert_allclose(boxes,expected,atol=.0001,rtol=0)
    print(f"{family} seed {seed}, {runtime}, {head}, {side} px, {policy}, {image_id}. {setting}. {details}")
    for x1,y1,x2,y2,score,cls in boxes:
        print(f"  {CLASSES[int(cls)]} {score:.5f} at ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
    hits=int(match_image(row,boxes)[2][:,0].sum())
    print(f"Matched {hits}/{len(row['classes'])} annotated objects at IoU 0.50. Unmatched detections: {len(boxes)-hits}.")
    if save:
        from steel.demo import pair_image
        title=f"{family} | seed {seed} | {runtime} | {head} | {side} px | {image_id}"
        footer=f"{setting} | {hits}/{len(row['classes'])} matched | {len(boxes)-hits} false boxes"
        save.parent.mkdir(parents=True,exist_ok=True)
        pair_image(row,gray,boxes,title,footer).save(save)
        print(save.resolve())
    if not parity_only:
        print(f"Live {runtime} boxes match the archived operating point.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["verify","replay"])
    parser.add_argument("--data",type=Path,default=Path("data"))
    parser.add_argument("--artifacts",type=Path,default=Path("artifacts"))
    parser.add_argument("--family",default="YOLO26n",choices=["YOLO26n","YOLOv10n","YOLOv8n","YOLOv5s","B","C","D"])
    parser.add_argument("--seed",type=int,default=0,choices=[0,1,2])
    parser.add_argument("--head",choices=["one","dense"])
    parser.add_argument("--runtime",choices=["openvino","pytorch"],default="openvino")
    parser.add_argument("--side",type=int,choices=[320,480],default=480)
    parser.add_argument("--policy",choices=["global","class_macro","class_minimum"],default="global")
    parser.add_argument("--image",default="inclusion_11")
    parser.add_argument("--split",default="test",choices=["test","val"])
    parser.add_argument("--save",type=Path,help="Write a PNG with ground truth and live predictions side by side")
    args=parser.parse_args()
    pin_cpu(4)
    if args.action=="verify":
        verify(args.data,args.artifacts)
    else:
        replay(args.data,args.artifacts,args.family,args.seed,args.head,args.image,args.split,args.save,args.runtime,args.policy,args.side)
