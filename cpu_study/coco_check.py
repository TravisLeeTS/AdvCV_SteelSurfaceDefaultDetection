"""Independently replay AP and operating counts for the six historical checkpoints."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from importlib.metadata import version
from io import StringIO
from pathlib import Path

from scripts.verify_research_metrics import Float32BoxEvaluator, coco_differences, coco_truth, rematch, xywh
from steel.bench import read_predictions
from steel.common import CLASSES, MAX_DET, put, read, sha
from .head_calibration import load_annotations


def historical_coco(data,out):
    checks=[]
    frozen=read(out/"stock/freeze.json")
    for split in ("val","test"):
        rows=load_annotations(data,split)
        with redirect_stdout(StringIO()):
            truth=coco_truth(rows)
        for entry in frozen["entries"]:
            folder=out/"stock"/split/entry["run"]
            source=folder/"predictions.jsonl.gz"
            record=read(folder/"quality.json")
            metrics=record["quality"]
            if (record["predictions_sha"]!=sha(source) or metrics["threshold"]!=entry["cutoff"]
                    or record["key"]["artifact_sha"]!=entry["artifact_sha"]):
                raise ValueError(f"Frozen calibration and predictions disagree at {source}")
            hits=[set() for _ in CLASSES]
            fps=Counter()
            detections=[]
            for i,(row,(image_id,boxes)) in enumerate(zip(rows,read_predictions(source),strict=True),1):
                if row["id"]!=image_id:
                    raise ValueError(f"Image order differs at {source}")
                all_boxes=boxes.tolist()
                rematch(row,[b for b in all_boxes if b[4]>=entry["cutoff"]],hits,fps)
                for b in all_boxes:
                    detections.append(dict(image_id=i,category_id=int(b[5])+1,bbox=xywh(b),xyxy=b[:4],score=b[4]))
            for c,label in enumerate(CLASSES):
                if (len(hits[c]),fps[c])!=(metrics["classes"][label]["tp"],metrics["classes"][label]["fp"]):
                    raise ValueError(f"Independent operating match differs at {source}/{label}")
            with redirect_stdout(StringIO()):
                observed=truth.loadRes(detections)
                standard=coco_differences(truth,observed,metrics,MAX_DET)
                for box in observed.anns.values():
                    box["area"]=max(0.,box["area"])
                aligned=coco_differences(truth,observed,metrics,MAX_DET,Float32BoxEvaluator)
            if max(map(abs,aligned.values()))>1e-9:
                raise ValueError(f"FP32-aligned independent COCO evaluation differs at {source}: {aligned}")
            checks.append(dict(split=split,run=entry["run"],predictions_sha=sha(source),
                               standard_coco_minus_study=standard,fp32_aligned_max_abs_delta=max(map(abs,aligned.values()))))
    receipt=dict(evaluator="pycocotools "+version("pycocotools"),cells=len(checks),class_count_rows=6*len(checks),
                 max_standard_map5095_difference=max(abs(c["standard_coco_minus_study"]["map5095"]) for c in checks),
                 max_standard_difference=max(abs(x) for c in checks for x in c["standard_coco_minus_study"].values()),
                 operating_counts_match=True,results=checks)
    print(f"Independent COCO check: {len(checks)} cells, {receipt['class_count_rows']} class-count rows, "
          f"maximum standard mAP50–95 gap {receipt['max_standard_map5095_difference']:.8f}.",flush=True)
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("data"))
    parser.add_argument("--out",type=Path,default=Path("artifacts/cpu-extension"))
    parser.add_argument("--save",type=Path)
    args=parser.parse_args()
    result=historical_coco(args.data,args.out)
    if args.save:
        put(args.save,result)
