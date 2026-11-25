"""Count dense candidates reaching live NMS under the two cutoff policies."""
import argparse
import csv
from pathlib import Path
from statistics import mean

from steel.common import read, table
from .benchmark import matrix, name


def collect(artifacts, out):
    rows=[]
    for split in ("val","test"):
        for entry in matrix(out):
            if entry["head"]!="dense":
                continue
            cell=name(entry)
            replay=read(out/"timing"/split/cell/"live-replay.json")
            counts=replay["counts"]
            if replay["verified_images"]!=360 or replay["key"]["cutoff"]!=entry["cutoff"]:
                raise ValueError(f"Missing live postprocessor record for {split}/{cell}")
            shared=None
            if entry["model"] in ("A","B","C","D","E"):
                old=artifacts/"study"/split/cell/"candidates-s0.csv"
                with old.open(newline="",encoding="utf-8") as stream:
                    original=list(csv.DictReader(stream))
                if [r["image_id"] for r in original]!=[r["image_id"] for r in counts]:
                    raise ValueError(f"Candidate image order differs at {old}")
                shared=mean(int(r["after_filter"]) for r in original)
            rows.append(dict(split=split,run=entry["run"],model=entry["model"],seed=entry["seed"],
                runtime=entry["runtime"],new_cutoff=entry["cutoff"],shared_cutoff=entry.get("shared_cutoff"),
                anchors=counts[0]["before_filter"],mean_to_nms=mean(r["after_filter"] for r in counts),
                shared_mean_to_nms=shared,mean_final=mean(r["final"] for r in counts)))
    destination=out/"diagnostics/dense-candidates.csv"
    table(destination,rows)
    return rows


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts",type=Path,default=Path("artifacts"))
    args=parser.parse_args()
    collect(args.artifacts,args.artifacts/"cpu-extension")
