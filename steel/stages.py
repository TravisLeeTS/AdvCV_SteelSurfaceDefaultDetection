"""Instrument one separate pass through each deployed pipeline."""
import gzip
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch

from .bench import Engine, name
from .common import digest, matrix, put, read, sha, source_hash, table
from .data import gray_image, load_rows
from .geometry import prepare
from .post import finish


STAGES=("preprocess_ns","bind_tensor_ns","model_ns","postprocess_ns")


@torch.inference_mode()
def stage_traces(data, out, split, phase, models, seeds, threads, only=None):
    data,out=Path(data),Path(out)
    study=out/("study" if phase=="runs" else phase+"-study")
    sample=load_rows(data,split)
    images=[gray_image(data,row) for row in sample]
    revision=source_hash("stages.py","bench.py","geometry.py","post.py","export.py","models.py")
    for cell in matrix(models,seeds):
        if only and name(cell) not in only:
            continue
        model,seed,runtime,head,side=cell
        path=study/split/name(cell)
        key=digest([revision,sha(path/"cell.json"),sha(path/"quality.json")])
        observations=path/"stage-trace.csv"
        manifest=path/"stage-trace.json"
        if manifest.exists() and observations.exists():
            saved=read(manifest)
            if saved["key"]==key and saved["observations_sha"]==sha(observations):
                continue
        metadata=read(path/"cell.json")
        engine=Engine(out/phase/f"{model}{seed}",runtime,head,side,threads)
        warm,transform=prepare(images[0],side)
        finish(engine.forward(engine.tensor(warm)),head,metadata["cutoff"],transform,images[0].shape)
        traces=[]
        with gzip.open(path/"predictions.jsonl.gz","rt",encoding="utf-8") as stream:
            for row,gray,reference in zip(sample,images,map(json.loads,stream),strict=True):
                if row["id"]!=reference["id"]:
                    raise ValueError("Stage trace and quality images differ")
                start=perf_counter_ns()
                tensor_array,transform=prepare(gray,side)
                ready=perf_counter_ns()
                tensor=engine.tensor(tensor_array)
                bound=perf_counter_ns()
                raw=engine.forward(tensor)
                computed=perf_counter_ns()
                detection,counts=finish(raw,head,metadata["cutoff"],transform,gray.shape)
                end=perf_counter_ns()
                expected=np.asarray(reference["operating"],np.float32).reshape(-1,6)
                if list(counts)!=reference["operating_counts"] or detection.shape!=expected.shape:
                    raise ValueError(f"Stage trace changed the operating counts for {row['id']}")
                np.testing.assert_allclose(detection,expected,rtol=0,atol=0.0001)
                traces.append(dict(image_id=row["id"],preprocess_ns=ready-start,bind_tensor_ns=bound-ready,
                    model_ns=computed-bound,postprocess_ns=end-computed,candidates=counts[0],filtered=counts[1],final=counts[2]))
        table(observations,traces)
        summary={stage:dict(zip(("p50_ms","p95_ms"),np.percentile([r[stage]/1e6 for r in traces],[50,95]),strict=True))
                 for stage in STAGES}
        put(manifest,dict(key=key,observations_sha=sha(observations),images=len(traces),summary=summary,
            protocol="One instrumented split pass with decoded images in memory, outside all primary and isolated timers.",
            boundaries={"preprocess_ns":"channel conversion, resize/letterbox, normalization and array construction",
                "bind_tensor_ns":"bind the runtime input tensor","model_ns":"synchronous branch, decode and native selection",
                "postprocess_ns":"operating filter, external NMS on dense path, final cap and original coordinates"}))
