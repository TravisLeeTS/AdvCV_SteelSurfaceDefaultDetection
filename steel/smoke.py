import copy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from ultralytics.cfg import get_cfg

from .bench import benchmark
from .common import digest, put, read, sha
from .data import gray_image, load_rows, prepare_data
from .export import PARITY, compile_cpu, export_cell, paired_boxes, trace_view
from .geometry import prepare
from .hardware import pin_cpu
from .ir import checked_bundle, inspect_ir
from .models import initial
from .post import finish
from .common import SCORE_FLOOR
from .report import tables
from .train import atom_save, image_batch, packed


@torch.inference_mode()
def reopen_exports(data,out,run,threads,receipt=None):
    requested=(("one",256),("one",480),("dense",480))
    for branch,side in requested:
        export_cell(data,run,branch,side,threads)
    sample=load_rows(data,"val")[0]
    gray=gray_image(data,sample)
    evidence=[]
    for branch,side in requested:
        folder,record=checked_bundle(run,branch,side)
        compiled,effective=compile_cpu(folder/"model.xml",threads)
        if list(compiled.input().shape)!=[1,3,side,side]:
            raise ValueError(f"Compiled input shape mismatch at {folder}")
        x,transform=prepare(gray,side)
        native,trace=trace_view(run/"best.pt",branch,torch.from_numpy(x))
        reference=native(torch.from_numpy(x)).numpy()
        actual=compiled.create_infer_request().infer(x)[0]
        if not np.isfinite(actual).all() or list(actual.shape)!=record["reopened_ir"]["output_shape"]:
            raise ValueError(f"Smoke inference failed on {folder}")
        if branch=="dense":
            np.testing.assert_allclose(actual[:,:4],reference[:,:4],atol=PARITY["raw_dense_box_px"],rtol=0)
            np.testing.assert_allclose(actual[:,4:],reference[:,4:],atol=PARITY["raw_dense_score_abs"],rtol=0)
        a=finish(reference,branch,SCORE_FLOOR,transform,gray.shape)[0]
        b=finish(actual,branch,SCORE_FLOOR,transform,gray.shape)[0]
        parity=paired_boxes(a,b)
        floors=[SCORE_FLOOR]
        if branch=="one" and len(a)==300: floors.append(float(a[:,4].min()))
        if (parity["matches"]<PARITY["min_match_fraction"]*max(len(a),len(b)) or
                any(min(abs(s-v) for v in floors)>PARITY["near_floor_or_topk_score"] for s in parity["missed"]+parity["extra"])):
            raise ValueError(f"Reopened smoke model disagrees with its native branch: {folder}")
        evidence.append(dict(bundle=folder.relative_to(out).as_posix(),bundle_sha=record["bundle_sha"],
            xml_sha=record["ir"]["xml_sha"],bin_sha=record["ir"]["bin_sha"],**record["reopened_ir"],
            executed_native_modules=trace["executed_modules"],effective_precision=effective["precision"],parity=parity))
    if len({r["bundle"] for r in evidence})!=3 or len({r["bundle_sha"] for r in evidence})!=3:
        raise ValueError("Smoke export bundles alias one another")
    rejected=[]
    for index,branch,side in [(0,"one",480),(1,"dense",480)]:
        try: inspect_ir(out/evidence[index]["bundle"]/"model.xml",branch,side)
        except ValueError as exc: rejected.append(str(exc))
        else: raise ValueError("A mismatched graph passed the smoke negative control")
    put(receipt or out/"smoke-study/export-reopen.json",dict(verified_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint_sha=sha(run/"best.pt"),image_id=sample["id"],graphs=evidence,
        wrong_shape_and_head_rejected=rejected,shared_resolution_and_head_480_bundle=evidence[1]["bundle_sha"]))


def cpu_smoke(data,out,weights,threads):
    data,out=Path(data),Path(out)
    if not (data/"split.json").exists(): prepare_data(data,out)
    pin_cpu(threads)
    run=out/"smoke/A0"
    run.mkdir(parents=True,exist_ok=True)
    path=run/"best.pt"
    key=digest(["cpu-smoke-v1",sha(data/"train.json")])
    if not path.exists() or not (run/"complete.json").exists() or read(run/"complete.json")["config_id"]!=key:
        model,transfer=initial("A",0,weights)
        model.args=get_cfg(overrides={"epochs":1,"imgsz":480})
        model.train()
        rows=load_rows(data,"train")
        sample=[r for c in range(6) for r in [x for x in rows if x["stratum"]==c][:2]]
        images=[gray_image(data,r) for r in sample]
        opt=torch.optim.AdamW(model.parameters(),lr=0.001)
        for indices in np.array_split(np.arange(len(sample)),3):
            pixels,classes,boxes,ids,removed=packed(indices,sample,images,0,0)
            batch=dict(img=image_batch(pixels,"cpu"),cls=torch.from_numpy(classes[:,None]),
                       bboxes=torch.from_numpy(boxes),batch_idx=torch.from_numpy(ids))
            loss,_=model.loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.sum().backward()
            opt.step()
        atom_save(path,dict(model_id="A",seed=0,yaml=copy.deepcopy(model.yaml),weights=model.state_dict(),epoch=0,config_id=key))
        put(run/"complete.json",dict(config_id=key,best_sha=sha(path),training_images=[r["id"] for r in sample],
                                   label="CPU smoke-test only",initialization=transfer["source_sha"]))
    reopen_exports(data,out,run,threads)
    selected=["A0-openvino-one-256","A0-openvino-one-480","A0-openvino-dense-480","A0-pytorch-one-480","A0-pytorch-dense-480"]
    benchmark(data,out,"val","smoke",["A"],[0],threads,3,8,1,only=selected)
    tables(out/"smoke-study",out/"smoke-study/report",["A"],[0],["val"])
